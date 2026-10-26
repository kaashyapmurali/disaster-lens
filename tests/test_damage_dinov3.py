"""Tiny random native DINOv3 models on synthetic CPU inputs; no weight downloads."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from disaster_lens.damage.features import PairedFeatureExtractor
from disaster_lens.damage.heads import TemporalFusionHead
from disaster_lens.damage.feature_cache import CachedFeatureDataset, build_cache, main as cache_main
from disaster_lens.data.manifest import FIELDS
from disaster_lens.data.scenes import PairedSceneDataset

HAS_TIMM = importlib.util.find_spec("timm") is not None
if HAS_TIMM:
    import timm
    from safetensors.torch import save_file

    from disaster_lens.damage.dinov3 import DINOv3Backbone, SATELLITE_MEAN, SATELLITE_STD, TIMM_MODEL_NAME


@unittest.skipUnless(HAS_TIMM, "Install the vit dependency group for DINOv3 tests.")
class DINOv3BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def make_backbone(self) -> "DINOv3Backbone":
        encoder = timm.create_model(
            TIMM_MODEL_NAME, pretrained=False, img_size=64, embed_dim=32,
            depth=1, num_heads=4, mlp_ratio=2, num_classes=0,
        )
        return DINOv3Backbone(encoder)

    def test_native_tokens_become_rectangular_grid_without_prefix_tokens(self) -> None:
        backbone = self.make_backbone().eval()
        images = torch.randn(2, 3, 64, 96)
        with torch.no_grad():
            tokens = backbone.encoder.forward_features(images)
            features = backbone(images)
        self.assertEqual(tuple(features.spatial_map.shape), (2, 32, 4, 6))
        expected = tokens[:, 5:, :].transpose(1, 2).reshape(2, 32, 4, 6)
        torch.testing.assert_close(features.spatial_map, expected)
        torch.testing.assert_close(features.context, tokens[:, 5:, :].mean(dim=1))

    def test_patch_order_and_context_exclude_cls_and_register_values(self) -> None:
        backbone = self.make_backbone()
        patches = torch.arange(16 * 32, dtype=torch.float32).reshape(1, 16, 32)
        tokens = torch.cat((torch.full((1, 5, 32), -10000.), patches), dim=1)
        with patch.object(backbone.encoder, "forward_features", return_value=tokens):
            features = backbone(torch.zeros(1, 3, 64, 64))
        torch.testing.assert_close(features.spatial_map[0, :, 2, 3], patches[0, 11])
        torch.testing.assert_close(features.context, patches.mean(dim=1))

    def test_native_rope_periods_and_omitted_attention_biases_are_preserved(self) -> None:
        backbone = self.make_backbone().eval()
        periods = backbone.encoder.rope.periods.clone()
        with torch.no_grad():
            backbone(torch.randn(1, 3, 64, 64))
        torch.testing.assert_close(backbone.encoder.rope.periods, periods, rtol=0, atol=0)
        self.assertEqual(backbone.encoder.rope.periods.dtype, torch.float32)
        self.assertEqual(backbone.cache_config()["model_config"]["attention_bias_parameters"], [])

    def test_frozen_paired_features_and_existing_fusion_work_without_encoder_gradients(self) -> None:
        extractor = PairedFeatureExtractor(self.make_backbone(), freeze_backbone=True)
        extractor.train()
        images = torch.randn(2, 3, 64, 64)
        boxes = [torch.tensor([[0., 0., 32., 32.], [16., 16., 48., 48.]]),
                 torch.tensor([[16., 16., 64., 64.]])]
        with patch.object(extractor.backbone.encoder, "forward_features",
                          wraps=extractor.backbone.encoder.forward_features) as encoded:
            features = extractor(images, images, boxes)
        self.assertEqual(encoded.call_count, 2)
        self.assertFalse(extractor.backbone.encoder.training)
        self.assertEqual(tuple(features.pre_building.shape), (3, 288))
        self.assertEqual(tuple(features.pre_context.shape), (2, 32))
        self.assertEqual(features.building_to_scene.tolist(), [0, 0, 1])
        torch.testing.assert_close(features.pre_building, features.post_building)
        head = TemporalFusionHead(building_dim=288, context_dim=32, projected_dim=8, hidden_dim=8)
        torch.nn.functional.cross_entropy(head(features), torch.tensor([0, 1, 3])).backward()
        self.assertTrue(all(parameter.grad is None for parameter in extractor.parameters()))
        self.assertIsNotNone(head.building_projector[0].weight.grad)

    def test_live_paired_features_preserve_gradients_through_both_dates(self) -> None:
        extractor = PairedFeatureExtractor(self.make_backbone(), freeze_backbone=False)
        pre = torch.randn(1, 3, 64, 64, requires_grad=True)
        post = torch.randn(1, 3, 64, 64, requires_grad=True)
        features = extractor(pre, post, [torch.tensor([[0., 0., 48., 48.]])])
        head = TemporalFusionHead(building_dim=288, context_dim=32, projected_dim=8, hidden_dim=8)
        torch.nn.functional.cross_entropy(head(features), torch.tensor([2])).backward()
        self.assertGreater(pre.grad.abs().sum().item(), 0)
        self.assertGreater(post.grad.abs().sum().item(), 0)
        gradient = extractor.backbone.encoder.patch_embed.proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(gradient.isfinite().all())

    def test_invalid_grids_and_output_layout_are_rejected(self) -> None:
        backbone = self.make_backbone()
        for images in (torch.randn(1, 3, 65, 64), torch.randn(1, 1, 64, 64),
                       torch.zeros(1, 3, 64, 64, dtype=torch.uint8)):
            with self.subTest(shape=images.shape, dtype=images.dtype), self.assertRaises(ValueError):
                backbone(images)
        with patch.object(backbone.encoder, "forward_features", return_value=torch.zeros(1, 16, 32)):
            with self.assertRaisesRegex(ValueError, "token layout"):
                backbone(torch.randn(1, 3, 64, 64))

    def test_checkpoint_roundtrip_is_local_only_and_preserves_outputs(self) -> None:
        backbone = self.make_backbone().eval()
        images = torch.randn(1, 3, 64, 64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            save_file(backbone.encoder.state_dict(), path / "model.safetensors")
            replacement = self.make_backbone().encoder
            with patch("disaster_lens.damage.dinov3.timm.create_model", return_value=replacement) as created:
                restored = DINOv3Backbone.from_local_checkpoint(path).eval()
            created.assert_called_once_with(
                TIMM_MODEL_NAME, pretrained=False, num_classes=0, dynamic_img_size=True,
            )
            with torch.no_grad():
                expected, actual = backbone(images), restored(images)
            torch.testing.assert_close(actual.spatial_map, expected.spatial_map)
            torch.testing.assert_close(actual.context, expected.context)
            malformed = dict(backbone.encoder.state_dict())
            malformed.pop("patch_embed.proj.weight")
            save_file(malformed, path / "model.safetensors")
            with patch("disaster_lens.damage.dinov3.timm.create_model", return_value=self.make_backbone().encoder):
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    DINOv3Backbone.from_local_checkpoint(path)

    def test_satellite_normalized_raw_cache_roundtrip_and_dependency_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase in ("pre", "post"):
                Image.new("RGB", (64, 64), color=(64, 96, 128)).save(root / f"{phase}.png")
            manifest = root / "train.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow({
                    "source_split": "train", "match_status": "matched",
                    "scene_id": "scene", "building_uid": "uid", "damage_label": "minor-damage",
                    "pre_image": "pre.png", "post_image": "post.png",
                    "pre_wkt_xy": "POLYGON ((8 8, 40 8, 40 40, 8 40, 8 8))",
                })
            dataset = PairedSceneDataset(
                manifest, root, image_size=64, mean=SATELLITE_MEAN, std=SATELLITE_STD,
            )
            extractor = PairedFeatureExtractor(self.make_backbone(), freeze_backbone=True)
            sample = dataset[0]
            expected = extractor(
                sample.pre_image.unsqueeze(0), sample.post_image.unsqueeze(0), [sample.pre_boxes],
            )
            cache_dir = root / "features"
            build_cache(dataset, extractor, cache_dir)
            cache = CachedFeatureDataset(cache_dir)
            for name, value in expected.as_dict().items():
                torch.testing.assert_close(cache[0].features.as_dict()[name], value)
            spec = cache.metadata["spec"]
            self.assertEqual(spec["preprocessing"]["mean"], list(SATELLITE_MEAN))
            self.assertEqual(spec["backbone_config"]["model_config"]["embed_dim"], 32)
            self.assertEqual(spec["roi"]["spatial_scale"], 1 / 16)
            before = (cache_dir / "metadata.json").read_bytes()
            with patch("disaster_lens.damage.dinov3.timm.__version__", "different-version"):
                with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
                    build_cache(dataset, extractor, cache_dir)
            self.assertEqual((cache_dir / "metadata.json").read_bytes(), before)
            self.assertEqual(json.loads(before)["cache_id"], cache.metadata["cache_id"])
            with torch.no_grad():
                extractor.backbone.encoder.rope.periods.add_(0.01)
            with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
                build_cache(dataset, extractor, cache_dir)
            self.assertEqual((cache_dir / "metadata.json").read_bytes(), before)
            checkpoint_dir = root / "tiny-checkpoint"
            checkpoint_dir.mkdir()
            save_file(extractor.backbone.encoder.state_dict(), checkpoint_dir / "model.safetensors")
            arguments = [
                "cache_features", "--encoder", "dinov3-vitl",
                "--encoder-dir", str(checkpoint_dir), "--cache-dir", str(root / "cli-cache"),
                "--data-dir", str(root), "--split-dir", str(root), "--splits", "train",
                "--image-size", "64", "--device", "cpu",
            ]
            replacement = self.make_backbone().encoder
            with patch("sys.argv", arguments), \
                 patch("disaster_lens.damage.dinov3.timm.create_model", return_value=replacement), \
                 self.assertRaisesRegex(ValueError, "Expected a ViT-L/16"):
                cache_main()
            self.assertFalse((root / "cli-cache").exists())


if __name__ == "__main__":
    unittest.main()
