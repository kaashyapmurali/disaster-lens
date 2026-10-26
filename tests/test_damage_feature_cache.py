"""Feature-cache integration tests use tiny synthetic images and CPU weights."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw
from filelock import FileLock, Timeout
from torch import nn

from disaster_lens.damage.backbones import SceneBackbone, SceneFeatures
from disaster_lens.damage.features import PairedFeatureExtractor
from disaster_lens.damage.heads import TemporalFusionHead
from disaster_lens.damage.feature_cache import (
    CachedFeatureDataset, build_cache, collate_cached_scenes, main as cache_main,
)
from disaster_lens.data.manifest import DAMAGE_LABELS, FIELDS
from disaster_lens.data.scenes import PairedSceneDataset, collate_pairs


class TinyCacheBackbone(SceneBackbone):
    def trainable_units(self):
        return []

    spatial_channels = 4
    context_dim = 4
    spatial_scale = 1 / 4

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=4, stride=4)
        self.register_buffer("scalar_buffer", torch.tensor(0, dtype=torch.long))

    def forward(self, images: torch.Tensor) -> SceneFeatures:
        spatial = self.conv(images).relu()
        return SceneFeatures(spatial, spatial.mean(dim=(-2, -1)))


class FeatureCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "train.csv"
        self.cache_dir = self.root / "features"
        rows = []
        for index, label in enumerate(DAMAGE_LABELS):
            scene = "scene-a" if index < 3 else "scene-b"
            paths = {}
            for phase, color in (("pre", (50, 100, 150)), ("post", (100, 80, 40))):
                relative = Path(f"raw/train/images/{scene}_{phase}_disaster.png")
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (80, 40), color=color)
                draw = ImageDraw.Draw(image)
                draw.rectangle((8, 5, 24, 20), fill=(220, 210, 200))
                draw.rectangle((35, 8, 60, 30), fill=(20, 200, 70))
                image.save(path)
                paths[phase] = relative.as_posix()
            rows.append({
                "source_split": "train", "scene_id": scene, "building_uid": f"uid-{index}",
                "pre_image": paths["pre"], "post_image": paths["post"],
                "match_status": "matched", "damage_label": label, "raw_damage_label": label,
                "pre_wkt_xy": (
                    f"POLYGON (({10 + index % 3 * 10} 5, {30 + index % 3 * 10} 5, "
                    f"{30 + index % 3 * 10} 15, {10 + index % 3 * 10} 15, {10 + index % 3 * 10} 5))"
                ),
                # This geometry must not be used to localize the target.
                "post_wkt_xy": "POLYGON ((40 20, 60 20, 60 30, 40 30, 40 20))",
            })
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.dataset = PairedSceneDataset(self.manifest, self.root, image_size=64)
        self.extractor = PairedFeatureExtractor(TinyCacheBackbone())

    def test_paired_loader_uses_pre_footprint_and_consistent_uid_order(self) -> None:
        with patch("PIL.Image.open", wraps=Image.open) as opened:
            sample = self.dataset[0]
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(sample.building_uids, ["uid-0", "uid-1", "uid-2"])
        self.assertEqual(sample.labels.tolist(), [0, 1, 2])
        torch.testing.assert_close(sample.pre_boxes[0], torch.tensor([8., 8., 24., 24.]))
        torch.testing.assert_close(sample.pre_boxes, sample.post_boxes)
        self.assertFalse(torch.equal(sample.pre_image, sample.post_image))

    def test_cache_matches_live_features_and_stores_context_once(self) -> None:
        paired = collate_pairs([self.dataset[0], self.dataset[1]])
        expected = self.extractor(paired.pre_images, paired.post_images, paired.pre_boxes, paired.post_boxes)
        with patch.object(self.extractor.backbone, "forward", wraps=self.extractor.backbone.forward) as encoded:
            summary = build_cache(self.dataset, self.extractor, self.cache_dir, batch_size=2)
        self.assertEqual(encoded.call_count, 2)
        self.assertEqual(summary, {"written_scenes": 2, "reused_scenes": 0, "buildings": 4})
        cache = CachedFeatureDataset(self.cache_dir)
        batch = collate_cached_scenes([cache[0], cache[1]])
        self.assertEqual(batch.building_uids, [f"uid-{i}" for i in range(4)])
        self.assertEqual(batch.labels.tolist(), [0, 1, 2, 3])
        for name, value in expected.as_dict().items():
            torch.testing.assert_close(batch.features.as_dict()[name], value)
        shard = cache[0].features
        self.assertEqual(tuple(shard.pre_context.shape), (1, 4))
        for value in shard.as_dict().values():
            # Avoid torch.save accidentally retaining the full batched tensor storage.
            self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())

    def test_complete_cache_reuse_needs_no_image_decoding_or_encoder_passes(self) -> None:
        build_cache(self.dataset, self.extractor, self.cache_dir)
        with patch("PIL.Image.open", side_effect=AssertionError("must not decode images")), \
             patch.object(self.extractor.backbone, "forward", side_effect=AssertionError("must not encode")):
            summary = build_cache(self.dataset, self.extractor, self.cache_dir)
        self.assertEqual(summary["written_scenes"], 0)
        self.assertEqual(summary["reused_scenes"], 2)

    def test_reader_and_head_work_without_source_images(self) -> None:
        build_cache(self.dataset, self.extractor, self.cache_dir)
        for path in (self.root / "raw/train/images").glob("*.png"):
            path.unlink()  # Only synthetic files in this test's temporary directory.
        cache = CachedFeatureDataset(self.cache_dir)
        batch = collate_cached_scenes([cache[0], cache[1]])
        head = TemporalFusionHead(building_dim=cache.building_dim, context_dim=cache.context_dim)
        logits = head(batch.features)
        self.assertEqual(tuple(logits.shape), (4, 4))
        nn.functional.cross_entropy(logits, batch.labels).backward()
        self.assertIsNotNone(head.fusion[0].weight.grad)

    def test_rejects_weight_changes_without_overwriting_cache(self) -> None:
        build_cache(self.dataset, self.extractor, self.cache_dir)
        before = (self.cache_dir / "metadata.json").read_bytes()
        with torch.no_grad():
            self.extractor.backbone.conv.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            build_cache(self.dataset, self.extractor, self.cache_dir)
        self.assertEqual(before, (self.cache_dir / "metadata.json").read_bytes())

    def test_rejects_preprocessing_and_roi_changes(self) -> None:
        build_cache(self.dataset, self.extractor, self.cache_dir)
        resized = PairedSceneDataset(self.manifest, self.root, image_size=96)
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            build_cache(resized, self.extractor, self.cache_dir)
        changed_roi = PairedFeatureExtractor(self.extractor.backbone, roi_size=2)
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            build_cache(self.dataset, changed_roi, self.cache_dir)

    def test_rejects_manifest_and_image_provenance_changes(self) -> None:
        build_cache(self.dataset, self.extractor, self.cache_dir)
        original = self.manifest.read_bytes()
        self.manifest.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Manifest changed since indexing"):
            build_cache(self.dataset, self.extractor, self.cache_dir)
        self.manifest.write_bytes(original)
        path = self.dataset.pre_paths["scene-a"]
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            build_cache(self.dataset, self.extractor, self.cache_dir)

    def test_rejects_changed_building_order(self) -> None:
        build_cache(self.dataset, self.extractor, self.cache_dir)
        self.dataset.scenes[0].buildings.reverse()
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            build_cache(self.dataset, self.extractor, self.cache_dir)

    def test_interrupted_cache_is_not_readable_and_resumes_missing_scenes(self) -> None:
        forward = self.extractor.forward
        calls = 0
        def fail_second_scene(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return forward(*args, **kwargs)
        with patch.object(self.extractor, "forward", side_effect=fail_second_scene):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                build_cache(self.dataset, self.extractor, self.cache_dir)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            CachedFeatureDataset(self.cache_dir)
        with patch.object(self.extractor, "forward", wraps=forward) as encoded:
            summary = build_cache(self.dataset, self.extractor, self.cache_dir)
        self.assertEqual(encoded.call_count, 1)
        self.assertEqual(summary["written_scenes"], 1)
        self.assertEqual(summary["reused_scenes"], 1)
        self.assertEqual(len(CachedFeatureDataset(self.cache_dir)), 2)

    def test_rejects_finetunable_encoder_before_writing_anything(self) -> None:
        trainable = PairedFeatureExtractor(TinyCacheBackbone(), freeze_backbone=False)
        with self.assertRaisesRegex(ValueError, "frozen backbone"):
            build_cache(self.dataset, trainable, self.cache_dir)
        self.assertFalse(self.cache_dir.exists())

    def test_concurrent_writer_is_rejected(self) -> None:
        self.cache_dir.mkdir()
        with FileLock(str(self.cache_dir / ".writer.lock"), timeout=0):
            with self.assertRaises(Timeout):
                build_cache(self.dataset, self.extractor, self.cache_dir)
        self.assertFalse((self.cache_dir / "metadata.json").exists())

    def test_rejects_mismatched_original_image_grids(self) -> None:
        Image.new("RGB", (40, 40)).save(self.dataset.pre_paths["scene-a"])
        with self.assertRaisesRegex(ValueError, "unequal source grids"):
            self.dataset[0]

    def test_detects_corrupted_shard_identity(self) -> None:
        build_cache(self.dataset, self.extractor, self.cache_dir)
        path = self.cache_dir / "scenes/scene-a.pt"
        payload = torch.load(path, weights_only=True)
        payload["cache_id"] = "wrong-cache"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            CachedFeatureDataset(self.cache_dir)[0]

    def test_cache_cli_persists_full_extraction_profile_and_refuses_overwrite(self) -> None:
        profile_path = self.root / "run/metrics.json"
        arguments = [
            "cache_features", "--data-dir", str(self.root), "--split-dir", str(self.root),
            "--cache-dir", str(self.cache_dir), "--splits", "train", "--image-size", "64",
            "--batch-size", "2", "--device", "cpu", "--profile-output", str(profile_path),
        ]
        with patch("sys.argv", arguments), \
             patch("disaster_lens.damage.classifier.ResNet18Backbone", return_value=TinyCacheBackbone()):
            cache_main()
        profile = json.loads(profile_path.read_text())
        self.assertTrue(profile["complete"])
        self.assertEqual(profile["batch_size_scenes"], 2)
        self.assertEqual(profile["splits"]["train"]["cache_summary"]["buildings"], 4)
        batches = profile["splits"]["train"]["batches"]
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["buildings_per_scene"], [3, 1])
        before = profile_path.read_bytes()
        with patch("sys.argv", arguments), self.assertRaises(FileExistsError):
            cache_main()
        self.assertEqual(profile_path.read_bytes(), before)

    def test_configured_normalization_changes_pixels_and_rejects_old_cache_reuse(self) -> None:
        mean, std = (0.1, 0.2, 0.3), (0.2, 0.3, 0.4)
        normalized = PairedSceneDataset(self.manifest, self.root, image_size=64, mean=mean, std=std)
        sample = normalized[0]
        expected = (torch.tensor([50, 100, 150]) / 255 - torch.tensor(mean)) / torch.tensor(std)
        torch.testing.assert_close(sample.pre_image[:, 0, 0], expected)
        self.assertEqual(sample.building_uids, self.dataset[0].building_uids)
        torch.testing.assert_close(sample.pre_boxes, self.dataset[0].pre_boxes)
        self.assertEqual(normalized.preprocessing_config()["mean"], list(mean))
        build_cache(self.dataset, self.extractor, self.cache_dir)
        before = (self.cache_dir / "metadata.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            build_cache(normalized, self.extractor, self.cache_dir)
        self.assertEqual((self.cache_dir / "metadata.json").read_bytes(), before)

    def test_invalid_normalization_is_rejected_before_loading_images(self) -> None:
        for std in ((1., 0., 1.), (1., -1., 1.), (1., float("nan"), 1.)):
            with self.subTest(std=std), self.assertRaisesRegex(ValueError, "normalization"):
                PairedSceneDataset(self.manifest, self.root, image_size=64, std=std)


if __name__ == "__main__":
    unittest.main()
