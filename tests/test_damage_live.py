"""Live damage training/export for every encoder × inputs × trainable-blocks choice; synthetic CPU data only.

Tiny random encoders stand in for the real ones: a 3-block DINOv3-architecture ViT and an untrained ResNet-18.
"""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader

from disaster_lens.damage.backbones import ResNet18Backbone
from disaster_lens.damage.classifier import DamageClassifier, build_head, normalization
from disaster_lens.damage.feature_cache import cache_spec
from disaster_lens.damage.features import PairedFeatureExtractor
from disaster_lens.damage.heads import PostOnlyHead, TemporalFusionHead
from disaster_lens.data.manifest import DAMAGE_LABELS, FIELDS
from disaster_lens.data.scenes import CLASS_TO_INDEX, PairedSceneDataset, collate_pairs

HAS_TIMM = importlib.util.find_spec("timm") is not None
if HAS_TIMM:
    import timm

    from disaster_lens.damage.dinov3 import TIMM_MODEL_NAME, DINOv3Backbone
    from disaster_lens.damage.train_live import (
        LEGACY_FINETUNED, LEGACY_POST_ONLY, _validate_precision, export_live, load_cached_head, load_live_model,
        run_live_epoch, train_live,
    )


def tiny_vit() -> "DINOv3Backbone":
    return DINOv3Backbone(timm.create_model(TIMM_MODEL_NAME, pretrained=False, img_size=64, embed_dim=32, depth=3,
                                            num_heads=4, mlp_ratio=2, num_classes=0))


def tiny_backbone(encoder: str):
    return tiny_vit() if encoder == "dinov3-vitl" else ResNet18Backbone(weights=None)


def write_split_files(root: Path, encoder: str = "dinov3-vitl") -> dict[str, PairedSceneDataset]:
    """Train/validation manifests with four buildings (one per class) each, as in the real split layout."""
    data = {}
    mean, std = normalization(encoder)
    for split in ("train", "validation"):
        rows = []
        for index, label in enumerate(DAMAGE_LABELS):
            scene = f"{split}-{index // 3}"
            for phase, color in (("pre", (64, 96, 128)), ("post", (128 + 30 * index, 96, 32))):
                Image.new("RGB", (64, 64), color=color).save(root / f"{scene}-{phase}.png")
            rows.append({"source_split": "train", "scene_id": scene, "building_uid": f"{split}-uid-{index}",
                         "match_status": "matched", "damage_label": label,
                         "pre_image": f"{scene}-pre.png", "post_image": f"{scene}-post.png",
                         "pre_wkt_xy": "POLYGON ((8 8, 40 8, 40 40, 8 40, 8 8))"})
        with (root / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        data[split] = PairedSceneDataset(root / f"{split}.csv", root, image_size=64, mean=mean, std=std)
    return data


def cached_head_checkpoint(root: Path, data: dict, backbone) -> Path:
    """A cached-feature fusion head trained (here: randomly initialized) on this backbone's features."""
    frozen = PairedFeatureExtractor(backbone, freeze_backbone=True)
    head = TemporalFusionHead(building_dim=288, context_dim=32, projected_dim=8, hidden_dim=8, dropout=0)
    config = {"architecture": "cached-paired-temporal-fusion", "objective": "cross-entropy",
              "class_to_index": CLASS_TO_INDEX,
              "model": {"building_dim": 288, "context_dim": 32, "projected_dim": 8, "hidden_dim": 8, "num_classes": 4, "dropout": 0},
              "caches": {split: {"spec": cache_spec(d, frozen), "cache_id": f"test-{split}"} for split, d in data.items()}}
    path = root / "cached-head.pt"
    torch.save({"epoch": 5, "config": config, "model_state_dict": head.state_dict(),
                "validation": {"macro_f1": 0., "confusion_matrix": [[0] * 4] * 4}}, path)
    return path


@unittest.skipUnless(HAS_TIMM, "Install the vit dependency group.")
class DamageClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def vit_model(self, blocks: int = 1, inputs: str = "paired") -> DamageClassifier:
        backbone = tiny_vit()
        head = build_head(inputs, 288, 32, **({"projected_dim": 8, "hidden_dim": 8, "dropout": 0} if inputs == "paired" else {}))
        return DamageClassifier(backbone, head, inputs=inputs, trainable_blocks=blocks)

    def test_gradients_reach_only_the_trainable_suffix_norm_and_head_from_both_dates(self):
        model = self.vit_model()
        encoder = model.backbone.encoder
        before = {name: value.clone() for name, value in model.state_dict().items()}
        pre, post = torch.randn(1, 3, 64, 64, requires_grad=True), torch.randn(1, 3, 64, 64, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameter_groups(backbone_lr=.001, head_lr=.01))
        loss = torch.nn.functional.cross_entropy(model(pre, post, [torch.tensor([[8., 8., 48., 48.], [16., 16., 64., 64.]])]),
                                                 torch.tensor([1, 3]))
        loss.backward()
        self.assertGreater(pre.grad.abs().sum().item(), 0)
        self.assertGreater(post.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(encoder.blocks[-1].attn.qkv.weight, before["extractor.backbone.encoder.blocks.2.attn.qkv.weight"]))
        self.assertFalse(torch.equal(encoder.norm.weight, before["extractor.backbone.encoder.norm.weight"]))
        self.assertFalse(encoder.training or encoder.blocks[0].training or encoder.patch_embed.training)
        self.assertTrue(encoder.blocks[-1].training)
        self.assertFalse(encoder.cls_token.requires_grad)

    def test_frozen_and_post_only_variants(self):
        frozen = self.vit_model(blocks=0)
        self.assertFalse(any(p.requires_grad for p in frozen.backbone.parameters()))
        self.assertEqual([g["name"] for g in frozen.parameter_groups(backbone_lr=1e-5, head_lr=1e-3)], ["head"])
        post = self.vit_model(blocks=0, inputs="post")
        self.assertIsInstance(post.head, PostOnlyHead)
        pre, image = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        boxes = [torch.tensor([[8., 8., 48., 48.]])]
        with torch.inference_mode():
            torch.testing.assert_close(post(pre, image, boxes), post(torch.zeros_like(pre), image, boxes))   # PRE unused
        resnet = DamageClassifier(ResNet18Backbone(weights=None), build_head("post", 1152, 512), inputs="post", trainable_blocks=2)
        trainable = {name.split(".")[3] for name, p in resnet.named_parameters() if p.requires_grad and name.startswith("extractor")}
        self.assertEqual(trainable, {"global_layers"})                      # layer3 + layer4 (= last two units)
        resnet.train()
        self.assertTrue(all(not m.training for m in resnet.modules() if isinstance(m, torch.nn.BatchNorm2d)))

    def test_invalid_choices_fail_before_training(self):
        backbone = tiny_vit()
        for blocks in (-1, 4):
            with self.subTest(blocks=blocks), self.assertRaisesRegex(ValueError, "trainable_blocks"):
                DamageClassifier(backbone, build_head("post", 288, 32), inputs="post", trainable_blocks=blocks)
        with self.assertRaisesRegex(ValueError, "Head dimensions"):
            DamageClassifier(backbone, TemporalFusionHead(), inputs="paired")
        with self.assertRaisesRegex(ValueError, "inputs"):
            DamageClassifier(backbone, build_head("post", 288, 32), inputs="pre")

    def test_bf16_keeps_roi_align_and_master_parameters_in_float32(self):
        model = self.vit_model()
        from disaster_lens.damage import features
        with patch("disaster_lens.damage.features.roi_align", wraps=features.roi_align) as roi:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64), [torch.tensor([[8., 8., 48., 48.]])])
            torch.nn.functional.cross_entropy(logits.float(), torch.tensor([3])).backward()
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertTrue(all(call.args[0].dtype == torch.float32 for call in roi.call_args_list))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertEqual(model.backbone.encoder.rope.periods.dtype, torch.float32)

    def test_precision_validation(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            _validate_precision("fp16", torch.device("cpu"))
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device"), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Native BF16 is unavailable"):
                _validate_precision("bf16", torch.device("cuda"))


@unittest.skipUnless(HAS_TIMM, "Install the vit dependency group.")
class LiveTrainingTests(unittest.TestCase):
    setUpClass, tearDownClass = DamageClassifierTests.setUpClass, DamageClassifierTests.tearDownClass

    def setUp(self):
        torch.manual_seed(0)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def train(self, encoder, inputs, blocks, output, **kwargs):
        backbone = kwargs.pop("backbone", None) or tiny_backbone(encoder)
        return train_live(encoder, inputs, self.root, self.root, output, trainable_blocks=blocks, image_size=64,
                          epochs=kwargs.pop("epochs", 1), backbone=backbone, **kwargs), backbone

    def test_every_switch_combination_trains_reloads_and_exports(self):
        for encoder in ("resnet18", "dinov3-vitl"):
            write_split_files(self.root, encoder)
            for inputs in ("post", "paired"):
                for blocks in (0, 1):
                    with self.subTest(encoder=encoder, inputs=inputs, blocks=blocks):
                        output = self.root / f"{encoder}-{inputs}-{blocks}"
                        history, _ = self.train(encoder, inputs, blocks, output)
                        checkpoint = torch.load(output / "best.pt", weights_only=True)
                        self.assertEqual((checkpoint["config"]["encoder"], checkpoint["config"]["inputs"],
                                          checkpoint["config"]["trainable_blocks"]), (encoder, inputs, blocks))
                        metrics = export_live(output, self.root, self.root / "validation.csv")
                        self.assertEqual(metrics["confusion_matrix"], history[0]["validation"]["confusion_matrix"])
                        self.assertAlmostEqual(metrics["loss"], history[0]["validation"]["loss"], places=5)
                        with (output / "validation_predictions/predictions.csv").open(newline="") as stream:
                            self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_paired_vit_fine_tuning_from_a_cached_head_is_self_contained(self):
        data = write_split_files(self.root)
        backbone = tiny_vit()
        head_path = cached_head_checkpoint(self.root, data, backbone)
        before = backbone.encoder.patch_embed.proj.weight.detach().clone()
        output = self.root / "run"
        history, _ = self.train("dinov3-vitl", "paired", 1, output, backbone=backbone, init_head=head_path, epochs=2,
                                batch_size=2)
        self.assertEqual(history[0]["train"]["optimizer_steps"], 1)
        torch.testing.assert_close(backbone.encoder.patch_embed.proj.weight, before, rtol=0, atol=0)   # frozen stem
        checkpoint = torch.load(output / "last.pt", weights_only=True)
        self.assertEqual(checkpoint["config"]["initialization"]["init_head_epoch"], 5)
        self.assertIn("encoder.rope.periods", checkpoint["backbone_buffers"])
        head_path.unlink()
        with patch("disaster_lens.damage.dinov3.DINOv3Backbone.from_local_checkpoint",
                   side_effect=AssertionError("export must not read pretrained weights")):
            metrics = export_live(output, self.root, self.root / "validation.csv", checkpoint_name="last")
        self.assertEqual(metrics["confusion_matrix"], history[-1]["validation"]["confusion_matrix"])
        with self.assertRaises(FileExistsError):
            export_live(output, self.root, self.root / "validation.csv", checkpoint_name="last")
        with self.assertRaises(FileExistsError):
            self.train("dinov3-vitl", "paired", 1, output)

    def test_cached_head_must_match_the_live_feature_recipe(self):
        data = write_split_files(self.root)
        backbone = tiny_vit()
        head_path = cached_head_checkpoint(self.root, data, backbone)
        load_cached_head(backbone, head_path, data, 3)
        with torch.no_grad():
            backbone.encoder.patch_embed.proj.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "feature recipe differs"):
            load_cached_head(backbone, head_path, data, 3)
        checkpoint = torch.load(head_path, weights_only=True)
        checkpoint["config"]["objective"] = "rps"
        torch.save(checkpoint, head_path)
        with self.assertRaisesRegex(ValueError, "cross-entropy"):
            load_cached_head(backbone, head_path, data, 3)
        with self.assertRaisesRegex(ValueError, "paired"):
            self.train("dinov3-vitl", "post", 0, self.root / "x", init_head=head_path)

    def test_bf16_runs_record_precision_and_keep_float32_state(self):
        write_split_files(self.root)
        output = self.root / "bf16"
        history, _ = self.train("dinov3-vitl", "paired", 1, output, precision="bf16")
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        self.assertEqual(checkpoint["config"]["precision"], "bf16")
        self.assertTrue(all(v.dtype == torch.float32 for v in checkpoint["model_state_dict"].values() if v.is_floating_point()))
        metrics = export_live(output, self.root, self.root / "validation.csv")
        self.assertEqual(metrics["precision"], "bf16")
        self.assertEqual(metrics["confusion_matrix"], history[0]["validation"]["confusion_matrix"])

    def test_smoke_run_records_invariants(self):
        write_split_files(self.root)
        output = self.root / "smoke"
        history, _ = self.train("dinov3-vitl", "paired", 1, output, smoke_test=True)
        self.assertTrue(all(json.loads((output / "smoke_checks.json").read_text()).values()))
        self.assertLessEqual(history[0]["train"]["optimizer_steps"], 4)

    def test_validation_pass_is_read_only(self):
        data = write_split_files(self.root)
        model = DamageClassifier(tiny_vit(), build_head("post", 288, 32), inputs="post", trainable_blocks=1)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        metrics, rows = run_live_epoch(model, DataLoader(data["validation"], batch_size=2, collate_fn=collate_pairs),
                                       torch.device("cpu"), export_rows=True)
        self.assertEqual((metrics["optimizer_steps"], len(rows)), (0, 4))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@unittest.skipUnless(HAS_TIMM, "Install the vit dependency group.")
class LegacyCheckpointTests(unittest.TestCase):
    """The reported fine-tuned ViT and post-only baseline were saved in earlier formats; both must still load."""

    def test_legacy_post_only_baseline_maps_onto_the_post_only_classifier(self):
        reference = DamageClassifier(ResNet18Backbone(weights=None), build_head("post", 1152, 512), inputs="post")
        legacy = {("encoder." + k.split("extractor.backbone.encoder.", 1)[1] if k.startswith("extractor.") else
                   k.replace("head.linear.", "classifier.")): v for k, v in reference.state_dict().items()}
        checkpoint = {"epoch": 3, "model_state_dict": legacy, "config": {
            "architecture": LEGACY_POST_ONLY, "encoder_frozen": True, "class_to_index": CLASS_TO_INDEX,
            "image_size": 64, "roi_size": 3}}
        model, recipe = load_live_model(checkpoint, torch.device("cpu"))
        self.assertEqual((recipe["encoder"], model.inputs, model.trainable_blocks), ("resnet18", "post", 0))
        images, boxes = torch.randn(1, 3, 64, 64), [torch.tensor([[8., 8., 40., 40.]])]
        with torch.inference_mode():
            torch.testing.assert_close(model(images, images, boxes), reference.eval()(images, images, boxes))

    def test_legacy_fine_tuned_vit_loads_with_its_recorded_constructor(self):
        backbone = tiny_vit()
        head = TemporalFusionHead(building_dim=288, context_dim=32, projected_dim=8, hidden_dim=8, dropout=0)
        reference = DamageClassifier(backbone, head, inputs="paired", trainable_blocks=1)
        config = {"architecture": LEGACY_FINETUNED, "class_to_index": CLASS_TO_INDEX, "timm_version": timm.__version__,
                  "backbone_name": TIMM_MODEL_NAME, "precision": "float32", "roi_size": 3, "trainable_blocks": 1,
                  "backbone_constructor": {"img_size": 64, "embed_dim": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 2.,
                                           "num_classes": 0, "dynamic_img_size": True},
                  "model": {"building_dim": 288, "context_dim": 32, "projected_dim": 8, "hidden_dim": 8, "num_classes": 4,
                            "dropout": 0},
                  "data_specs": {"validation": {"preprocessing": {"image_size": 64}}}}
        checkpoint = {"epoch": 3, "config": config, "model_state_dict": reference.state_dict(),
                      "backbone_buffers": dict(backbone.named_buffers())}
        model, recipe = load_live_model(checkpoint, torch.device("cpu"))
        self.assertEqual((recipe["encoder"], recipe["image_size"], model.trainable_blocks), ("dinov3-vitl", 64, 1))
        pre, post, boxes = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64), [torch.tensor([[8., 8., 48., 48.]])]
        with torch.inference_mode():
            torch.testing.assert_close(model(pre, post, boxes), reference.eval()(pre, post, boxes))
        checkpoint["config"]["architecture"] = "something-else"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_live_model(checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
