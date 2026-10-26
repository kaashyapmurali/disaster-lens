"""Detector training/export workflow on synthetic scenes. A tiny stand-in model avoids large checkpoint IO;
the real Mask R-CNN forward/backward path is covered in test_detection_model.
No pretrained files, real imagery, remote APIs, or GPU execution are used here.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from disaster_lens.data.building_index import build_localization_index
from disaster_lens.detection.data import collate_localization
from disaster_lens.detection.evaluate import evaluate_instances
from disaster_lens.detection.instance_data import InstanceSegmentationDataset
from disaster_lens.detection.model import DetectorConfig
from disaster_lens.detection.train import batch_timing_summary, export, make_parser, train


class TinyDetector(nn.Module):
    """Stand-in with Mask R-CNN's module names and output contract (compact ROI masks)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.rpn = nn.Linear(1, 1)
        self.roi_heads = nn.Linear(1, 1)
        self.roi_heads.mask_head = nn.Linear(1, 1)
        self.roi_heads.mask_predictor = nn.Linear(1, 1)
        for parameter in self.parameters():
            nn.init.constant_(parameter, .1)

    def forward(self, images, targets=None):
        value = sum(parameter.square().mean() for parameter in self.parameters())
        if self.training:
            return {"loss_classifier": value, "loss_box_reg": value * .5, "loss_objectness": value * .2,
                    "loss_rpn_box_reg": value * .1, "loss_mask": value * .3}
        return [{"boxes": image.new_tensor([[1., 1., 8., 8.]]),
                 "scores": value.sigmoid().reshape(1),
                 "labels": torch.ones(1, dtype=torch.int64, device=image.device),
                 "mask_roi_probs": image.new_ones((1, 1, 28, 28)),
                 "mask_image_size": tuple(image.shape[-2:])} for image in images]


class LocalizationTrainingTests(unittest.TestCase):
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
        self.index = self.root / "index"
        self.run = self.root / "run"
        assignments = self.root / "assignments.csv"
        assignments.write_text("scene_id,split\ntrain-a,train\ntrain-b,train\nval-a,validation\nval-b,validation\n")
        for scene, empty in (("train-a", False), ("train-b", True), ("val-a", False), ("val-b", True)):
            label = self.root / "raw/train/labels" / f"{scene}_pre_disaster.json"
            image = self.root / "raw/train/images" / f"{scene}_pre_disaster.png"
            label.parent.mkdir(parents=True, exist_ok=True)
            image.parent.mkdir(parents=True, exist_ok=True)
            features = [] if empty else [{"properties": {"feature_type": "building", "uid": f"{scene}-uid"},
                                          "wkt": "POLYGON ((1 1, 8 1, 8 8, 1 8, 1 1))"}]
            label.write_text(json.dumps({"features": {"xy": features},
                                         "metadata": {"disaster": "negative" if empty else "positive"}}))
            Image.new("RGB", (32, 32), color=(30, 60, 90)).save(image)
        build_localization_index(self.root, assignments, self.index)
        self.factory = patch("disaster_lens.detection.train.build_detector",
                             side_effect=lambda *args, **kwargs: TinyDetector())
        self.factory.start()
        self.addCleanup(self.factory.stop)

    def train_args(self):
        return make_parser().parse_args([
            "train", "--fresh-training", "--data-dir", str(self.root), "--index-dir", str(self.index),
            "--output-dir", str(self.run), "--device", "cpu", "--random-init", "--smoke-test",
            "--image-size", "64", "--max-detections", "12", "--post-nms-proposals", "16",
            "--pre-nms-proposals", "32",
        ])

    def export_args(self, name="best"):
        return make_parser().parse_args([
            "export", "--data-dir", str(self.root), "--index-dir", str(self.index),
            "--run-dir", str(self.run), "--checkpoint-name", name, "--device", "cpu",
        ])

    def test_train_checkpoint_reload_and_separate_best_last_exports(self) -> None:
        result = train(self.train_args())
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["history"][0]["train"]["optimizer_steps"], 2)
        self.assertEqual(result["config"]["data"]["validation"]["scene_ids"], ["val-a", "val-b"])
        self.assertFalse((self.run / "validation_predictions").exists())
        saved = torch.load(self.run / "best.pt", map_location="cpu", weights_only=True)
        self.assertIn("rng_state", saved)
        self.assertEqual(saved["rng_state"]["torch_cpu"].dtype, torch.uint8)
        self.assertEqual(saved["rng_state"]["train_loader_generator"].dtype, torch.uint8)
        self.assertIsInstance(saved["rng_state"]["numpy"]["keys"], list)
        self.assertTrue(all(json.loads((self.run / "smoke_checks.json").read_text()).values()))
        self.assertEqual(json.loads((self.run / "status.json").read_text())["state"], "completed")
        timings = json.loads((self.run / "timings.json").read_text())["stages"]
        stages = {record["name"]: record for record in timings}
        self.assertTrue(all(record["state"] == "complete" for record in timings))
        for name in ("train_index_validation", "validation_index_validation",
                     "model_initialization_and_device_transfer", "epoch_1_train", "epoch_1_validation",
                     "epoch_1_last_checkpoint_write", "epoch_1_best_checkpoint_write"):
            self.assertIn(name, stages)
        self.assertEqual(result["history"][0]["train"]["batch_timing"]["after_first_batch_count"], 1)
        # Export must not require the original initialization file.
        best = export(self.export_args())
        last = export(self.export_args("last"))
        self.assertEqual(best["pooled"], result["history"][0]["validation"]["pooled"])
        self.assertEqual(last["pooled"], best["pooled"])
        self.assertEqual(best["pooled"]["reference_buildings"], 1)
        self.assertEqual(best["per_event"]["negative"]["at_score_threshold_iou50"]["false_positives"], 1)
        rows = [json.loads(line) for line in (self.run / "validation_predictions/predictions.jsonl").read_text().splitlines()]
        self.assertEqual([row["scene_id"] for row in rows], ["val-a", "val-b"])
        self.assertEqual(rows[0]["detections"][0]["matched_reference_uid_mask_iou50"], "val-a-uid")
        self.assertIsNone(rows[1]["detections"][0]["matched_reference_uid_mask_iou50"])
        self.assertNotEqual(rows[0]["detections"][0]["building_id"], "val-a-uid")
        self.assertEqual(rows[1]["reference_buildings"], [])
        self.assertTrue(best["provenance"]["smoke_test"])
        self.assertEqual(len(best["provenance"]["predictions_sha256"]), 64)
        with self.assertRaises(FileExistsError):
            export(self.export_args())
        with self.assertRaises(FileExistsError):
            train(self.train_args())

    def make_test_source(self, scenes: dict[str, bool]) -> Path:
        """An xBD test-source tree (raw/test) and its single-split index."""
        for scene, empty in scenes.items():
            label = self.root / "raw/test/labels" / f"{scene}_pre_disaster.json"
            image = self.root / "raw/test/images" / f"{scene}_pre_disaster.png"
            label.parent.mkdir(parents=True, exist_ok=True)
            image.parent.mkdir(parents=True, exist_ok=True)
            features = [] if empty else [{"properties": {"feature_type": "building", "uid": f"{scene}-uid"},
                                          "wkt": "POLYGON ((1 1, 8 1, 8 8, 1 8, 1 1))"}]
            label.write_text(json.dumps({"features": {"xy": features}, "metadata": {"disaster": "positive"}}))
            Image.new("RGB", (32, 32), color=(30, 60, 90)).save(image)
        index = self.root / "test-index"
        build_localization_index(self.root, None, index, source_split="test")
        return index

    def test_test_source_index_loads_as_its_own_split_and_exports_without_recorded_metrics(self) -> None:
        index = self.make_test_source({"test-a": False, "test-b": True})
        data = InstanceSegmentationDataset(index, self.root, "test")
        self.assertEqual([row["scene_id"] for row in data.scenes], ["test-a", "test-b"])
        with self.assertRaises(ValueError):
            build_localization_index(self.root, self.root / "assignments.csv", self.root / "x", source_split="test")
        args = self.train_args()
        args.random_init, args.smoke_test = False, False
        args.coco_weights, args.weights_sha256, args.epochs = Path("synthetic-not-read.pt"), "a" * 64, 1
        train(args)
        exporting = self.export_args()
        exporting.split, exporting.index_dir = "test", index
        metrics = export(exporting)
        self.assertEqual(metrics["export_verification"], "test_split_no_recorded_metrics")
        self.assertEqual((metrics["provenance"]["split"], metrics["pooled"]["scenes"]), ("test", 2))
        self.assertTrue((self.run / "test_predictions/predictions.jsonl").is_file())
        self.assertFalse((self.run / "validation_predictions").exists())
        with self.assertRaises(FileExistsError):
            export(exporting)

    def test_test_export_refuses_scenes_seen_in_training(self) -> None:
        index = self.make_test_source({"val-a": False})        # same scene ID as a validation scene
        args = self.train_args()
        args.random_init, args.smoke_test = False, False
        args.coco_weights, args.weights_sha256, args.epochs = Path("synthetic-not-read.pt"), "a" * 64, 1
        train(args)
        exporting = self.export_args()
        exporting.split, exporting.index_dir = "test", index
        with self.assertRaisesRegex(ValueError, "overlap"):
            export(exporting)

    def test_batch_timing_summary_keeps_startup_first_and_later_batches_separate(self) -> None:
        profile = {"iterator_startup_seconds": 1., "batches": [
            {"loader_wait_seconds": 2., "transfer_seconds": 3.},
            {"loader_wait_seconds": 4., "transfer_seconds": 5.},
        ]}
        summary = batch_timing_summary(profile)
        self.assertEqual(summary["iterator_startup_seconds"], 1.)
        self.assertEqual(summary["first_batch"]["loader_wait_seconds"], 2.)
        self.assertEqual(summary["phase_totals_seconds"]["loader_wait_seconds"], 6.)
        self.assertEqual(summary["after_first_batch_mean_seconds"]["transfer_seconds"], 5.)
        single = batch_timing_summary({"iterator_startup_seconds": 0., "batches": profile["batches"][:1]})
        self.assertIsNone(single["after_first_batch_mean_seconds"]["transfer_seconds"])

    def test_checkpoint_selection_keeps_first_epoch_on_tied_ap(self) -> None:
        args = self.train_args()
        args.random_init = False
        args.smoke_test = False
        args.coco_weights = Path("synthetic-not-read.pt")
        args.weights_sha256 = "a" * 64
        args.epochs = 2
        train(args)
        best = torch.load(self.run / "best.pt", weights_only=True)
        last = torch.load(self.run / "last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(last["epoch"], 2)
        self.assertFalse(torch.equal(best["model_state_dict"]["backbone.weight"],
                                     last["model_state_dict"]["backbone.weight"]))

    def test_smoke_detects_mutation_of_frozen_state(self) -> None:
        original = TinyDetector.forward

        def changed_buffer(model, images, targets=None):
            if not hasattr(model, "counter"):
                model.register_buffer("counter", torch.zeros(1))
            model.counter.add_(1)
            return original(model, images, targets)

        with patch.object(TinyDetector, "forward", changed_buffer):
            with self.assertRaisesRegex(RuntimeError, "smoke invariants failed"):
                train(self.train_args())
        self.assertEqual(json.loads((self.run / "status.json").read_text())["state"], "failed")
        checks = json.loads((self.run / "smoke_checks.json").read_text())
        self.assertFalse(checks["normalization_buffers_unchanged"])

    def test_fresh_training_must_be_explicit_and_does_not_claim_resume(self) -> None:
        args = self.train_args()
        args.fresh_training = False
        with self.assertRaisesRegex(ValueError, "does not continue"):
            train(args)
        self.assertFalse(self.run.exists())

    def test_requires_pretrained_init_for_non_smoke_and_disallows_random_cuda(self) -> None:
        args = self.train_args()
        args.smoke_test = False
        with self.assertRaisesRegex(ValueError, "restricted"):
            train(args)
        args.random_init = False
        with self.assertRaisesRegex(ValueError, "local COCO_V1"):
            train(args)
        args = self.train_args()
        args.device = "cuda"
        with self.assertRaisesRegex(ValueError, "restricted"):
            train(args)
        self.assertFalse(self.run.exists())

    def test_nonfinite_loss_stops_without_retry_export_or_checkpoint(self) -> None:
        args = self.train_args()
        with patch.object(TinyDetector, "forward", return_value={"loss": torch.tensor(float("nan"), requires_grad=True)}):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                train(args)
        self.assertEqual(json.loads((self.run / "status.json").read_text())["state"], "failed")
        self.assertFalse((self.run / "best.pt").exists())
        self.assertFalse((self.run / "validation_predictions").exists())
        stages = {record["name"]: record for record in json.loads((self.run / "timings.json").read_text())["stages"]}
        self.assertEqual(stages["epoch_1_train"]["state"], "failed")

    def test_export_changed_index_and_changed_config_fail_without_partial_output(self) -> None:
        train(self.train_args())
        config_path = self.run / "config.json"
        original = config_path.read_text()
        config = json.loads(original)
        config["seed"] += 1
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            export(self.export_args())
        config_path.write_text(original)
        path = self.index / "scenes.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed localization index"):
            export(self.export_args())
        self.assertFalse((self.run / "validation_predictions").exists())

    def test_export_reloaded_metric_disagreement_does_not_publish(self) -> None:
        train(self.train_args())
        original = TinyDetector.forward

        def wrong_boxes(model, images, targets=None):
            outputs = original(model, images, targets)
            for output in outputs:
                output["boxes"] += 10.
            return outputs

        with patch.object(TinyDetector, "forward", wrong_boxes):
            with self.assertRaisesRegex(ValueError, "metrics differ"):
                export(self.export_args())
        self.assertFalse((self.run / "validation_predictions").exists())
        self.assertEqual(list(self.run.glob(".localization-export-*")), [])
        failed = self.run / "failed_best_validation_predictions"
        self.assertTrue((failed / "predictions.jsonl").exists())
        self.assertEqual(json.loads((failed / "metrics.json").read_text())["export_verification"], "failed_checkpoint_metrics")
        self.assertTrue((failed / "comparison_failure.json").exists())

    def test_evaluation_is_read_only_and_checks_scene_coverage_and_labels(self) -> None:
        dataset = InstanceSegmentationDataset(self.index, self.root, "validation")
        model = TinyDetector()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_localization)
        metrics = evaluate_instances(model, loader, torch.device("cpu"), config=DetectorConfig())
        self.assertEqual(metrics["pooled"]["scenes"], 2)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        duplicate_loader = DataLoader(dataset, sampler=[0, 0], collate_fn=collate_localization)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            evaluate_instances(model, duplicate_loader, torch.device("cpu"), config=DetectorConfig())
        incomplete = DataLoader(dataset, sampler=[0], collate_fn=collate_localization)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            evaluate_instances(model, incomplete, torch.device("cpu"), config=DetectorConfig())
        original = TinyDetector.forward

        def wrong_labels(model, images, targets=None):
            outputs = original(model, images, targets)
            outputs[0]["labels"].zero_()
            return outputs

        with patch.object(TinyDetector, "forward", wrong_labels):
            with self.assertRaisesRegex(ValueError, "Invalid building detections"):
                evaluate_instances(model, loader, torch.device("cpu"), config=DetectorConfig())


if __name__ == "__main__":
    unittest.main()
