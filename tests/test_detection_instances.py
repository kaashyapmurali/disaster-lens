"""Small end-to-end mask workflow tests; no real weights or remote compute."""

import json
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import test_detection_training as fixtures
from disaster_lens.detection.mask_metrics import decode_window
from disaster_lens.detection.model import DetectorConfig, build_detector
from disaster_lens.data.building_index import build_localization_index
from disaster_lens.detection.evaluate import evaluate_instances
from disaster_lens.detection.instance_data import InstanceSegmentationDataset
from disaster_lens.detection.data import collate_localization
from disaster_lens.detection.train import ARCHITECTURE, export, train


TinyMaskDetector = fixtures.TinyDetector


class InstanceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        fixtures.LocalizationTrainingTests.setUp(self)
        self.enterContext(patch("disaster_lens.detection.train.build_detector",
                                side_effect=lambda *args, **kwargs: TinyMaskDetector()))

    def train_args(self):
        return fixtures.LocalizationTrainingTests.train_args(self)

    def export_args(self):
        return fixtures.LocalizationTrainingTests.export_args(self)

    def test_mask_training_selection_checkpoint_and_compact_export(self):
        result = train(self.train_args())
        self.assertEqual(result["config"]["architecture"], ARCHITECTURE)
        self.assertIn("mask AP", result["config"]["selection"])
        self.assertEqual(result["history"][0]["validation"]["definition"]["metric_kind"], "mask")
        self.assertEqual(result["history"][0]["validation"]["box_metrics"]["definition"]["metric_kind"], "box")
        checks = json.loads((self.run / "smoke_checks.json").read_text())
        self.assertTrue(checks["mask_head_updated"])
        self.assertTrue(checks["mask_predictor_updated"])
        self.assertFalse((self.run / "validation_predictions").exists())
        metrics = export(self.export_args())
        self.assertEqual(metrics["pooled"], result["history"][0]["validation"]["pooled"])
        rows = [json.loads(line) for line in (self.run / "validation_predictions/predictions.jsonl").read_text().splitlines()]
        self.assertEqual([row["scene_id"] for row in rows], ["val-a", "val-b"])
        reference = rows[0]["reference_buildings"][0]
        self.assertEqual(reference["reference_uid"], "val-a-uid")
        restored = decode_window(reference["mask_window_rle"], (32, 32))
        self.assertEqual(restored.area, 49)
        self.assertIn("mask_window_rle", rows[0]["detections"][0])
        self.assertNotIn("mask_roi_probs", rows[0]["detections"][0])  # No huge float JSON payload.
        self.assertEqual(metrics["provenance"]["mask_targets"]["geometry_policy"], "make-valid-polygonal")
        with self.assertRaises(FileExistsError):
            export(self.export_args())

    def test_tiled_export_is_separate_marked_not_comparable_and_single_tile_equals_untiled(self):
        result = train(self.train_args())
        untiled = export(self.export_args())
        args = self.export_args()
        args.tile_size, args.tile_min_overlap, args.tile_batch_size = 32, 8, 2     # scenes are 32 px: one tile
        tiled = export(args)
        self.assertTrue((self.run / "validation_predictions_tile32/predictions.jsonl").is_file())
        self.assertTrue((self.run / "validation_predictions/predictions.jsonl").is_file())   # untiled export untouched
        self.assertEqual(tiled["export_verification"], "tiled_inference_not_comparable_to_checkpoint_metrics")
        self.assertEqual(tiled["provenance"]["tiling"]["tile_size"], 32)
        self.assertEqual(tiled["pooled"], untiled["pooled"])
        self.assertEqual(tiled["checkpoint_recorded_metrics"]["pooled"], result["history"][0]["validation"]["pooled"])
        self.assertIsNone(untiled["provenance"]["tiling"])
        with self.assertRaises(FileExistsError):
            export(args)
        bad = self.export_args()
        bad.tile_size, bad.tile_min_overlap, bad.tile_batch_size = 16, 8, 2         # below the tile floor
        with self.assertRaises(ValueError):
            export(bad)

    def test_hole_is_penalized_by_mask_ap_even_when_box_is_correct(self):
        path = self.root / "raw/train/labels/val-a_pre_disaster.json"
        record = json.loads(path.read_text())
        record["features"]["xy"][0]["wkt"] = "POLYGON ((1 1,8 1,8 8,1 8,1 1),(2 2,2 7,7 7,7 2,2 2))"
        path.write_text(json.dumps(record))
        self.index = self.root / "hole-index"
        build_localization_index(self.root, self.root / "assignments.csv", self.index)
        result = train(self.train_args())
        validation = result["history"][0]["validation"]
        self.assertEqual(validation["box_metrics"]["pooled"]["ap50"], 1.)
        self.assertEqual(validation["pooled"]["ap50"], 0.)

    def test_instance_evaluation_checks_full_coverage_and_rejects_dense_outputs(self):
        data = InstanceSegmentationDataset(self.index, self.root, "validation")
        loader = DataLoader(data, batch_size=1, collate_fn=collate_localization)
        model = TinyMaskDetector()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        metrics = evaluate_instances(model, loader, torch.device("cpu"), config=DetectorConfig())
        self.assertEqual(metrics["pooled"]["scenes"], 2)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        duplicate = DataLoader(data, sampler=[0, 0], collate_fn=collate_localization)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            evaluate_instances(model, duplicate, torch.device("cpu"), config=DetectorConfig())
        incomplete = DataLoader(data, sampler=[0], collate_fn=collate_localization)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            evaluate_instances(model, incomplete, torch.device("cpu"), config=DetectorConfig())
        with self.assertRaisesRegex(ValueError, "compact outputs"):
            evaluate_instances(model, loader, torch.device("cpu"),
                               config=DetectorConfig(mask_representation="dense"))

    def test_real_compact_mask_model_passes_instance_evaluator(self):
        config = DetectorConfig(min_size=32, max_size=32, anchor_sizes=(8, 16, 32, 64, 128),
                                             rpn_pre_nms_top_n=8, rpn_post_nms_top_n=4,
                                             detections_per_image=4, rpn_batch_size=8, box_batch_size=8)
        model = build_detector(config)
        data = InstanceSegmentationDataset(self.index, self.root, "validation")
        metrics = evaluate_instances(model, DataLoader(data, batch_size=1, collate_fn=collate_localization),
                                     torch.device("cpu"), config=config)
        self.assertEqual(metrics["pooled"]["scenes"], 2)
        self.assertEqual(metrics["pooled"]["reference_buildings"], 1)
        self.assertEqual(metrics["definition"]["metric_kind"], "mask")

    def test_full_mask_epochs_keep_earliest_tied_mask_ap_checkpoint(self):
        args = self.train_args()
        args.random_init = False
        args.smoke_test = False
        args.coco_weights = self.root / "synthetic-mask-not-read.pt"
        args.weights_sha256 = "a" * 64
        args.epochs = 2
        result = train(args)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(torch.load(self.run / "best.pt", weights_only=True)["epoch"], 1)
        self.assertEqual(torch.load(self.run / "last.pt", weights_only=True)["epoch"], 2)
        self.assertFalse((self.run / "validation_predictions").exists())

    def test_geometry_version_mismatch_blocks_export(self):
        train(self.train_args())
        with patch("disaster_lens.detection.train.shapely.geos_version_string", "synthetic-mismatch"):
            with self.assertRaisesRegex(ValueError, "geometry implementation"):
                export(self.export_args())
        self.assertFalse((self.run / "validation_predictions").exists())


if __name__ == "__main__":
    unittest.main()
