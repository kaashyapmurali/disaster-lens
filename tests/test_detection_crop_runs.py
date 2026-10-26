"""Crop training, tiled validation, and protocol-change continuation through train()/export()."""

import json
import unittest
from pathlib import Path

import test_detection_instances as fixture
from disaster_lens.detection.train import _validation_subset, export, make_parser, train


class CropTrainingRunTests(unittest.TestCase):
    setUpClass = fixture.InstanceTrainingTests.setUpClass
    tearDownClass = fixture.InstanceTrainingTests.tearDownClass
    setUp = fixture.InstanceTrainingTests.setUp

    def common(self, run: Path) -> list[str]:
        return ["--data-dir", str(self.root), "--index-dir", str(self.index), "--output-dir", str(run), "--device", "cpu",
                "--image-size", "64", "--max-detections", "12", "--post-nms-proposals", "16", "--pre-nms-proposals", "32",
                "--geometry-policy", "make-valid-polygonal"]

    def parent(self, name: str = "parent") -> Path:
        run = self.root / name
        train(make_parser().parse_args(["train", "--fresh-training", "--coco-weights", "synthetic.pt", "--weights-sha256", "a" * 64,
                                        "--epochs", "1", *self.common(run)]))
        return run

    def crop_flags(self) -> list[str]:
        return ["--crop-size", "32", "--crops-per-scene", "2", "--validation-tile-size", "32", "--validation-tile-min-overlap", "8", "--validation-subset", "1"]

    def resume(self, parent: Path, name: str, extra: list[str]):
        return train(make_parser().parse_args(["train", "--resume-from", str(parent), "--additional-epochs", "1",
                                               *self.common(self.root / name), *extra]))

    def test_protocol_change_needs_flag_is_recorded_and_runs_crops_with_tiled_subset_validation(self):
        parent = self.parent()
        with self.assertRaisesRegex(ValueError, "allow-protocol-change"):
            self.resume(parent, "refused", self.crop_flags())
        self.assertFalse((self.root / "refused").exists())
        result = self.resume(parent, "child", [*self.crop_flags(), "--allow-protocol-change"])
        config = result["config"]
        self.assertEqual(config["training_protocol"]["mode"], "random_crops")
        self.assertEqual(config["training_protocol"]["upsampling"], 2.0)
        self.assertEqual(config["validation_protocol"]["mode"], "tiled")
        self.assertEqual(len(config["validation_protocol"]["subset_scene_ids"]), 1)
        self.assertEqual(set(config["recipe_changes"]), {"training_protocol", "validation_protocol"})
        self.assertEqual(config["recipe_changes"]["training_protocol"]["parent"], {"mode": "whole_scene"})
        parent_config = json.loads((parent / "config.json").read_text())
        self.assertEqual(config["data"], parent_config["data"])      # full-population identities unchanged
        self.assertEqual(result["history"][0]["train"]["scenes"], 2 * parent_config["data"]["train"]["scene_ids"].__len__())
        self.assertEqual(result["history"][0]["validation"]["pooled"]["scenes"], 1)
        # A grandchild with the same protocol needs no flag.
        grandchild = self.resume(self.root / "child", "grandchild", self.crop_flags())
        self.assertEqual(grandchild["config"]["recipe_changes"], {})
        self.assertEqual([item["epoch"] for item in grandchild["config"]["lineage"]], [1, 2])

    def test_export_of_a_crop_validated_checkpoint_must_be_tiled_and_records_the_protocol(self):
        parent = self.parent()
        child = self.root / "child"
        self.resume(parent, "child", [*self.crop_flags(), "--allow-protocol-change"])
        args = make_parser().parse_args(["export", "--data-dir", str(self.root), "--index-dir", str(self.index),
                                         "--run-dir", str(child), "--device", "cpu"])
        with self.assertRaisesRegex(ValueError, "tiled/subset protocol"):
            export(args)
        args.tile_size, args.tile_min_overlap, args.tile_batch_size = 32, 8, 2
        metrics = export(args)
        self.assertEqual(metrics["checkpoint_recorded_metrics"]["protocol"]["mode"], "tiled")
        self.assertEqual(metrics["pooled"]["scenes"], 2)                  # the export covers the full validation split

    def test_fresh_crop_training_and_argument_validation(self):
        result = train(make_parser().parse_args(["train", "--fresh-training", "--coco-weights", "synthetic.pt", "--weights-sha256", "a" * 64,
                                                 "--epochs", "1", *self.common(self.root / "fresh"), "--crop-size", "32"]))
        self.assertEqual(result["config"]["training_protocol"]["crops_per_scene"], 1)
        self.assertEqual(result["config"]["validation_protocol"]["mode"], "whole_scene")
        for extra, message in ((["--allow-protocol-change"], "only to --resume-from"),
                               (["--smoke-test", "--random-init", "--validation-subset", "1"], "smoke-test")):
            args = make_parser().parse_args(["train", "--fresh-training", "--coco-weights", "synthetic.pt", "--weights-sha256", "a" * 64,
                                             *self.common(self.root / "bad"), *extra])
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                train(args)

    def test_validation_subset_is_event_balanced_deterministic_and_seeded(self):
        rows = [{"scene_id": f"{event}-{i}", "event": event, "buildings": [1]} for event, n in (("a", 10), ("b", 3), ("c", 6)) for i in range(n)]
        rows.append({"scene_id": "empty", "event": "d", "buildings": []})          # scenes without buildings are never chosen
        chosen = _validation_subset(rows, 9, seed=1)
        self.assertEqual(len(chosen), 9)
        events = [row["event"] for row in chosen]
        self.assertEqual({e: events.count(e) for e in "abc"}, {"a": 3, "b": 3, "c": 3})
        self.assertEqual(chosen, _validation_subset(rows, 9, seed=1))
        self.assertNotEqual([r["scene_id"] for r in chosen], [r["scene_id"] for r in _validation_subset(rows, 9, seed=2)])
        self.assertEqual(len(_validation_subset(rows, 100, seed=1)), 19)         # capped at the eligible population
        self.assertNotIn("empty", [row["scene_id"] for row in _validation_subset(rows, 100, seed=1)])
        self.assertEqual([r["scene_id"] for r in chosen], sorted(r["scene_id"] for r in chosen))
        with self.assertRaises(ValueError):
            _validation_subset(rows, 0, seed=1)


if __name__ == "__main__":
    unittest.main()
