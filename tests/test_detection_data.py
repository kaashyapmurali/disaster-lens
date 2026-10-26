"""Synthetic PRE-only localization fixtures; no real data or downloads."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from disaster_lens.data.building_index import (
    build_localization_index, building_bounds, file_sha256,
)
from disaster_lens.detection.data import LocalizationDataset, collate_localization


class LocalizationDataTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "index"
        self.assignments = self.root / "assignments.csv"
        self.assignments.write_text("scene_id,split\nscene-a,train\nscene-b,validation\n")
        # No post annotation/image is created. Unknown/missing pre subtype has
        # no effect; no post UID match or damage label is required.
        self.write_scene("scene-a", [self.feature("unknown", subtype="un-classified"),
                                     self.feature("pre-only")])
        self.write_scene("scene-b", [])
        self.write_scene("scene-c", [self.feature("unassigned")])
        self.write_scene("scene-d", [])

    @staticmethod
    def feature(uid: str, geometry: str | None = None, **properties: str) -> dict:
        return {
            "properties": {"feature_type": "building", "uid": uid, **properties},
            "wkt": geometry or "POLYGON ((10 5, 30 5, 30 15, 10 15, 10 5))",
        }

    def write_scene(self, scene: str, features: list[dict], size: tuple[int, int] = (80, 40)) -> None:
        root = self.root / "raw/train"
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels" / f"{scene}_pre_disaster.json").write_text(json.dumps({
            "features": {"xy": features}, "metadata": {"disaster": "example"},
        }))
        Image.new("RGB", size, color=(51, 102, 153)).save(
            root / "images" / f"{scene}_pre_disaster.png")

    def build(self) -> dict:
        return build_localization_index(self.root, self.assignments, self.output)

    def change_index(self, mutate) -> None:
        path = self.output / "scenes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report_path = self.output / "report.json"
        report = json.loads(report_path.read_text())
        report["scenes_sha256"] = file_sha256(path)
        report_path.write_text(json.dumps(report))

    def test_index_preserves_assignments_unknown_targets_empty_and_unassigned(self) -> None:
        source = self.root / "raw/train/labels/scene-a_pre_disaster.json"
        before = source.read_bytes()
        report = self.build()
        self.assertEqual(report["counts"]["train"]["buildings"], 2)
        self.assertEqual(report["counts"]["validation"]["empty_scenes"], 1)
        self.assertEqual(report["counts"]["unassigned"]["scenes"], 2)
        self.assertEqual(report["counts"]["unassigned"]["empty_scenes"], 1)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(file_sha256(self.output / "scenes.jsonl"), report["scenes_sha256"])
        with self.assertRaises(FileExistsError):
            self.build()
        with self.assertRaisesRegex(ValueError, "unassigned"):
            LocalizationDataset(self.output, self.root, "unassigned")

    def test_loader_native_pixels_boxes_class_ids_and_collation(self) -> None:
        self.build()
        train = LocalizationDataset(self.output, self.root, "train")
        validation = LocalizationDataset(self.output, self.root, "validation")
        self.assertEqual(len(train), 1)
        self.assertEqual({row["scene_id"] for row in train.scenes} &
                         {row["scene_id"] for row in validation.scenes}, set())
        sample = train[0]
        self.assertEqual(sample.reference_uids, ["pre-only", "unknown"])
        self.assertEqual(sample.image.shape, (3, 40, 80))
        torch.testing.assert_close(sample.image[:, 0, 0], torch.tensor([.2, .4, .6]))
        torch.testing.assert_close(sample.target["boxes"], torch.tensor([[10., 5., 30., 15.]] * 2))
        self.assertEqual(sample.target["labels"].tolist(), [1, 1])
        self.assertEqual(sample.target["labels"].dtype, torch.int64)
        self.assertEqual(sample.target["area"].tolist(), [200., 200.])
        empty = validation[0]
        self.assertEqual(empty.target["boxes"].shape, (0, 4))
        self.assertEqual(empty.target["labels"].shape, (0,))
        images, targets = collate_localization([sample, empty])
        self.assertEqual(len(images), 2)
        self.assertNotIn("reference_uids", targets[0])

    def test_topologically_invalid_polygon_retains_flagged_box_without_repair(self) -> None:
        bowtie = "POLYGON ((0 0, 20 20, 0 20, 20 0, 0 0))"
        self.write_scene("scene-a", [self.feature("bowtie", bowtie)])
        report = self.build()
        self.assertEqual(report["counts"]["train"]["invalid_geometry"], 1)
        self.assertEqual(report["counts"]["train"]["usable_boxes"], 1)
        dataset = LocalizationDataset(self.output, self.root, "train")
        self.assertFalse(dataset.scenes[0]["buildings"][0]["geometry_valid"])
        self.assertIsNotNone(dataset.scenes[0]["buildings"][0]["geometry_issue"])
        torch.testing.assert_close(dataset[0].target["boxes"], torch.tensor([[0., 0., 20., 20.]]))

    def test_clips_partial_box_but_refuses_wholly_outside_target_scene(self) -> None:
        partial = "POLYGON ((-5 -5, 10 -5, 10 10, -5 10, -5 -5))"
        bounds = building_bounds(partial, 80, 40)
        self.assertEqual(bounds["box"], [0., 0., 10., 10.])
        self.assertTrue(bounds["clipped"])
        outside = "POLYGON ((90 0, 100 0, 100 10, 90 10, 90 0))"
        self.write_scene("scene-a", [self.feature("bad", outside)])
        report = self.build()
        self.assertEqual(report["counts"]["train"]["box_issue:outside_image"], 1)
        with self.assertRaisesRegex(ValueError, "Unusable target"):
            LocalizationDataset(self.output, self.root, "train")

    def test_bad_geometry_is_audited_not_dropped(self) -> None:
        geometries = ["bad wkt", "POLYGON EMPTY", "POINT (1 2)",
                      "POLYGON ((0 0, 0 1, 0 2, 0 0))"]
        for geometry in geometries:
            with self.subTest(geometry=geometry):
                self.assertIsNone(building_bounds(geometry, 80, 40)["box"])

    def test_duplicate_uid_fails_without_publishing_partial_output(self) -> None:
        self.write_scene("scene-a", [self.feature("same"), self.feature("same")])
        with self.assertRaisesRegex(ValueError, "duplicate building UID"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".localization-*")), [])

    def test_cross_scene_uid_rejected(self) -> None:
        self.write_scene("scene-b", [self.feature("pre-only")])
        with self.assertRaisesRegex(ValueError, "spans scenes"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_bad_or_missing_assignments_fail_atomically(self) -> None:
        for text in ("scene_id,split\nscene-a,test\n", "scene_id,split\nscene-a,train\nscene-a,train\n",
                     "scene_id,split\nmissing-scene,train\n"):
            with self.subTest(assignments=text):
                self.assignments.write_text(text)
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_missing_pre_image_fails_without_output(self) -> None:
        (self.root / "raw/train/images/scene-a_pre_disaster.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_index_and_annotation_changes_rejected(self) -> None:
        self.build()
        index_path = self.output / "scenes.jsonl"
        original = index_path.read_bytes()
        index_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "changed localization index"):
            LocalizationDataset(self.output, self.root, "train")
        index_path.write_bytes(original)
        annotation = self.root / "raw/train/labels/scene-a_pre_disaster.json"
        annotation.write_text(annotation.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "annotation changed"):
            LocalizationDataset(self.output, self.root, "train")

    def test_source_split_and_path_tampering_rejected(self) -> None:
        self.build()
        self.change_index(lambda rows: rows[0].update(pre_image="../outside.png"))
        with self.assertRaisesRegex(ValueError, "pre-event xBD train-source"):
            LocalizationDataset(self.output, self.root, "train")
        self.change_index(lambda rows: rows[0].update(source_split="test"))
        with self.assertRaisesRegex(ValueError, "source/split mismatch"):
            LocalizationDataset(self.output, self.root, "train")

    def test_dimension_changes_rejected_at_load(self) -> None:
        self.build()
        dataset = LocalizationDataset(self.output, self.root, "train")
        self.write_scene("scene-a", [], size=(40, 80))
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
