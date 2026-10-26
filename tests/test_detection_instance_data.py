"""Synthetic PRE polygons/masks only. Never read real imagery or repair labels."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from disaster_lens.data.footprints import (
    MaskGeometryError, audit_instance_targets, rasterize_footprint_window,
)
from disaster_lens.data.building_index import build_localization_index
from disaster_lens.detection.instance_data import InstanceSegmentationDataset
from disaster_lens.detection.data import collate_localization


RECTANGLE = "POLYGON ((1 1, 8 1, 8 8, 1 8, 1 1))"


class InstanceMaskDataTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = self.root / "index"
        assignments = self.root / "assignments.csv"
        assignments.write_text("scene_id,split\ntrain,train\nvalidation,validation\n")
        self.assignments = assignments
        self.write_scene("train", [("unknown", RECTANGLE), ("pre-only", RECTANGLE)])
        self.write_scene("validation", [])

    def write_scene(self, scene: str, buildings: list[tuple[str, str]]) -> None:
        label = self.root / "raw/train/labels" / f"{scene}_pre_disaster.json"
        image = self.root / "raw/train/images" / f"{scene}_pre_disaster.png"
        label.parent.mkdir(parents=True, exist_ok=True)
        image.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(json.dumps({"features": {"xy": [
            {"properties": {"uid": uid, "feature_type": "building", "subtype": "un-classified"}, "wkt": geometry}
            for uid, geometry in buildings]}}))
        Image.new("RGB", (16, 12), (50, 100, 150)).save(image)

    def build(self) -> None:
        build_localization_index(self.root, self.assignments, self.index)

    def test_loader_preserves_instance_order_overlaps_and_unknown_severity(self) -> None:
        self.build()
        dataset = InstanceSegmentationDataset(self.index, self.root, "train", mask_representation="dense")
        sample = dataset[0]
        self.assertEqual(sample.reference_uids, ["pre-only", "unknown"])
        self.assertEqual(sample.image.shape, (3, 12, 16))
        self.assertEqual(sample.target["masks"].shape, (2, 12, 16))
        self.assertEqual(sample.target["masks"].dtype, torch.uint8)
        self.assertEqual(sample.target["area"].tolist(), [49., 49.])
        self.assertEqual(sample.target["labels"].tolist(), [1, 1])
        self.assertTrue(torch.equal(sample.target["masks"][0], sample.target["masks"][1]))
        self.assertEqual(int(sample.target["masks"][0, 1, 1]), 1)
        self.assertEqual(int(sample.target["masks"][0, 8, 8]), 0)
        images, targets = collate_localization([sample])
        self.assertEqual(len(images), 1)
        self.assertIn("masks", targets[0])
        self.assertNotIn("reference_uids", targets[0])

    def test_empty_scene_masks_are_native_shaped(self) -> None:
        self.build()
        sample = InstanceSegmentationDataset(self.index, self.root, "validation", mask_representation="dense")[0]
        self.assertEqual(sample.target["masks"].shape, (0, 12, 16))
        self.assertEqual(sample.target["boxes"].shape, (0, 4))
        self.assertEqual(sample.target["area"].shape, (0,))

    def test_pixel_centers_holes_multipolygons_and_clipping(self) -> None:
        hole = "POLYGON ((0 0,10 0,10 10,0 10,0 0),(2 2,2 8,8 8,8 2,2 2))"
        mask = rasterize_footprint_window(hole, 16, 12)
        self.assertEqual(mask.area, 64)
        self.assertEqual(int(mask.pixels[4, 4]), 0)
        self.assertEqual(int(mask.pixels[0, 0]), 1)
        multi = "MULTIPOLYGON (((0 0,2 0,2 2,0 2,0 0)),((4 0,6 0,6 2,4 2,4 0)))"
        self.assertEqual(rasterize_footprint_window(multi, 16, 12).area, 8)
        clipped = rasterize_footprint_window("POLYGON ((-2 -2,3 -2,3 3,-2 3,-2 -2))", 16, 12)
        self.assertEqual(clipped.bounds, (0, 0, 3, 3))
        self.assertEqual(clipped.area, 9)
        boundary = rasterize_footprint_window("POLYGON ((.5 .5,2.5 .5,2.5 2.5,.5 2.5,.5 .5))", 16, 12)
        self.assertEqual(boundary.area, 9)  # Boundary pixel centers included explicitly.

    def test_invalid_and_subpixel_geometry_blocked_without_repair(self) -> None:
        for geometry, reason in (
            ("bad wkt", "unparseable_geometry"), ("POLYGON EMPTY", "empty_or_nonfinite_geometry"),
            ("POINT (1 2)", "non_polygon"),
            ("POLYGON ((0 0,10 10,0 10,10 0,0 0))", "invalid_geometry"),
            ("POLYGON ((.1 .1,.4 .1,.4 .4,.1 .4,.1 .1))", "zero_rasterized_pixels"),
            ("POLYGON ((20 20,22 20,22 22,20 22,20 20))", "outside_or_zero_area_after_clipping"),
        ):
            with self.subTest(reason=reason), self.assertRaises(MaskGeometryError) as raised:
                rasterize_footprint_window(geometry, 16, 12)
            self.assertEqual(raised.exception.reason, reason)

    def test_invalid_topology_refuses_dataset_before_training(self) -> None:
        self.write_scene("train", [("invalid", "POLYGON ((0 0,10 10,0 10,10 0,0 0))")])
        self.build()
        with self.assertRaisesRegex(ValueError, "invalid geometries"):
            InstanceSegmentationDataset(self.index, self.root, "train")

    def test_dense_mask_guard_raises_instead_of_dropping_instances(self) -> None:
        self.build()
        dataset = InstanceSegmentationDataset(self.index, self.root, "train", max_mask_bytes=16 * 12,
                                              mask_representation="dense")
        with self.assertRaisesRegex(MemoryError, "No buildings were dropped"):
            dataset[0]

    def test_opt_in_repair_preserves_uid_and_raw_annotations_and_audits_components(self) -> None:
        invalid = "POLYGON ((0 0,10 10,0 10,10 0,0 0))"
        self.write_scene("train", [("same-building", invalid)])
        self.build()
        source = self.root / "raw/train/labels/train_pre_disaster.json"
        before = source.read_bytes()
        sample = InstanceSegmentationDataset(self.index, self.root, "train",
                                             geometry_policy="make-valid-polygonal")[0]
        self.assertEqual(sample.reference_uids, ["same-building"])
        self.assertIn("compact_masks", sample.target)
        self.assertNotIn("masks", sample.target)
        self.assertEqual(len(sample.geometry_audit), 1)
        self.assertEqual(sample.geometry_audit[0]["reference_uid"], "same-building")
        self.assertEqual(sample.geometry_audit[0]["polygon_components"], 2)
        self.assertEqual(source.read_bytes(), before)
        report = audit_instance_targets(self.root, self.index, self.root / "repaired-audit.json",
                                         geometry_policy="make-valid-polygonal")
        self.assertTrue(report["repairs"])
        self.assertFalse(report["raw_annotation_edits"])
        self.assertFalse(report["target_drops"])
        self.assertEqual(report["counts"]["train"]["repaired_buildings"], 1)
        self.assertEqual(len(report["repair_records"]), 1)

    def test_repair_audit_records_non_polygon_parts_and_preserves_valid_holes(self) -> None:
        spike = "POLYGON ((0 0,4 0,4 4,0 4,0 0,-2 0,0 0))"
        repaired = rasterize_footprint_window(spike, 16, 12, geometry_policy="make-valid-polygonal")
        self.assertEqual(repaired.area, 16)
        self.assertEqual(repaired.repair["make_valid_output_type"], "GeometryCollection")
        self.assertIn("LineString", repaired.repair["non_polygon_components_excluded_from_area_mask"])
        valid = "POLYGON ((0 0,10 0,10 10,0 10,0 0),(2 2,2 8,8 8,8 2,2 2))"
        unchanged = rasterize_footprint_window(valid, 16, 12, geometry_policy="make-valid-polygonal")
        self.assertIsNone(unchanged.repair)
        self.assertEqual(unchanged.area, 64)
        line_only = "POLYGON ((0 0,1 0,2 0,0 0))"
        with self.assertRaisesRegex(MaskGeometryError, "repair_has_no_valid_polygonal_area"):
            rasterize_footprint_window(line_only, 16, 12, geometry_policy="make-valid-polygonal")

    def test_default_compact_targets_respect_small_storage_guard(self) -> None:
        self.build()
        sample = InstanceSegmentationDataset(self.index, self.root, "train", max_mask_bytes=100)[0]
        self.assertEqual(sample.target["compact_masks"].stored_bytes, 98)
        self.assertNotIn("masks", sample.target)

    def test_zero_pixel_valid_footprint_refused_at_loading(self) -> None:
        self.write_scene("train", [("tiny", "POLYGON ((.1 .1,.4 .1,.4 .4,.1 .4,.1 .1))")])
        self.build()
        dataset = InstanceSegmentationDataset(self.index, self.root, "train")
        with self.assertRaisesRegex(ValueError, "zero_rasterized_pixels"):
            dataset[0]

    def test_audit_keeps_problem_uids_raw_data_and_reference_populations(self) -> None:
        self.write_scene("train", [("invalid", "POLYGON ((0 0,10 10,0 10,10 0,0 0))"), ("valid", RECTANGLE)])
        self.write_scene("unassigned", [("unassigned", RECTANGLE)])
        self.build()
        source = self.root / "raw/train/labels/train_pre_disaster.json"
        before = source.read_bytes()
        output = self.root / "audit.json"
        audit = audit_instance_targets(self.root, self.index, output)
        self.assertEqual(audit["counts"]["train"]["buildings"], 2)
        self.assertEqual(audit["counts"]["train"]["blocked:invalid_geometry"], 1)
        self.assertEqual(audit["counts"]["train"]["usable_masks"], 1)
        self.assertEqual(audit["problems"][0]["reference_uid"], "invalid")
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(audit["largest_dense_target_stacks"]["train"]["dense_uint8_mask_bytes"], 2 * 16 * 12)
        self.assertFalse(audit["repairs"])
        self.assertFalse(audit["target_drops"])
        with self.assertRaises(FileExistsError):
            audit_instance_targets(self.root, self.index, output)
        source.write_bytes(before + b"\n")
        with self.assertRaisesRegex(ValueError, "changed since indexing"):
            audit_instance_targets(self.root, self.index, self.root / "changed-audit.json")


if __name__ == "__main__":
    unittest.main()
