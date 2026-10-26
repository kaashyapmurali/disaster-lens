"""Manifest tests use synthetic annotations; no dataset download is required."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from disaster_lens.data.manifest import build_manifest


def feature(uid: str, label: str | None = None) -> dict:
    properties = {"feature_type": "building", "uid": uid}
    if label is not None:
        properties["subtype"] = label
    return {
        "properties": properties,
        "wkt": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
    }


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name)
        self.output = self.data_dir / "processed/train_buildings.csv"
        self.scene = "example-event_00000001"

    def write_scene(self, pre: list[dict], post: list[dict]) -> None:
        for phase, features in (("pre", pre), ("post", post)):
            root = self.data_dir / "raw/train"
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "labels").mkdir(parents=True, exist_ok=True)
            (root / "images" / f"{self.scene}_{phase}_disaster.png").touch()
            record = {
                "features": {"xy": features},
                "metadata": {
                    "disaster": "example-event",
                    "disaster_type": "wind",
                    "capture_date": f"{phase}-date",
                },
            }
            (root / "labels" / f"{self.scene}_{phase}_disaster.json").write_text(
                json.dumps(record)
            )

    def test_keeps_four_classes_and_matches_by_uid_not_order(self) -> None:
        labels = ("no-damage", "minor-damage", "major-damage", "destroyed")
        self.write_scene(
            [feature(str(i)) for i in range(4)],
            [feature(str(i), labels[i]) for i in reversed(range(4))],
        )
        counts = build_manifest(self.data_dir, "train", self.output)
        with self.output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(counts["rows"], 4)
        self.assertEqual({r["damage_label"] for r in rows}, set(labels))
        for row in rows:
            self.assertEqual(row["damage_label"], labels[int(row["building_uid"])])
            self.assertEqual(row["event"], "example-event")
            self.assertEqual(row["disaster_type"], "wind")
            self.assertEqual(row["raw_damage_label"], row["damage_label"])

    def test_excludes_unknown_missing_and_unmatched_without_changing_source(self) -> None:
        pre = [feature(uid) for uid in ("known", "unknown", "missing", "pre-only")]
        post = [
            feature("known", "minor-damage"),
            feature("unknown", "un-classified"),
            feature("missing"),
            feature("post-only", "destroyed"),
        ]
        self.write_scene(pre, post)
        source = self.data_dir / "raw/train/labels" / f"{self.scene}_post_disaster.json"
        before = source.read_bytes()
        counts = build_manifest(self.data_dir, "train", self.output)
        with self.output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["building_uid"] for r in rows], ["known"])
        self.assertEqual(counts["excluded:unknown_damage"], 3)
        self.assertEqual(counts["excluded:unmatched_building"], 1)
        self.assertEqual(source.read_bytes(), before)

    def test_refuses_overwrite(self) -> None:
        self.write_scene([feature("a")], [feature("a", "no-damage")])
        build_manifest(self.data_dir, "train", self.output)
        with self.assertRaises(FileExistsError):
            build_manifest(self.data_dir, "train", self.output)

    def test_duplicate_uids_fail_without_publishing_partial_manifest(self) -> None:
        self.write_scene([feature("a"), feature("a")], [feature("a", "no-damage")])
        with self.assertRaisesRegex(ValueError, "duplicate building UID"):
            build_manifest(self.data_dir, "train", self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".csv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
