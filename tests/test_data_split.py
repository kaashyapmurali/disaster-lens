"""Split tests use small synthetic manifests, not satellite imagery."""

import csv
import tempfile
import unittest
from pathlib import Path

from disaster_lens.data.manifest import DAMAGE_LABELS, FIELDS
from disaster_lens.data.split import create_split, read_examples


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "buildings.csv"
        self.rows = []
        for event, hazard in (("event-a", "wind"), ("event-b", "fire")):
            for scene_index in range(10):
                scene = f"{event}_{scene_index:08d}"
                for label in DAMAGE_LABELS:
                    self.rows.append({
                        "source_split": "train", "scene_id": scene,
                        "building_uid": f"{scene}-{label}", "event": event,
                        "disaster_type": hazard, "damage_label": label,
                        "match_status": "matched", "raw_damage_label": label,
                    })
        self.write_manifest()

    def write_manifest(self) -> None:
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def read_output(self, directory: Path, split: str) -> list[dict[str, str]]:
        with (directory / f"{split}.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_scene_isolation_coverage_and_reproducibility(self) -> None:
        first, second = self.root / "first", self.root / "second"
        report = create_split(self.manifest, first, seed=42)
        create_split(self.manifest, second, seed=42)
        train = self.read_output(first, "train")
        validation = self.read_output(first, "validation")
        self.assertFalse({r["scene_id"] for r in train} & {r["scene_id"] for r in validation})
        self.assertFalse({r["building_uid"] for r in train} & {r["building_uid"] for r in validation})
        self.assertEqual(len(train) + len(validation), len(self.rows))
        self.assertEqual({r["building_uid"] for r in train + validation},
                         {r["building_uid"] for r in self.rows})
        self.assertEqual((first / "scene_assignments.csv").read_bytes(),
                         (second / "scene_assignments.csv").read_bytes())
        self.assertEqual(len(validation), len(self.rows) // 5)
        self.assertTrue(all(row["present_in_both"] for row in report["strata"]))
        for rows in (train, validation):
            self.assertEqual({r["damage_label"] for r in rows}, set(DAMAGE_LABELS))
            self.assertEqual({r["event"] for r in rows}, {"event-a", "event-b"})
        with self.assertRaises(FileExistsError):
            create_split(self.manifest, first)

    def test_reports_stratum_with_only_one_supporting_scene(self) -> None:
        self.rows.append({
            "source_split": "train", "scene_id": "rare_00000001", "building_uid": "rare-building",
            "event": "rare", "disaster_type": "tsunami", "damage_label": "minor-damage",
            "match_status": "matched", "raw_damage_label": "minor-damage",
        })
        self.write_manifest()
        report = create_split(self.manifest, self.root / "rare-split")
        rare = next(row for row in report["strata"] if row["event"] == "rare")
        self.assertEqual(rare["supporting_scenes"], 1)
        self.assertFalse(rare["present_in_both"])
        self.assertEqual(rare["train"] + rare["validation"], 1)

    def test_rejects_unknown_labels_or_test_sources(self) -> None:
        for key, value in (("damage_label", "unknown"), ("source_split", "test")):
            with self.subTest(field=key):
                original = self.rows[0][key]
                self.rows[0][key] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "source-training manifest"):
                    read_examples(self.manifest)
                self.rows[0][key] = original

    def test_rejects_duplicate_or_cross_scene_building_ids(self) -> None:
        self.rows[1]["building_uid"] = self.rows[0]["building_uid"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate example"):
            read_examples(self.manifest)
        self.rows[1]["scene_id"] = "different-scene"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "spans scenes"):
            read_examples(self.manifest)


if __name__ == "__main__":
    unittest.main()
