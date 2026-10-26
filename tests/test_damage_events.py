"""Event metrics are derived from synthetic saved predictions, without inference."""

import csv
import tempfile
import unittest
from pathlib import Path

from disaster_lens.damage.events import evaluate_by_event
from disaster_lens.data.manifest import DAMAGE_LABELS, FIELDS


class EventEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "validation.csv"
        self.predictions = self.root / "predictions.csv"
        self.rows = []
        references = []
        for index, label in enumerate(DAMAGE_LABELS):
            event = "event-a" if index < 2 else "event-b"
            scene = f"{event}-scene"
            references.append({
                "source_split": "train", "match_status": "matched",
                "scene_id": scene, "building_uid": f"uid-{index}",
                "event": event, "disaster_type": "wind", "damage_label": label,
            })
            self.rows.append({
                "scene_id": scene, "building_uid": f"uid-{index}",
                "reference_label": label,
                "predicted_label": "no-damage" if index == 1 else label,
            })
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(references)
        self.write_predictions()

    def write_predictions(self) -> None:
        with self.predictions.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("scene_id", "building_uid", "reference_label", "predicted_label"),
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def test_event_metrics_preserve_support_and_reconcile_pooled_predictions(self) -> None:
        output = self.root / "report"
        report = evaluate_by_event(self.predictions, self.manifest, output)
        self.assertEqual(report["overall"]["buildings"], 4)
        self.assertAlmostEqual(report["overall"]["macro_f1"], 2 / 3)
        first = report["events"]["event-a"]
        self.assertAlmostEqual(first["macro_f1"], 1 / 6)
        self.assertAlmostEqual(first["macro_f1_supported_classes"], 1 / 3)
        self.assertEqual(first["scenes"], 1)
        self.assertEqual(first["per_class"]["minor-damage"]["support"], 1)
        self.assertEqual(first["per_class"]["minor-damage"]["f1"], 0)
        self.assertFalse(first["per_class"]["major-damage"]["has_reference_support"])
        self.assertEqual(first["per_class"]["no-damage"]["predicted_count"], 2)
        second = report["events"]["event-b"]
        self.assertEqual(second["macro_f1_supported_classes"], 1)
        with (output / "event_class_metrics.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)
        before = (output / "metrics.json").read_bytes()
        with self.assertRaises(FileExistsError):
            evaluate_by_event(self.predictions, self.manifest, output)
        self.assertEqual((output / "metrics.json").read_bytes(), before)

    def test_rejects_missing_duplicate_unknown_and_misaligned_predictions(self) -> None:
        original = [dict(row) for row in self.rows]
        cases = [
            (original[:-1], "coverage"),
            (original + [original[0]], "Duplicate prediction"),
            ([{**original[0], "building_uid": "other"}, *original[1:]], "absent from"),
            ([{**original[0], "reference_label": "minor-damage"}, *original[1:]], "reference label"),
            ([{**original[0], "predicted_label": "unknown"}, *original[1:]], "Unknown predicted"),
        ]
        for rows, message in cases:
            with self.subTest(message=message):
                self.rows = rows
                self.write_predictions()
                output = self.root / "invalid"
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_by_event(self.predictions, self.manifest, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
