"""Break saved validation predictions down by event and original damage class."""

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.data.split import read_examples
from disaster_lens.damage.train_cached import classification_metrics
from disaster_lens.data.scenes import CLASS_TO_INDEX


def evaluate_by_event(predictions: Path, manifest: Path, output_dir: Path) -> dict:
    """Join by scene/building identity; never change splits or run a model."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    examples = read_examples(manifest)
    expected = {(row["scene_id"], row["building_uid"]): row for row in examples}
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    targets: list[int] = []
    predicted: list[int] = []
    with predictions.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["scene_id"], row["building_uid"]
            if key not in expected:
                raise ValueError(f"Prediction identity is absent from the validation manifest: {key}")
            if key in seen:
                raise ValueError(f"Duplicate prediction identity: {key}")
            seen.add(key)
            reference = expected[key]
            if row["reference_label"] != reference["damage_label"]:
                raise ValueError(f"Prediction reference label differs from the manifest: {key}")
            if row["predicted_label"] not in CLASS_TO_INDEX:
                raise ValueError(f"Unknown predicted damage label: {row['predicted_label']}")
            grouped[reference["event"]].append({**row, **reference, "predicted_label": row["predicted_label"]})
            targets.append(CLASS_TO_INDEX[reference["damage_label"]])
            predicted.append(CLASS_TO_INDEX[row["predicted_label"]])
    if seen != set(expected):
        raise ValueError("Prediction coverage differs from the complete validation manifest.")
    events = {}
    class_rows = []
    event_rows = []
    for event, rows in sorted(grouped.items()):
        event_targets = [CLASS_TO_INDEX[row["damage_label"]] for row in rows]
        event_predictions = [CLASS_TO_INDEX[row["predicted_label"]] for row in rows]
        metrics = classification_metrics(event_targets, event_predictions)
        predicted_counts = Counter(row["predicted_label"] for row in rows)
        supported = [label for label in DAMAGE_LABELS if metrics["per_class"][label]["support"] > 0]
        metrics.update({
            "scenes": len({row["scene_id"] for row in rows}),
            "disaster_types": sorted({row["disaster_type"] for row in rows}),
            "supported_classes": supported,
            "macro_f1_supported_classes": sum(metrics["per_class"][label]["f1"] for label in supported) / len(supported),
        })
        event_rows.append({
            "event": event, "disaster_types": "|".join(metrics["disaster_types"]),
            "scenes": metrics["scenes"], "buildings": metrics["buildings"],
            "accuracy": metrics["accuracy"], "macro_f1_all_four_classes": metrics["macro_f1"],
            "macro_f1_supported_classes": metrics["macro_f1_supported_classes"],
            "supported_classes": "|".join(supported),
        })
        for label in DAMAGE_LABELS:
            scores = metrics["per_class"][label]
            scores.update(predicted_count=predicted_counts[label], has_reference_support=scores["support"] > 0)
            class_rows.append({
                "event": event, "damage_label": label,
                "event_scenes": metrics["scenes"], "event_buildings": metrics["buildings"], **scores,
            })
        events[event] = metrics
    overall = classification_metrics(targets, predicted)
    pooled_confusion = [
        [sum(metrics["confusion_matrix"][i][j] for metrics in events.values())
         for j in range(len(DAMAGE_LABELS))]
        for i in range(len(DAMAGE_LABELS))
    ]
    if pooled_confusion != overall["confusion_matrix"]:
        raise RuntimeError("Per-event confusion matrices do not reconcile with pooled predictions.")
    sources = {}
    for name, path in (("predictions", predictions), ("validation_manifest", manifest)):
        with path.open("rb") as stream:
            sources[name] = {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    report = {
        "sources": sources, "class_to_index": CLASS_TO_INDEX,
        "metric_policy": {
            "primary": "Pooled classification macro-F1; selection belongs to the run config.",
            "event_macro_f1": "All four classes, with zero_division=0; missing classes can lower this diagnostic.",
            "event_macro_f1_supported_classes": "Mean F1 only over classes with positive reference support; diagnostic only.",
            "absent_classes": "Raw metrics use zero_division=0; has_reference_support=false means class F1/recall are not assessable.",
            "confusion_matrix": "Rows=reference, columns=predicted, in class_to_index order.",
            "scope": "Existing within-event validation predictions; not held-out-event evaluation.",
        },
        "overall": overall, "events": events,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".events-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        (staging / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for name, records in (("event_summary.csv", event_rows), ("event_class_metrics.csv", class_rows)):
            with (staging / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        staging.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/processed/splits/seed42/validation.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_by_event(args.predictions, args.manifest, args.output_dir)
    print(f"Saved {len(report['events'])} event breakdowns for {report['overall']['buildings']:,} buildings.")
    for event, metrics in report["events"].items():
        print(f"{event}: {metrics['buildings']:,} buildings; "
              f"four-class macro-F1={metrics['macro_f1']:.4f}; "
              f"supported-class macro-F1={metrics['macro_f1_supported_classes']:.4f}")


if __name__ == "__main__":
    main()
