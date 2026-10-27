"""Aggregate numbers for the report bundle (CPU; reads local model outputs).

    uv run --group notebook python report/scripts/metrics.py

Writes report/bundle/data/: aggregate-metrics.json (per damage model, detector summaries), class-recall.json (recall per
class with known and detected footprints, per damage model), test-flow.json (where the labeled test buildings end
up), no-change.json (pre-event image given as both dates) and change-measures.csv (validation deciles of simple image
change against predicted severity).
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from change_features import FEATURES, binned, compute, severity_of
from paths import (
    CHANGE_CACHE,
    CLASSES,
    DAMAGE_RUN,
    DATA,
    DATASETS,
    NO_CHANGE_RUN,
    ROOT,
    VALIDATION_EXPORT,
    VALIDATION_MANIFEST,
    VALIDATION_PREDICTIONS,
    write_json,
)

PROBABILITIES = [f"probability_{c}" for c in CLASSES]
SCORE_THRESHOLD = 0.5
MODELS = [  # label, training run, validation end-to-end scores, test end-to-end run
    ("Post-only ResNet-18", "baseline-l4-20261017-175000", "end-to-end-crops-ep8-resnet-post-only-v1.json",
     "outputs/local/test/e2e-resnet-post-only"),
    ("Paired ResNet-18", "fusion-head-l4-20261017-01", "end-to-end-crops-ep8-resnet-paired-v1.json",
     "outputs/local/test/e2e-resnet-paired"),
    ("Frozen DINOv3 ViT-L", "vit-fusion-head-l4-20261018-01", "end-to-end-crops-ep8-vit-frozen-v1.json",
     "outputs/modal/test-e2e-vit-frozen"),
    ("Fine-tuned DINOv3 ViT-L", "vit-finetune-bf16-l4-20261020-01", "end-to-end-crops-ep8-v1.json",
     "outputs/modal/test-e2e-vit-finetuned"),
]
DETECTORS = [  # label, verified export metrics, error-analysis summary
    ("Native scale, whole scenes", "outputs/modal/segmenter-full-l4-01/validation_predictions/metrics.json",
     "outputs/analysis/instance-errors-v1/report.json"),
    ("2× crops and tiles (selected)", "outputs/modal/segmenter-crops-l4-02/validation_predictions_tile512/metrics.json",
     "outputs/analysis/instance-errors-crops-ep8-v1/report.json"),
]
TEST_DETECTOR = ("2× crops and tiles, test split", "outputs/modal/segmenter-crops-l4-02/test_predictions_tile512/metrics.json",
                 "outputs/analysis/instance-errors-crops-ep8-test-v1/report.json")
CHANGE_EVENTS = [("santa-rosa-wildfire", "Wildfire (Santa Rosa)"), ("hurricane-michael", "Wind (Michael)"),
                 ("midwest-flooding", "Flood (Midwest)"), ("mexico-earthquake", "Earthquake (Mexico)")]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def predicted(row: dict) -> int:
    return int(np.argmax([float(row[c]) for c in PROBABILITIES]))


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def best_matches(run: Path, truth: dict[str, int]) -> dict[str, int]:
    """Labeled building uid -> predicted class of its best-scoring matched detection (score >= 0.5)."""
    best: dict[str, tuple[float, int]] = {}
    for row in rows(run / "detection_predictions.csv"):
        score, uid = float(row["score"]), row["matched_reference_uid"]
        if score < SCORE_THRESHOLD or not uid or uid not in truth:
            continue
        if uid not in best or score > best[uid][0]:
            best[uid] = (score, predicted(row))
    return {uid: cls for uid, (_, cls) in best.items()}


def aggregate_metrics() -> dict:
    models = []
    for label, run, validation_file, _ in MODELS:
        result = load(f"outputs/modal/{run}.result.json")["best_validation"]
        validation = load(f"outputs/analysis/{validation_file}")
        assert abs(validation["reference"]["identity"]["all"]["macro_f1"] - result["macro_f1"]) < 1e-3   # reproduced
        models.append({"label": label, "validation_macro_f1": result["macro_f1"], "per_class": result["per_class"]})

    def detector(label: str, metrics_path: str, errors_path: str) -> dict:
        pooled = load(metrics_path)["pooled"]
        small = load(errors_path)["references_by_size"]["small"]
        at = pooled["at_score_threshold_iou50"]
        return {"label": label, "ap": pooled["ap_50_95"], "ap50": pooled["ap50"], "recall": at["recall"],
                "precision": at["precision"], "small": small["matched"] / small["total"],
                "unmatched": at["false_positives"], "buildings": pooled["reference_buildings"], "scenes": pooled["scenes"]}

    return {"damage_models": models, "detectors": [detector(*d) for d in DETECTORS], "test_detector": detector(*TEST_DETECTOR)}


def class_recall() -> dict:
    out = {"classes": CLASSES, "known_footprints": {}, "end_to_end": {}}
    for label, _, _, run in MODELS:
        reference = rows(ROOT / run / "reference_predictions.csv")
        truth = {r["building_uid"]: CLASSES.index(r["reference_label"]) for r in reference}
        total, hit = Counter(), Counter()
        for r in reference:
            t = CLASSES.index(r["reference_label"])
            total[t] += 1
            hit[t] += predicted(r) == t
        matches = best_matches(ROOT / run, truth)
        found, correct = Counter(), Counter()
        for uid, t in truth.items():
            if uid in matches:
                found[t] += 1
                correct[t] += matches[uid] == t
        out["known_footprints"][label] = {CLASSES[c]: hit[c] / total[c] for c in range(4)}
        out["end_to_end"][label] = {CLASSES[c]: {"buildings": total[c], "found": found[c] / total[c],
                                                 "correct_given_found": correct[c] / found[c],
                                                 "recall": correct[c] / total[c]} for c in range(4)}
    return out


def test_flow() -> dict:
    """Fine-tuned ViT on the test split: found / missed, class outcome of found buildings, extra detections."""
    reference = rows(DAMAGE_RUN / "reference_predictions.csv")
    truth = {r["building_uid"]: CLASSES.index(r["reference_label"]) for r in reference}
    matches = best_matches(DAMAGE_RUN, truth)
    extra = Counter()
    for row in rows(DAMAGE_RUN / "detection_predictions.csv"):
        uid = row["matched_reference_uid"]
        if float(row["score"]) >= SCORE_THRESHOLD and not uid:
            extra[CLASSES[predicted(row)]] += 1
    found, missed, outcome = Counter(), Counter(), Counter()
    for uid, t in truth.items():
        if uid in matches:
            p = matches[uid]
            found[CLASSES[t]] += 1
            outcome["correct" if p == t else "adjacent" if abs(p - t) == 1 else "far"] += 1
        else:
            missed[CLASSES[t]] += 1
    confusion = [[0] * 4 for _ in CLASSES]
    footprints = Counter()
    for r in reference:
        t, p = CLASSES.index(r["reference_label"]), predicted(r)
        confusion[t][p] += 1
        footprints["correct" if p == t else "adjacent" if abs(p - t) == 1 else "far"] += 1
    return {"labeled": len(truth), "found": sum(found.values()), "missed": sum(missed.values()),
            "found_by_class": dict(found), "missed_by_class": dict(missed), "found_outcome": dict(outcome),
            "extra": sum(extra.values()), "extra_by_class": dict(extra), "footprints_outcome": dict(footprints),
            "footprints_confusion": confusion}


def no_change() -> dict:
    """Predicted class counts per reference class: real images vs. the pre-event image given as both dates."""
    real = {r["building_uid"]: r for r in rows(DAMAGE_RUN / "reference_predictions.csv")}
    probe = {r["building_uid"]: r for r in rows(NO_CHANGE_RUN / "reference_predictions.csv")}
    if set(real) != set(probe) or not json.loads((NO_CHANGE_RUN / "metrics.json").read_text()).get("no_change"):
        raise ValueError("The no-change run must cover the same buildings and be marked no_change.")
    out = {}
    for c in CLASSES:
        ids = [u for u, r in real.items() if r["reference_label"] == c]
        out[c] = {"n": len(ids),
                  "real": {k: sum(CLASSES[predicted(real[u])] == k for u in ids) for k in CLASSES},
                  "no_change": {k: sum(CLASSES[predicted(probe[u])] == k for u in ids) for k in CLASSES}}
    return out


def change_measures() -> list[dict]:
    """Per-event decile summaries of image change against predicted and reference severity (validation)."""
    if not CHANGE_CACHE.is_file():
        computed = compute(VALIDATION_EXPORT, VALIDATION_PREDICTIONS, VALIDATION_MANIFEST, DATASETS)
        CHANGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CHANGE_CACHE.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(computed[0]))
            writer.writeheader()
            writer.writerows(computed)
    measured = rows(CHANGE_CACHE)
    table = []
    for event, name in CHANGE_EVENTS:
        subset = [r for r in measured if r["event"] == event]
        for feature in FEATURES:
            x = np.array([float(r[feature]) for r in subset])
            severity = np.array([float(r["predicted_severity"]) for r in subset])
            reference = np.array([severity_of(r["true_label"]) for r in subset])
            centers, mean_predicted, support = binned(x, severity)
            _, mean_reference, _ = binned(x, reference)
            for c, p, r, n in zip(centers, mean_predicted, mean_reference, support, strict=True):
                table.append({"event": event, "event_label": name, "measure": feature, "bin_center": float(c),
                              "mean_predicted_severity": float(p), "mean_reference_severity": float(r),
                              "buildings": int(n)})
    return table


def main() -> None:
    write_json(DATA / "aggregate-metrics.json", aggregate_metrics())
    write_json(DATA / "class-recall.json", class_recall())
    write_json(DATA / "test-flow.json", test_flow())
    write_json(DATA / "no-change.json", no_change())
    table = change_measures()
    with (DATA / "change-measures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("metrics: wrote", ", ".join(sorted(p.name for p in DATA.iterdir())))


if __name__ == "__main__":
    main()
