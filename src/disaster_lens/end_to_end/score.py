"""Score end-to-end damage predictions against the annotations (CPU only).

    python -m disaster_lens.end_to_end.score --run-dir OUT --output scores.json

Scenario B (annotated footprints): macro-F1 / accuracy over the labeled buildings.
Scenario A (detected footprints), at each detection score threshold:
  matched                 labeled buildings found by a detection (mask IoU >= 0.5); one best-scoring detection each
  damage_given_match      macro-F1 / accuracy of the damage class on those matched buildings
  end_to_end              per-class F1 where unmatched labeled buildings count as misses and detections that match
                          no labeled building (or duplicate a match) count as false positives in their predicted class
Detections matched to a footprint without a damage label are excluded from scoring.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from disaster_lens.data.manifest import DAMAGE_LABELS

COLUMNS = [f"probability_{label}" for label in DAMAGE_LABELS]
THRESHOLDS = (0.5, 0.35)


def macro_f1(truth: np.ndarray, pred: np.ndarray, fp_extra: np.ndarray | None = None, fn_extra: np.ndarray | None = None) -> float:
    scores = []
    for c in range(len(DAMAGE_LABELS)):
        tp = np.sum((truth == c) & (pred == c))
        fp = np.sum((truth != c) & (pred == c)) + (fp_extra[c] if fp_extra is not None else 0)
        fn = np.sum((truth == c) & (pred != c)) + (fn_extra[c] if fn_extra is not None else 0)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.)
    return float(np.mean(scores))


def _rows(path: Path) -> list[dict]:
    """Rows of a prediction CSV; earlier runs with several alignment arms keep only the unshifted one."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and "arm" in rows[0]:
        rows = [row for row in rows if row["arm"].endswith("_identity")]
    return rows


def predicted_class(row: dict) -> int:
    return int(np.argmax([float(row[c]) for c in COLUMNS]))


def score_reference(rows: list[dict]) -> dict:
    truth = np.array([DAMAGE_LABELS.index(r["reference_label"]) for r in rows])
    pred = np.array([predicted_class(r) for r in rows])
    return {"buildings": len(rows), "macro_f1": macro_f1(truth, pred), "accuracy": float((truth == pred).mean())}


def score_detected(rows: list[dict], truth_by_uid: dict[str, int], threshold: float) -> dict:
    best: dict[str, tuple[float, int]] = {}
    fp = np.zeros(len(DAMAGE_LABELS))
    for r in rows:
        score = float(r["score"])
        if score < threshold:
            continue
        pred, uid = predicted_class(r), r["matched_reference_uid"]
        if uid and uid not in truth_by_uid:
            continue                                     # matched a footprint without a damage label: not scorable
        if not uid:
            fp[pred] += 1
        elif uid not in best or score > best[uid][0]:
            if uid in best:
                fp[best[uid][1]] += 1
            best[uid] = (score, pred)
        else:
            fp[pred] += 1
    truth = np.array([truth_by_uid[u] for u in best])
    pred = np.array([p for _, p in best.values()])
    fn = np.zeros(len(DAMAGE_LABELS))
    for uid, c in truth_by_uid.items():
        if uid not in best:
            fn[c] += 1
    return {"matched": len(best), "recall_of_labeled": len(best) / len(truth_by_uid),
            "damage_given_match_macro_f1": macro_f1(truth, pred), "damage_given_match_accuracy": float((truth == pred).mean()),
            "end_to_end_macro_f1": macro_f1(truth, pred, fp_extra=fp, fn_extra=fn), "unmatched_false_positives": int(fp.sum())}


def score_run(run_dir: Path, thresholds=THRESHOLDS) -> dict:
    reference = _rows(run_dir / "reference_predictions.csv")
    detections = _rows(run_dir / "detection_predictions.csv")
    truth_by_uid = {r["building_uid"]: DAMAGE_LABELS.index(r["reference_label"]) for r in reference}
    return {"annotated_footprints": score_reference(reference),
            "detected": {f"{t:g}": score_detected(detections, truth_by_uid, t) for t in thresholds}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    check = metrics.get("reference_check") or metrics.get("reference_identity_check")
    if check is not None and not check["passed"]:
        raise ValueError("The run failed its reference check.")
    report = score_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
