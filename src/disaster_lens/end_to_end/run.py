"""Damage predictions for detected buildings (scenario A) and annotated footprints (scenario B) in one pass.

    python -m disaster_lens.end_to_end.run --checkpoint DAMAGE.pt --export-dir DETECTOR_EXPORT \
        --manifest validation.csv --data-dir datasets --output-dir OUT [--saved-predictions CSV] [--encoder-dir DIR]

Each date is encoded once per scene; the damage model then classifies two box sets: the annotated footprints
(scenario B, which must reproduce the model's recorded validation results) and the detected footprints from the
instance export (scenario A). Footprint boxes are used at the same position on the pre- and post-event images.
Predictions only: scoring against labels is done by `end_to_end.score`.
"""

import argparse
import csv
import json
import math
import tempfile
import time
from pathlib import Path

import torch
from torch import Tensor

from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.detection.mask_metrics import verified_export

PROBABILITY_COLUMNS = [f"probability_{label}" for label in DAMAGE_LABELS]
DETECTION_FLOOR = 0.1
ARMS = ("reference", "detected")


def load_detections(export_dir: Path, floor: float) -> tuple[dict, dict]:
    """Verified instance export -> per-scene detections scoring >= floor (native pixel coordinates)."""
    metrics, path, digest = verified_export(export_dir)
    scenes = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            keep = [d for d in row["detections"] if d["score"] >= floor]
            scenes[row["scene_id"]] = {
                "width": row["width"], "height": row["height"], "event": row["event"],
                "ids": [d["building_id"] for d in keep], "scores": [d["score"] for d in keep],
                "boxes": [d["box"] for d in keep], "matched": [d["matched_reference_uid_mask_iou50"] for d in keep]}
    return scenes, {"predictions_sha256": digest, "epoch": metrics["provenance"]["epoch"],
                    "run_id": metrics["provenance"]["run_id"]}


def scene_arms(reference_boxes: Tensor, detections: dict, image_size: int, device: torch.device) -> dict[str, tuple[Tensor, Tensor]]:
    """(pre boxes, post boxes) per arm; boxes are identical on both dates. Scenes must be at native size."""
    if detections["width"] != image_size or detections["height"] != image_size:
        raise ValueError("Detections and damage inputs must share the native image grid.")
    reference = reference_boxes.to(device)
    detected = torch.tensor(detections["boxes"], dtype=torch.float32, device=device).reshape(-1, 4)
    return {"reference": (reference, reference), "detected": (detected, detected)}


def run_end_to_end(*, predictor, dataset, detections: dict, output_dir: Path, device: torch.device,
                   num_workers: int = 0, reference_check: dict | None = None, progress=print,
                   provenance: dict | None = None, no_change: bool = False) -> dict:
    """Predict both arms for every scene, publish CSVs atomically, then verify the annotated-footprint arm.

    no_change: diagnostic; the pre-event image is also given as the post-event image (no visible change).
    """
    from torch.utils.data import DataLoader

    from disaster_lens.data.scenes import collate_pairs

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    scene_ids = [scene.scene_id for scene in dataset.scenes]
    if not set(scene_ids) <= set(detections):
        raise ValueError("The instance export must cover every scene of the damage manifest.")
    # Evaluation covers the manifest's scenes (those with labeled buildings); other exported scenes are ignored.
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_pairs)
    reference_rows, detection_rows = [], []
    started = time.perf_counter()
    for index, batch in enumerate(loader, start=1):
        scene_id = batch.scene_ids[0]
        arms = scene_arms(batch.pre_boxes[0], detections[scene_id], dataset.image_size, device)
        pre_images = batch.pre_images.to(device)
        post_images = pre_images if no_change else batch.post_images.to(device)
        probabilities = predictor.predict(pre_images, post_images, arms, device)
        for uid, label, row in zip(batch.building_uids[0], batch.labels[0].tolist(), probabilities["reference"].tolist(), strict=True):
            reference_rows.append({"scene_id": scene_id, "building_uid": uid, "reference_label": DAMAGE_LABELS[label],
                                   **dict(zip(PROBABILITY_COLUMNS, map(repr, row), strict=True))})
        d = detections[scene_id]
        for did, score, matched, row in zip(d["ids"], d["scores"], d["matched"], probabilities["detected"].tolist(), strict=True):
            detection_rows.append({"scene_id": scene_id, "building_id": did, "score": score, "matched_reference_uid": matched or "",
                                   **dict(zip(PROBABILITY_COLUMNS, map(repr, row), strict=True))})
        if index % 50 == 0 or index == len(loader):
            progress(f"end-to-end: {index}/{len(loader)} scenes; {time.perf_counter() - started:.0f}s")
    check = verify_reference(reference_rows, reference_check) if reference_check else None
    metrics = {"scenes": len(scene_ids), "reference_rows": len(reference_rows), "detection_rows": len(detection_rows),
               "elapsed_seconds": time.perf_counter() - started, "precision": predictor.precision,
               "reference_check": check, "no_change": no_change, "provenance": provenance or {}}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        for name, rows in (("reference_predictions.csv", reference_rows), ("detection_predictions.csv", detection_rows)):
            with (staging / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (staging / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        staging.rename(output_dir)
    if check is not None and not check["passed"]:
        raise ValueError(f"Annotated-footprint predictions do not reproduce the model's recorded results: {check}")
    return metrics


def verify_reference(rows: list[dict], check: dict) -> dict:
    if "saved_predictions" in check:
        return verify_saved_predictions(rows, Path(check["saved_predictions"]))
    return verify_confusion(rows, check["confusion_matrix"], check["macro_f1"])


def verify_confusion(rows: list[dict], recorded: list[list[int]], recorded_macro_f1: float, *,
                     max_changed_fraction: float = 1e-3, max_macro_f1_diff: float = 1e-3) -> dict:
    """Reproduce the recorded validation confusion matrix (when per-building predictions were not saved)."""
    confusion = [[0] * len(DAMAGE_LABELS) for _ in DAMAGE_LABELS]
    for row in rows:
        predicted = max(range(len(DAMAGE_LABELS)), key=lambda i: float(row[PROBABILITY_COLUMNS[i]]))
        confusion[DAMAGE_LABELS.index(row["reference_label"])][predicted] += 1
    total = sum(map(sum, confusion))
    changed = sum(abs(a - b) for ours, theirs in zip(confusion, recorded, strict=True) for a, b in zip(ours, theirs, strict=True)) / 2
    f1 = []
    for c in range(len(DAMAGE_LABELS)):
        tp = confusion[c][c]
        denominator = 2 * tp + sum(confusion[c]) - tp + sum(row[c] for row in confusion) - tp
        f1.append(2 * tp / denominator if denominator else 0.)
    macro_f1 = sum(f1) / len(f1)
    result = {"method": "recorded confusion matrix", "buildings": total, "changed_predictions_upper_bound": changed,
              "macro_f1": macro_f1, "recorded_macro_f1": recorded_macro_f1, "confusion_matrix": confusion}
    result["passed"] = bool(total == sum(map(sum, recorded)) and changed <= max_changed_fraction * total
                            and abs(macro_f1 - recorded_macro_f1) <= max_macro_f1_diff)
    return result


def verify_saved_predictions(rows: list[dict], saved_predictions: Path, *, min_agreement: float = .999,
                             max_mean_abs_diff: float = 1e-3) -> dict:
    """Reproduce the run's saved per-building validation predictions (same model, same inputs)."""
    saved = {}
    with saved_predictions.open(newline="") as stream:
        for row in csv.DictReader(stream):
            saved[(row["scene_id"], row["building_uid"])] = [float(row[c]) for c in PROBABILITY_COLUMNS]
    ours = {(r["scene_id"], r["building_uid"]): [float(r[c]) for c in PROBABILITY_COLUMNS] for r in rows}
    if set(ours) != set(saved):
        return {"passed": False, "reason": "building identity sets differ", "ours": len(ours), "saved": len(saved)}
    agree, diffs, worst = 0, 0., 0.
    for key, values in ours.items():
        other = saved[key]
        agree += max(range(4), key=values.__getitem__) == max(range(4), key=other.__getitem__)
        d = [abs(a - b) for a, b in zip(values, other, strict=True)]
        diffs += sum(d) / 4
        worst = max(worst, max(d))
    n = len(ours)
    result = {"method": "saved per-building predictions", "buildings": n, "argmax_agreement": agree / n,
              "mean_abs_probability_diff": diffs / n, "max_abs_probability_diff": worst}
    result["passed"] = bool(result["argmax_agreement"] >= min_agreement and result["mean_abs_probability_diff"] <= max_mean_abs_diff
                            and math.isfinite(worst))
    return result


def main() -> None:
    from disaster_lens.damage.predictors import load_damage_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Damage-model checkpoint (live or cached head).")
    parser.add_argument("--encoder-dir", type=Path, help="Local DINOv3 weights; cached DINOv3 heads only.")
    parser.add_argument("--saved-predictions", type=Path, help="The run's saved validation predictions (stricter check).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True, help="Verified detector export.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detection-floor", type=float, default=DETECTION_FLOOR)
    parser.add_argument("--skip-reference-check", action="store_true",
                        help="For splits other than the model's validation split, where no recorded results exist.")
    parser.add_argument("--no-change", action="store_true",
                        help="Diagnostic: give the pre-event image as the post-event image too (requires --skip-reference-check).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if args.no_change and not args.skip_reference_check:
        parser.error("--no-change changes the inputs; it cannot reproduce recorded results (add --skip-reference-check).")
    detections, export_identity = load_detections(args.export_dir, args.detection_floor)
    loaded = load_damage_model(args.checkpoint, args.manifest, args.data_dir, device, encoder_dir=args.encoder_dir,
                               saved_predictions=args.saved_predictions)
    metrics = run_end_to_end(
        predictor=loaded.predictor, dataset=loaded.dataset, detections=detections, output_dir=args.output_dir,
        device=device, num_workers=args.num_workers, no_change=args.no_change,
        reference_check=None if args.skip_reference_check else loaded.reference_check,
        provenance={**loaded.provenance, "manifest_sha256": loaded.dataset.manifest_sha256,
                    "detector_predictions_sha256": export_identity["predictions_sha256"],
                    "detector_run_id": export_identity["run_id"], "detector_epoch": export_identity["epoch"],
                    "detection_floor": args.detection_floor, "torch": torch.__version__, "device": str(device)})
    print(json.dumps({k: v for k, v in metrics.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
