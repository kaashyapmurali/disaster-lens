"""Pixel-mask IoU matching and compact, uncompressed window-RLE serialization."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from disaster_lens.detection.box_metrics import SceneMatches, match_overlaps
from disaster_lens.data.footprints import MaskWindow


VERIFIED_EXPORTS = ("matched_checkpoint_metrics", "tiled_inference_not_comparable_to_checkpoint_metrics",
                    "test_split_no_recorded_metrics")


def verified_export(export_dir: Path) -> tuple[dict, Path, str]:
    """(metrics, predictions path, predictions SHA-256) of a verified instance export.

    Accepts an untiled export that reproduced its checkpoint metrics, or a tiled export (a different
    predictor, verified for identity/provenance by the export runner). The prediction file must match
    its recorded hash either way.
    """
    metrics = json.loads((export_dir / "metrics.json").read_text())
    path = export_dir / "predictions.jsonl"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if metrics.get("export_verification") not in VERIFIED_EXPORTS or digest != metrics["provenance"]["predictions_sha256"]:
        raise ValueError("Requires a verified export with matching prediction hash.")
    return metrics, path, digest


def encode_window(window: MaskWindow) -> dict:
    """Row-major alternating zero/one runs; coordinates locate the native crop.

    This is window-local C-order RLE, NOT COCO's full-image column-major format.
    """
    flat = window.pixels.reshape(-1)
    if window.pixels.dtype != np.uint8 or not np.isin(flat, (0, 1)).all():
        raise ValueError("Binary uint8 window required.")
    if len(flat):
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        runs = np.diff(np.r_[0, changes, len(flat)]).tolist()
        if flat[0]:
            runs.insert(0, 0)
    else:
        runs = [0]
    return {"bounds": list(window.bounds), "size": list(window.pixels.shape),
            "order": "C", "counts": runs}


def decode_window(record: dict, image_size: tuple[int, int], max_pixels: int = 16 * 1024**2) -> MaskWindow:
    if record.get("order") != "C" or len(record.get("bounds", ())) != 4 or len(record.get("size", ())) != 2:
        raise ValueError("Expected native-window C-order RLE.")
    x1, y1, x2, y2 = record["bounds"]
    height, width = image_size
    counts = record.get("counts")
    if not isinstance(counts, list):
        raise ValueError("RLE counts must be a list.")
    if (any(type(value) is not int for value in (*record["bounds"], *record["size"]))
            or not 0 <= x1 <= x2 <= width or not 0 <= y1 <= y2 <= height
            or record["size"] != [y2 - y1, x2 - x1]
            or (x2 - x1) * (y2 - y1) > max_pixels
            or not counts or len(counts) > (x2 - x1) * (y2 - y1) + 1
            or any(type(count) is not int or count < 0 for count in counts)
            or any(count == 0 for count in counts[1:])
            or sum(counts) != (x2 - x1) * (y2 - y1)):
        raise ValueError("Invalid or oversized mask RLE.")
    pixels = np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts)
    return MaskWindow((x1, y1, x2, y2), pixels.reshape(record["size"]))


def mask_iou_row(
    prediction: MaskWindow, references: tuple[MaskWindow, ...], areas: np.ndarray,
    reference_bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Compute only overlapping native windows, never full-image mask stacks."""
    overlaps = np.zeros(len(references), dtype=np.float64)
    prediction_area = prediction.area
    if prediction_area == 0 or not references:
        return overlaps
    bounds = np.asarray([window.bounds for window in references]) if reference_bounds is None else reference_bounds
    px1, py1, px2, py2 = prediction.bounds
    candidates = np.flatnonzero((bounds[:, 0] < px2) & (bounds[:, 2] > px1)
                               & (bounds[:, 1] < py2) & (bounds[:, 3] > py1))
    for index in candidates:
        reference = references[index]
        rx1, ry1, rx2, ry2 = reference.bounds
        x1, y1, x2, y2 = max(px1, rx1), max(py1, ry1), min(px2, rx2), min(py2, ry2)
        intersection = int(np.logical_and(prediction.pixels[y1-py1:y2-py1, x1-px1:x2-px1],
                                         reference.pixels[y1-ry1:y2-ry1, x1-rx1:x2-rx1]).sum())
        overlaps[index] = intersection / (prediction_area + areas[index] - intersection)
    return overlaps


def match_mask_scene(
    scene_id: str, event: str, predictions: list[MaskWindow] | Callable[[int], MaskWindow], scores: np.ndarray,
    references: tuple[MaskWindow, ...], *, max_detections: int,
) -> SceneMatches:
    if not callable(predictions) and len(predictions) != len(scores):
        raise ValueError("Mask predictions and scores are misaligned.")
    areas = np.asarray([window.area for window in references], dtype=np.float64)
    bounds = np.asarray([window.bounds for window in references]).reshape(-1, 4)
    def overlaps(index: int) -> np.ndarray:
        window = predictions(index) if callable(predictions) else predictions[index]
        return mask_iou_row(window, references, areas, bounds)
    return match_overlaps(scene_id, event, scores, areas, overlaps,
                          max_detections=max_detections)
