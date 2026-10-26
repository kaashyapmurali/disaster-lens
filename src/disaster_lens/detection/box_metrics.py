"""Single-class box localization metrics, with explicit dense-scene limits.

101-point interpolated AP over IoU 0.50:0.05:0.95, with score-ordered greedy
one-to-one matching per image. Not the full COCO evaluator: no crowd/ignore
annotations, area-range AP, or COCO's default maxDets=100. Size diagnostics
are GT-box-area recall slices of the SAME global matches, not separate AP.
"""

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

IOU_THRESHOLDS = tuple(round(.5 + .05 * index, 2) for index in range(10))


def metrics_agree(actual: object, expected: object) -> bool:
    """Exact structure/counts; allow only tiny floating reduction roundoff.

    One changed match/reference is far larger than these tolerances. This does
    not allow batch/population/threshold changes or waive source identity checks.
    """
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(metrics_agree(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(metrics_agree(a, b) for a, b in zip(actual, expected, strict=True))
    if type(actual) is float and type(expected) is float:
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    if type(actual) not in (type(None), bool, int, str):
        return False
    return type(actual) is type(expected) and actual == expected


@dataclass
class SceneMatches:
    scene_id: str
    event: str
    scores: np.ndarray
    order: np.ndarray
    matched_indices: np.ndarray  # [IoU threshold, detection]; -1 means false positive
    reference_areas: np.ndarray


def _boxes(values: np.ndarray) -> np.ndarray:
    boxes = np.asarray(values, dtype=np.float64)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
            or (boxes < 0).any() or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("Boxes must be finite nonnegative positive-area [N,4] xyxy coordinates.")
    return boxes


def match_scene(
    scene_id: str, event: str, boxes: np.ndarray, scores: np.ndarray,
    reference_boxes: np.ndarray, *, max_detections: int,
) -> SceneMatches:
    """Greedily match each score-ranked detection to its best available reference."""
    boxes, references = _boxes(boxes), _boxes(reference_boxes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(boxes),):
        raise ValueError("Detection boxes/scores are misaligned.")
    areas = np.prod(references[:, 2:] - references[:, :2], axis=1)
    def overlaps(index: int) -> np.ndarray:
        box = boxes[index]
        intersection = np.prod(np.maximum(0., np.minimum(box[2:], references[:, 2:])
                                         - np.maximum(box[:2], references[:, :2])), axis=1)
        return intersection / (np.prod(box[2:] - box[:2]) + areas - intersection)
    return match_overlaps(scene_id, event, scores, areas, overlaps, max_detections=max_detections)


def match_overlaps(
    scene_id: str, event: str, scores: np.ndarray, reference_areas: np.ndarray,
    overlaps: Callable[[int], np.ndarray], *, max_detections: int,
) -> SceneMatches:
    """Common score-ordered one-to-one matching for box OR pixel-mask IoU rows."""
    scores = np.asarray(scores, dtype=np.float64)
    areas = np.asarray(reference_areas, dtype=np.float64)
    if (not scene_id or not event or type(max_detections) is not int or max_detections < 1
            or scores.ndim != 1 or areas.ndim != 1 or not np.isfinite(areas).all() or (areas <= 0).any()
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("Invalid identifiers, limits, scores, or reference areas.")
    order = np.argsort(-scores, kind="stable")[:max_detections]
    matches = np.full((len(IOU_THRESHOLDS), len(order)), -1, dtype=np.int64)
    used = np.zeros((len(IOU_THRESHOLDS), len(areas)), dtype=bool)
    for detection, original_index in enumerate(order):
        ious = np.asarray(overlaps(int(original_index)), dtype=np.float64)
        if ious.shape != areas.shape or not np.isfinite(ious).all() or (ious < 0).any() or (ious > 1).any():
            raise ValueError("Invalid IoU row.")
        candidates = np.flatnonzero(ious >= IOU_THRESHOLDS[0])
        candidates = candidates[np.argsort(-ious[candidates], kind="stable")]
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            for reference in candidates:
                if ious[reference] < threshold:
                    break
                if not used[threshold_index, reference]:
                    matches[threshold_index, detection] = reference
                    used[threshold_index, reference] = True
                    break
    return SceneMatches(scene_id, event, scores[order], order, matches, areas)


def _summarize(scenes: list[SceneMatches], operating_threshold: float) -> dict:
    count = sum(len(scene.reference_areas) for scene in scenes)
    scores = np.concatenate([scene.scores for scene in scenes])
    matched = np.concatenate([scene.matched_indices >= 0 for scene in scenes], axis=1)
    # Canonical scene order + stable input detection order defines exact-score ties.
    order = np.argsort(-scores, kind="stable")
    matched = matched[:, order]
    ap, recall = [], []
    for true_positives in matched:
        cumulative = np.cumsum(true_positives)
        recall_curve = cumulative / count if count else np.zeros(len(cumulative))
        precision_curve = cumulative / np.arange(1, len(cumulative) + 1)
        if len(precision_curve):
            precision_curve = np.maximum.accumulate(precision_curve[::-1])[::-1]
        positions = np.searchsorted(recall_curve, np.linspace(0., 1., 101), side="left")
        interpolated = np.zeros(101)
        valid = positions < len(precision_curve)
        interpolated[valid] = precision_curve[positions[valid]]
        ap.append(float(interpolated.mean()) if count else None)
        recall.append(float(cumulative[-1] / count) if count and len(cumulative) else 0. if count else None)
    operating_mask = scores[order] >= operating_threshold
    tp = int(matched[0, operating_mask].sum())
    predictions = int(operating_mask.sum())
    size_recall = {}
    for name, lower, upper in (("small", 0., 32. ** 2), ("medium", 32. ** 2, 96. ** 2),
                               ("large", 96. ** 2, float("inf"))):
        support = sum(int(((scene.reference_areas >= lower) & (scene.reference_areas < upper)).sum())
                      for scene in scenes)
        recovered = 0
        for scene in scenes:
            indices = scene.matched_indices[0]
            indices = indices[indices >= 0]
            recovered += int(((scene.reference_areas[indices] >= lower)
                              & (scene.reference_areas[indices] < upper)).sum())
        size_recall[name] = {"support": support, "matched_iou50": recovered,
                             "recall_iou50": recovered / support if support else None}
    return {
        "scenes": len(scenes), "reference_buildings": count, "detections": len(scores),
        "ap_50_95": float(np.mean(ap)) if count else None,
        "ap50": ap[0], "ap75": ap[5],
        "mean_recall_50_95": float(np.mean(recall)) if count else None,
        "recall_by_iou": {f"{threshold:.2f}": value for threshold, value in zip(IOU_THRESHOLDS, recall, strict=True)},
        "at_score_threshold_iou50": {
            "threshold": operating_threshold, "predictions": predictions,
            "true_positives": tp, "false_positives": predictions - tp,
            "false_negatives": count - tp,
            "precision": tp / predictions if predictions else 0. if count else None,
            "recall": tp / count if count else None,
        },
        "size_recall_at_output_limit": size_recall,
    }


def summarize_localization(
    scenes: list[SceneMatches], *, max_detections: int, score_floor: float,
    operating_threshold: float = .5, metric_kind: str = "box",
) -> dict:
    """Pool buildings/detections, not scene AP averages; events use their own pools."""
    if metric_kind not in ("box", "mask"):
        raise ValueError("Choose box or mask metrics explicitly.")
    if (not scenes or len({scene.scene_id for scene in scenes}) != len(scenes)
            or type(max_detections) is not int or max_detections < 1
            or not 0 <= score_floor <= operating_threshold <= 1):
        raise ValueError("Need unique nonempty scenes, valid limits, and floor <= operating threshold <= 1.")
    if any(len(scene.scores) > max_detections or (scene.scores < score_floor).any() for scene in scenes):
        raise ValueError("Predictions violate the recorded detection limit/score floor.")
    scenes = sorted(scenes, key=lambda scene: scene.scene_id)
    return {
        "definition": {
            "task": ("single-class pre-event building bounding-box localization" if metric_kind == "box"
                     else "single-class pre-event building instance-mask segmentation"),
            "metric_kind": metric_kind,
            "iou_thresholds": list(IOU_THRESHOLDS), "recall_interpolation_points": 101,
            "max_detections_per_image": max_detections, "score_floor": score_floor,
            "matching": "score-descending greedy one-to-one, best available reference IoU",
            "ties": "within-scene input order; pooled equal scores ordered by scene ID then detection order",
            "size_slices": ("native clipped GT box area" if metric_kind == "box" else "native raster GT mask pixel area")
                           + ": small <32^2; medium [32^2,96^2); large >=96^2",
            "size_recall": "slices of global IoU50 matches, all retained scores; not area-range AP",
            "caveat": "Not standard COCO AP100: configurable dense-scene maxDets, no crowd/ignore handling.",
            "empty_support": "AP/recall null; false-positive counts still reported",
        },
        "pooled": _summarize(scenes, operating_threshold),
        "per_event": {event: _summarize([scene for scene in scenes if scene.event == event], operating_threshold)
                      for event in sorted({scene.event for scene in scenes})},
    }
