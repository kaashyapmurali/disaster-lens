"""Detector candidates with precomputed mask IoUs, so they can be re-scored and re-evaluated exactly (CPU).

A candidate is one exported detection at or above a score floor. Its mask-IoU row against the scene's annotated
footprints is computed once; any new set of scores is then matched and summarized with the same greedy matching and
AP code as the detector export (`box_metrics.match_overlaps` / `summarize_localization`).
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from disaster_lens.detection.box_metrics import (
    SceneMatches,
    match_overlaps,
    summarize_localization,
)
from disaster_lens.detection.mask_metrics import (
    decode_window,
    mask_iou_row,
    verified_export,
)

GEOMETRY_FEATURES = ("score", "score_logit", "log_mask_area", "log_box_area", "log_aspect", "mask_fill",
                     "max_iou_higher", "max_inside_higher", "log_neighbours", "log_scene_confident")


@dataclass
class SceneCandidates:
    scene_id: str
    event: str
    height: int
    width: int
    pre_image: str
    building_ids: list[str]
    scores: np.ndarray                 # [N] original detector scores, descending
    boxes: np.ndarray                  # [N, 4] native xyxy
    mask_rles: list[dict]              # window-local RLE per candidate (decode with mask_metrics.decode_window)
    mask_areas: np.ndarray             # [N] pixels
    overlap_index: list[np.ndarray]    # per candidate: indices of references with IoU > 0
    overlap_iou: list[np.ndarray]      # per candidate: those IoUs
    reference_uids: list[str]
    reference_areas: np.ndarray        # [R] pixels

    def __len__(self) -> int:
        return len(self.scores)

    def overlaps(self, index: int) -> np.ndarray:
        row = np.zeros(len(self.reference_uids))
        row[self.overlap_index[index]] = self.overlap_iou[index]
        return row

    def best_iou(self) -> np.ndarray:
        """Best IoU with any annotated footprint, per candidate (independent of the other candidates)."""
        return np.array([iou.max() if len(iou) else 0. for iou in self.overlap_iou])

    def best_outlines(self, threshold: float = .5) -> np.ndarray:
        """True for each footprint's best-overlapping candidate (IoU >= threshold); duplicates are False.

        Independent of the detector scores; ties go to the earlier (higher-scored) candidate.
        """
        best = np.full(len(self.reference_uids), -1)
        best_iou = np.zeros(len(self.reference_uids))
        for candidate, (index, iou) in enumerate(zip(self.overlap_index, self.overlap_iou, strict=True)):
            better = (iou >= threshold) & (iou > best_iou[index])
            best[index[better]], best_iou[index[better]] = candidate, iou[better]
        target = np.zeros(len(self), dtype=bool)
        target[best[best >= 0]] = True
        return target


def _scene(line: str, floor: float) -> SceneCandidates:
    record = json.loads(line)
    size = (record["height"], record["width"])
    references = tuple(decode_window(b["mask_window_rle"], size) for b in record["reference_buildings"])
    areas = np.array([w.area for w in references], dtype=np.float64)
    bounds = np.array([w.bounds for w in references]).reshape(-1, 4)
    keep = [d for d in record["detections"] if d["score"] >= floor]
    if [d["score"] for d in keep] != sorted((d["score"] for d in keep), reverse=True):
        raise ValueError(f"Export detections are not score-sorted in {record['scene_id']}.")
    index, iou, mask_areas = [], [], []
    for detection in keep:
        window = decode_window(detection["mask_window_rle"], size)
        row = mask_iou_row(window, references, areas, bounds)
        nonzero = np.flatnonzero(row)
        index.append(nonzero)
        iou.append(row[nonzero])
        mask_areas.append(window.area)
    return SceneCandidates(
        scene_id=record["scene_id"], event=record["event"], height=record["height"], width=record["width"],
        pre_image=record["pre_image"], building_ids=[d["building_id"] for d in keep],
        scores=np.array([d["score"] for d in keep], dtype=np.float64),
        boxes=np.array([d["box"] for d in keep], dtype=np.float64).reshape(-1, 4),
        mask_rles=[d["mask_window_rle"] for d in keep], mask_areas=np.array(mask_areas, dtype=np.float64),
        overlap_index=index, overlap_iou=iou,
        reference_uids=[b["reference_uid"] for b in record["reference_buildings"]], reference_areas=areas)


def load_candidates(export_dir: Path, floor: float, *, workers: int = 1) -> tuple[list[SceneCandidates], dict]:
    """Candidates with score >= floor for every exported scene (scenes without footprints only add false positives)."""
    metrics, path, digest = verified_export(export_dir)
    if not metrics["definition"]["score_floor"] <= floor < 1:
        raise ValueError("The candidate floor must lie between the export's score floor and 1.")
    with path.open() as stream:
        lines = stream.readlines()
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            scenes = list(pool.map(_scene, lines, [floor] * len(lines), chunksize=8))
    else:
        scenes = [_scene(line, floor) for line in lines]
    identity = {"predictions_sha256": digest, "floor": floor, "export_definition": metrics["definition"]}
    return scenes, identity


def match(scenes: list[SceneCandidates], scores: list[np.ndarray], *, max_detections: int = 2500) -> list[SceneMatches]:
    """Greedy one-to-one matching of each scene's candidates under new scores (same code as the export)."""
    if len(scores) != len(scenes) or any(len(s) != len(c) for s, c in zip(scores, scenes, strict=True)):
        raise ValueError("One score per candidate is required.")
    return [match_overlaps(scene.scene_id, scene.event, np.asarray(s, dtype=np.float64), scene.reference_areas,
                           scene.overlaps, max_detections=max_detections) for scene, s in zip(scenes, scores, strict=True)]


def localization_metrics(scenes: list[SceneCandidates], scores: list[np.ndarray], *, score_floor: float,
                         operating_threshold: float = .5, max_detections: int = 2500) -> dict:
    """Pooled mask AP / recall / precision for re-scored candidates."""
    return summarize_localization(match(scenes, scores, max_detections=max_detections), metric_kind="mask",
                                  max_detections=max_detections, score_floor=score_floor,
                                  operating_threshold=operating_threshold)


def matched_reference_uids(scenes: list[SceneCandidates], scores: list[np.ndarray]) -> dict[str, str]:
    """building_id -> annotated reference uid matched at mask IoU 0.5 under the given scores ('' if unmatched)."""
    out = {}
    for scene, matches in zip(scenes, match(scenes, scores), strict=True):
        for position, original in enumerate(matches.order):
            reference = matches.matched_indices[0, position]
            out[scene.building_ids[original]] = scene.reference_uids[reference] if reference >= 0 else ""
    return out


def geometry_features(scene: SceneCandidates, confident: float = .5) -> np.ndarray:
    """[N, len(GEOMETRY_FEATURES)] detector score, shape and overlap-with-higher-scored-candidate features."""
    n = len(scene)
    boxes = scene.boxes
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    box_areas = np.maximum(widths * heights, 1e-6)
    if n:
        x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
        y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = intersection / (box_areas[:, None] + box_areas[None, :] - intersection)
        inside = intersection / box_areas[:, None]            # share of candidate i's box inside candidate j's box
        higher = np.tril(np.ones((n, n), dtype=bool), k=-1)   # j < i: scores are sorted descending
        max_iou_higher = np.where(higher, iou, 0).max(axis=1)
        max_inside_higher = np.where(higher, inside, 0).max(axis=1)
        neighbours = ((iou > .1) & ~np.eye(n, dtype=bool)).sum(axis=1)
    else:
        max_iou_higher = max_inside_higher = neighbours = np.zeros(0)
    scores = np.clip(scene.scores, 1e-6, 1 - 1e-6)
    columns = [scene.scores, np.log(scores / (1 - scores)), np.log1p(scene.mask_areas), np.log(box_areas),
               np.log(np.maximum(widths, 1e-3) / np.maximum(heights, 1e-3)), scene.mask_areas / box_areas,
               max_iou_higher, max_inside_higher, np.log1p(neighbours),
               np.full(n, math.log1p(int((scene.scores >= confident).sum())))]
    return np.stack(columns, axis=1).astype(np.float32) if n else np.zeros((0, len(GEOMETRY_FEATURES)), np.float32)
