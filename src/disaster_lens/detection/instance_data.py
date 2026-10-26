"""PRE-event per-building mask targets; strict geometry, no silent label edits."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from disaster_lens.detection.masks import CompactMasks
from disaster_lens.data.footprints import GEOMETRY_POLICIES, MaskGeometryError, rasterize_footprint_window
from disaster_lens.data.manifest import index_features
from disaster_lens.detection.data import LocalizationDataset


@dataclass
class InstanceSegmentationSample:
    scene_id: str
    image: Tensor
    target: dict[str, Tensor | CompactMasks]
    reference_uids: list[str]
    geometry_audit: list[dict]  # Repair metadata only, not model input.


class InstanceSegmentationDataset(LocalizationDataset):
    """Compact native windows by default; explicit dense mode is a reference.

    Overlapping buildings have separate masks, not one semantic label map.
    Class 1 means building regardless of damage severity; UIDs remain metadata.
    The stored-mask byte guard is a safety limit, NOT a target filter. Compact
    windows stay on CPU until positive ROI sampling; dense mode is guarded and
    intended for small references. Repair audits stay outside model targets.
    """

    def __init__(
        self, index_dir: Path, data_dir: Path, split: str, *, max_mask_bytes: int = 512 * 1024**2,
        mask_representation: str = "compact", geometry_policy: str = "strict",
    ) -> None:
        if type(max_mask_bytes) is not int or max_mask_bytes < 1:
            raise ValueError("max_mask_bytes must be a positive integer.")
        if mask_representation not in ("compact", "dense") or geometry_policy not in GEOMETRY_POLICIES:
            raise ValueError("Choose compact/dense masks and an explicit supported geometry policy.")
        super().__init__(index_dir, data_dir, split)
        self.max_mask_bytes = max_mask_bytes
        self.mask_representation, self.geometry_policy = mask_representation, geometry_policy
        invalid = [(row["scene_id"], building["reference_uid"]) for row in self.scenes
                   for building in row["buildings"] if not building["geometry_valid"]]
        if invalid and geometry_policy == "strict":
            raise ValueError(f"Strict mask targets blocked by {len(invalid)} invalid geometries in {split}; "
                             f"first={invalid[0]}. Audit and choose a policy before training; no targets dropped.")

    def __getitem__(self, index: int) -> InstanceSegmentationSample:
        row = self.scenes[index]
        height, width = row["height"], row["width"]
        required = len(row["buildings"]) * height * width  # uint8, before tensor/cast/worker copies
        if self.mask_representation == "dense" and required > self.max_mask_bytes:
            raise MemoryError(f"{row['scene_id']}: dense uint8 masks need {required} bytes, exceeding "
                              f"the {self.max_mask_bytes}-byte guard. No buildings were dropped.")
        sample = super().__getitem__(index)
        annotation = self._source_path(row["pre_annotation"], row["scene_id"], "labels", ".json", row["source_split"])
        raw = annotation.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["annotation_sha256"]:
            raise ValueError(f"Pre annotation changed since indexing: {row['scene_id']}")
        features = index_features(json.loads(raw), annotation)
        if set(features) != set(sample.reference_uids):
            raise ValueError("Mask annotations and indexed reference UIDs differ.")
        windows = []
        for uid in sample.reference_uids:
            try:
                windows.append(rasterize_footprint_window(features[uid].get("wkt"), width, height,
                                                         geometry_policy=self.geometry_policy))
            except MaskGeometryError as error:
                raise ValueError(f"{row['scene_id']}/{uid}: unusable mask ({error}); no targets dropped.") from error
        compact = CompactMasks((height, width), tuple(windows))
        if compact.stored_bytes > self.max_mask_bytes:
            raise MemoryError("Compact target windows exceed the explicit byte guard; no targets dropped.")
        target: dict[str, Tensor | CompactMasks] = dict(sample.target)
        if self.mask_representation == "compact":
            target["compact_masks"] = compact
        else:
            target["masks"] = compact.dense_reference(max_bytes=self.max_mask_bytes)
        target["area"] = torch.tensor([window.area for window in windows], dtype=torch.float32)
        repairs = [{"scene_id": row["scene_id"], "reference_uid": uid,
                    "annotation_sha256": row["annotation_sha256"], **window.repair}
                   for uid, window in zip(sample.reference_uids, windows, strict=True) if window.repair is not None]
        return InstanceSegmentationSample(sample.scene_id, sample.image, target, sample.reference_uids, repairs)
