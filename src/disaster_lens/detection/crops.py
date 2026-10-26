"""Random-crop training samples for the compact Mask R-CNN (scale-up training).

A crop of `crop_size` px is cut from the native scene and handed to the model unchanged; the model's own
resize (min_size = max_size = image_size) then upsamples it, e.g. 512 -> 1024 is 2x. Targets are cropped
consistently:

* a building is kept only if >= `min_visible` of its mask pixels remain in the crop (otherwise it is
  dropped, i.e. treated as background -- a deliberate, documented source of label noise at crop edges);
* kept boxes are the original footprint boxes clipped to the crop; masks are clipped windows;
* a crop with no kept building is redrawn up to `resample_empty` times, then accepted as a pure-background crop.

Crops are a deterministic function of (seed, epoch, item index), so an interrupted run replays identically.
Several crops per scene re-read the scene (about 40 ms), which is negligible next to a GPU step.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from disaster_lens.detection.masks import CompactMasks
from disaster_lens.data.footprints import MaskWindow
from disaster_lens.detection.instance_data import InstanceSegmentationDataset, InstanceSegmentationSample


def crop_window(window: MaskWindow, x1: int, y1: int, x2: int, y2: int) -> MaskWindow | None:
    """Intersect a mask window with the crop rectangle and re-express it in crop coordinates."""
    wx1, wy1, wx2, wy2 = window.bounds
    ix1, iy1, ix2, iy2 = max(wx1, x1), max(wy1, y1), min(wx2, x2), min(wy2, y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return None
    pixels = window.pixels[iy1 - wy1:iy2 - wy1, ix1 - wx1:ix2 - wx1]
    return MaskWindow((ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1), np.ascontiguousarray(pixels), window.repair)


def crop_target(target: dict, x0: int, y0: int, size: int, min_visible: float) -> tuple[dict, list[int]]:
    """Crop a compact-mask target; returns (target, indices of kept source buildings)."""
    masks: CompactMasks = target["compact_masks"]
    boxes = target["boxes"]
    kept, windows, new_boxes = [], [], []
    for index, window in enumerate(masks.windows):
        cropped = crop_window(window, x0, y0, x0 + size, y0 + size)
        full = window.area
        if cropped is None or cropped.area == 0 or full == 0 or cropped.area / full < min_visible:
            continue
        x1, y1, x2, y2 = boxes[index].tolist()
        clipped = [max(x1 - x0, 0.), max(y1 - y0, 0.), min(x2 - x0, float(size)), min(y2 - y0, float(size))]
        if clipped[2] - clipped[0] < 1. or clipped[3] - clipped[1] < 1.:
            continue
        kept.append(index)
        windows.append(cropped)
        new_boxes.append(clipped)
    count = len(kept)
    return {
        "boxes": torch.tensor(new_boxes, dtype=boxes.dtype).reshape(count, 4),
        "labels": target["labels"][kept] if count else target["labels"][:0],
        "image_id": target["image_id"],
        "area": torch.tensor([window.area for window in windows], dtype=target["area"].dtype),
        "iscrowd": target["iscrowd"][kept] if count else target["iscrowd"][:0],
        "compact_masks": CompactMasks((size, size), tuple(windows)),
    }, kept


class CropInstanceDataset(Dataset):
    """`crops_per_scene` random crops for every scene of `base`, in a per-epoch deterministic pattern."""

    def __init__(self, base: InstanceSegmentationDataset, crop_size: int, crops_per_scene: int, seed: int, *,
                 min_visible: float = .5, resample_empty: int = 5) -> None:
        if (type(crop_size) is not int or type(crops_per_scene) is not int or type(resample_empty) is not int
                or crop_size < 32 or crops_per_scene < 1 or resample_empty < 0 or not 0 < min_visible <= 1):
            raise ValueError("Need crop_size >= 32, crops_per_scene >= 1, resample_empty >= 0, 0 < min_visible <= 1.")
        if base.mask_representation != "compact":
            raise ValueError("Crop training requires compact masks.")
        if any(min(row["height"], row["width"]) < crop_size for row in base.scenes):
            raise ValueError("Every scene must be at least as large as the crop.")
        self.base, self.crop_size, self.crops_per_scene = base, crop_size, crops_per_scene
        self.seed, self.min_visible, self.resample_empty, self.epoch = seed, min_visible, resample_empty, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base) * self.crops_per_scene

    def metadata(self) -> dict:
        return {"mode": "random_crops", "crop_size": self.crop_size, "crops_per_scene": self.crops_per_scene,
                "min_visible_fraction": self.min_visible, "resample_empty_attempts": self.resample_empty,
                "partial_building_rule": "kept if >= min_visible of mask pixels remain; boxes/masks clipped to the crop",
                "origin": "uniform integer offsets; deterministic in (seed, epoch, item)"}

    def __getitem__(self, item: int) -> InstanceSegmentationSample:
        if not 0 <= item < len(self):
            raise IndexError(item)
        sample = self.base[item // self.crops_per_scene]
        height, width = sample.image.shape[-2:]
        generator = np.random.default_rng([self.seed, self.epoch, item])
        for attempt in range(self.resample_empty + 1):
            x0 = int(generator.integers(0, width - self.crop_size + 1))
            y0 = int(generator.integers(0, height - self.crop_size + 1))
            target, kept = crop_target(sample.target, x0, y0, self.crop_size, self.min_visible)
            if kept or attempt == self.resample_empty:
                break
        image = sample.image[:, y0:y0 + self.crop_size, x0:x0 + self.crop_size].contiguous()
        return InstanceSegmentationSample(sample.scene_id, image, target, [sample.reference_uids[i] for i in kept], [])
