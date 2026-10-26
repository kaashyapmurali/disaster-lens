"""Overlapping-tile inference for the compact Mask R-CNN, returning scene-level predictions.

Each tile is fed to the unchanged model as an ordinary image (the model's own resize upsamples it,
e.g. 512 px -> 1024 px with min_size=max_size=1024) and its boxes come back in tile coordinates.
Boxes are shifted to scene coordinates; compact ROI masks (28x28, relative to their box) are
invariant to a translation, so they are carried over untouched. Duplicates from overlapping tiles
are removed in two steps:

1. A detection clipped by an *interior* tile edge whose visible part lies entirely inside the
   overlap with the neighbouring tile is discarded: that neighbour sees the building unclipped.
   Detections clipped at the scene border, or larger than the overlap, are kept.
2. Class-agnostic box NMS across tiles (same rule as inside the model), then top-k by score.

The wrapper has the same call signature and output format as the model (list of dicts with boxes,
scores, labels, mask_roi_probs, mask_image_size), so evaluation and export code are unchanged.
A building larger than the tile overlap can come back as several clipped fragments; the overlap
must exceed the largest building of interest.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torchvision.ops import nms


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int = 512
    min_overlap: int = 128
    tile_batch_size: int = 3
    edge_margin: float = 2.0
    nms_threshold: float = 0.5
    detections_per_image: int = 2500
    score_threshold: float = 0.001

    def __post_init__(self) -> None:
        for name in ("tile_size", "min_overlap", "tile_batch_size", "detections_per_image"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer.")
        if self.tile_size < 32 or not 0 <= self.min_overlap < self.tile_size:
            raise ValueError("Need tile_size >= 32 and 0 <= min_overlap < tile_size.")
        if self.tile_batch_size < 1 or self.detections_per_image < 1:
            raise ValueError("tile_batch_size and detections_per_image must be positive.")
        if not (math.isfinite(self.edge_margin) and self.edge_margin >= 0
                and 0 < self.nms_threshold <= 1 and 0 <= self.score_threshold < 1):
            raise ValueError("Invalid edge margin, NMS threshold, or score threshold.")
        if self.min_overlap <= 2 * self.edge_margin:
            raise ValueError("min_overlap must exceed twice the edge margin.")

    def metadata(self) -> dict:
        return asdict(self)


def tile_starts(length: int, tile: int, min_overlap: int) -> tuple[list[int], int, int]:
    """Evenly spaced tile origins covering [0, length]; returns (starts, tile used, actual overlap).

    An image no larger than the tile is a single full-image tile (no upsampling beyond the model's own).
    """
    if tile >= length:
        return [0], length, length
    count = math.ceil((length - min_overlap) / (tile - min_overlap))
    starts = [round(i * (length - tile) / (count - 1)) for i in range(count)]
    return starts, tile, tile - max(b - a for a, b in zip(starts, starts[1:], strict=False))


class TiledInstancePredictor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, config: TilingConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, images: list[Tensor]) -> list[dict]:
        if self.training:
            raise RuntimeError("Tiled prediction is inference only.")
        return [self._predict(image) for image in images]

    def _predict(self, image: Tensor) -> dict:
        config = self.config
        height, width = image.shape[-2:]
        xs, tile_w, overlap_x = tile_starts(width, config.tile_size, config.min_overlap)
        ys, tile_h, overlap_y = tile_starts(height, config.tile_size, config.min_overlap)
        origins = [(x, y) for y in ys for x in xs]
        boxes, scores, probabilities = [], [], []
        for first in range(0, len(origins), config.tile_batch_size):
            chunk = origins[first:first + config.tile_batch_size]
            outputs = self.model([image[:, y:y + tile_h, x:x + tile_w] for x, y in chunk])
            for (x, y), output in zip(chunk, outputs, strict=True):
                if "masks" in output or tuple(output["mask_image_size"]) != (tile_h, tile_w):
                    raise ValueError("Expected compact ROI masks on the tile grid.")
                keep = self._not_clipped_by_overlap(
                    output["boxes"], x, y, tile_w, tile_h, width, height, overlap_x, overlap_y)
                shifted = output["boxes"][keep] + output["boxes"].new_tensor([x, y, x, y])
                boxes.append(shifted)
                scores.append(output["scores"][keep])
                probabilities.append(output["mask_roi_probs"][keep])
        boxes, scores, probabilities = torch.cat(boxes), torch.cat(scores), torch.cat(probabilities)
        order = nms(boxes, scores, config.nms_threshold)       # descending score
        order = order[scores[order] >= config.score_threshold][:config.detections_per_image]
        return {"boxes": boxes[order], "scores": scores[order],
                "labels": torch.ones(len(order), dtype=torch.int64, device=boxes.device),
                "mask_roi_probs": probabilities[order], "mask_image_size": (height, width)}

    def _not_clipped_by_overlap(self, boxes: Tensor, x0: int, y0: int, tile_w: int, tile_h: int,
                                width: int, height: int, overlap_x: int, overlap_y: int) -> Tensor:
        margin = self.config.edge_margin
        drop = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(1) if len(boxes) else (boxes.new_empty(0),) * 4
        if x0 > 0:                                  # interior left edge
            drop |= (x1 <= margin) & (x2 <= overlap_x - margin)
        if x0 + tile_w < width:                     # interior right edge
            drop |= (x2 >= tile_w - margin) & (x1 >= tile_w - overlap_x + margin)
        if y0 > 0:
            drop |= (y1 <= margin) & (y2 <= overlap_y - margin)
        if y0 + tile_h < height:
            drop |= (y2 >= tile_h - margin) & (y1 >= tile_h - overlap_y + margin)
        return ~drop
