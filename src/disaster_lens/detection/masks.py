"""Compact CPU instance targets and native-window prediction materialization.

ROI projection/paste conventions follow torchvision 0.29.1 (BSD-3-Clause;
see LICENSES/torchvision-BSD-3-Clause.txt). No N x full-image mask allocation
is made unless dense_reference() is explicitly called for small CPU tests.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import expand_boxes, expand_masks
from torchvision.ops import roi_align

from disaster_lens.data.footprints import MaskWindow


@dataclass(frozen=True)
class CompactMasks:
    image_size: tuple[int, int]  # height, width
    windows: tuple[MaskWindow, ...]

    def __post_init__(self) -> None:
        height, width = self.image_size
        if type(height) is not int or type(width) is not int or min(height, width) <= 0:
            raise ValueError("Positive integer image dimensions required.")
        for window in self.windows:
            x1, y1, x2, y2 = window.bounds
            if (not 0 <= x1 <= x2 <= width or not 0 <= y1 <= y2 <= height
                    or window.pixels.shape != (y2 - y1, x2 - x1)
                    or window.pixels.dtype != np.uint8
                    or not np.isin(window.pixels, (0, 1)).all()):
                raise ValueError("Invalid compact binary mask window.")

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def stored_bytes(self) -> int:
        return sum(window.pixels.nbytes for window in self.windows)

    def dense_reference(self, max_bytes: int = 32 * 1024**2) -> Tensor:
        """Guarded CPU reference/debug path, never used by compact training."""
        if len(self) * self.image_size[0] * self.image_size[1] > max_bytes:
            raise MemoryError("Dense reference exceeds explicit byte guard.")
        result = torch.zeros((len(self), *self.image_size), dtype=torch.uint8)
        for index, window in enumerate(self.windows):
            x1, y1, x2, y2 = window.bounds
            result[index, y1:y2, x1:x2] = torch.from_numpy(window.pixels)
        return result

    def resized(self, image_size: tuple[int, int]) -> "CompactMasks":
        """Exactly reuse PyTorch's nearest-neighbor source-index mapping on CPU.

        Instances are preserved even when downsampling removes every positive
        pixel, matching dense torchvision resize rather than dropping targets.
        """
        if image_size == self.image_size:
            return self
        out_h, out_w = image_size
        if type(out_h) is not int or type(out_w) is not int or min(out_h, out_w) <= 0:
            raise ValueError("Positive integer resize dimensions required.")
        in_h, in_w = self.image_size
        xs = F.interpolate(torch.arange(in_w, dtype=torch.float32)[None, None, None, :],
                           size=(1, out_w), mode="nearest")[0, 0, 0].long().numpy()
        ys = F.interpolate(torch.arange(in_h, dtype=torch.float32)[None, None, :, None],
                           size=(out_h, 1), mode="nearest")[0, 0, :, 0].long().numpy()
        windows = []
        for window in self.windows:
            x1, y1, x2, y2 = window.bounds
            cols = np.flatnonzero((xs >= x1) & (xs < x2))
            rows = np.flatnonzero((ys >= y1) & (ys < y2))
            if not len(cols) or not len(rows):
                windows.append(MaskWindow((0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8), window.repair))
                continue
            left, right, top, bottom = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
            pixels = window.pixels[(ys[top:bottom] - y1)[:, None], (xs[left:right] - x1)[None, :]]
            windows.append(MaskWindow((left, top, right, bottom), pixels, window.repair))
        return CompactMasks(image_size, tuple(windows))

    def project(self, boxes: Tensor, matched_indices: Tensor, output_size: int) -> Tensor:
        """ROIAlign only matched compact windows, not a full float32 GT stack.

        One zero border INSIDE the native image reproduces dense bilinear ROI
        sampling; image-edge borders retain torchvision's clamping convention.
        CPU->device copies are limited to positive matched windows.
        """
        if (boxes.ndim != 2 or boxes.shape[1] != 4 or matched_indices.shape != (len(boxes),)
                or matched_indices.dtype != torch.int64
                or not boxes.is_floating_point() or not torch.isfinite(boxes).all()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
                or type(output_size) is not int or output_size < 1):
            raise ValueError("Invalid mask ROI projection inputs.")
        indices = matched_indices.detach().cpu().long().tolist()
        if any(index < 0 or index >= len(self) for index in indices):
            raise ValueError("Matched mask index outside instance range.")
        result = boxes.new_zeros((len(boxes), output_size, output_size))
        height, width = self.image_size
        for reference in sorted(set(indices)):
            positions = [index for index, value in enumerate(indices) if value == reference]
            window = self.windows[reference]
            if window.pixels.size == 0:
                continue
            x1, y1, x2, y2 = window.bounds
            left, right, top, bottom = int(x1 > 0), int(x2 < width), int(y1 > 0), int(y2 < height)
            pixels = F.pad(torch.from_numpy(window.pixels), (left, right, top, bottom))
            feature = pixels.to(device=boxes.device, dtype=boxes.dtype)[None, None]
            selection = torch.tensor(positions, dtype=torch.int64, device=boxes.device)
            offset = boxes.new_tensor([x1 - left, y1 - top, x1 - left, y1 - top])
            local_boxes = boxes[selection] - offset
            result[selection] = roi_align(feature, [local_boxes], output_size, spatial_scale=1.,
                                           sampling_ratio=-1, aligned=False)[:, 0]
        return result


@dataclass(frozen=True)
class MaskProbabilityWindow:
    bounds: tuple[int, int, int, int]
    probabilities: Tensor  # CPU [window_height,window_width], not whole image

    def binary(self, threshold: float = .5) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("Use a finite mask threshold strictly between 0 and 1.")
        return (self.probabilities >= threshold).numpy().astype(np.uint8)


def materialize_mask_window(roi_probability: Tensor, native_box: Tensor, image_size: tuple[int, int]) -> MaskProbabilityWindow:
    """Paste a single ROI mask into its clipped native window, on CPU.

    Reproduces torchvision's padding/box expansion/interpolation conventions
    without allocating an N-instance full-image tensor. Boxes must already be
    in the original image grid. This is not registration or damage inference.
    """
    height, width = image_size
    probability = roi_probability.detach().float().cpu()
    box = native_box.detach().float().cpu()
    if probability.ndim == 3 and probability.shape[0] == 1:
        probability = probability[0]
    if (min(height, width) <= 0 or probability.ndim != 2 or probability.shape[0] != probability.shape[1]
            or probability.numel() == 0 or not torch.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()
            or box.shape != (4,) or not torch.isfinite(box).all()
            or (box[:2] < 0).any() or box[2] > width or box[3] > height or (box[2:] <= box[:2]).any()):
        raise ValueError("Invalid native-grid ROI mask/box.")
    padded, scale = expand_masks(probability[None, None], padding=1)
    expanded = expand_boxes(box[None], scale)[0].to(torch.int64)
    bx1, by1, bx2, by2 = expanded.tolist()
    resized = F.interpolate(padded, size=(max(by2 - by1 + 1, 1), max(bx2 - bx1 + 1, 1)),
                            mode="bilinear", align_corners=False)[0, 0]
    x1, y1, x2, y2 = max(bx1, 0), max(by1, 0), min(bx2 + 1, width), min(by2 + 1, height)
    return MaskProbabilityWindow((x1, y1, x2, y2), resized[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1].contiguous().clone())
