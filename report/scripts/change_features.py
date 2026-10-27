"""Per-building pre/post change measures, joined with a damage model's predictions (CPU only).

Three simple image measures per labeled building, computed on the pre-event footprint mask:

* vegetation change  mean excess-green index (2g - r - b on chromaticity) in a ring around the footprint,
                     post minus pre. RGB only (xBD has no near-infrared band), so this is a greenness proxy.
* color change       mean per-pixel RGB distance between post and pre inside the footprint, scaled to [0, 1].
* structure change   1 - normalized cross-correlation of gradient magnitude inside the footprint;
                     0 = same edge pattern, 1 = unrelated (or edges appeared/vanished entirely).

These describe the images, not the model: plotting predicted severity against them shows associations, not
what the model attends to. Pixel differences also include a few pixels of pre/post misregistration.
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.detection.mask_metrics import decode_window, verified_export

FEATURES = ("vegetation_change", "color_change", "structure_change")
RING = 6  # px of surroundings used for vegetation


def excess_green(rgb: np.ndarray) -> np.ndarray:
    total = rgb.sum(axis=-1) + 1e-6
    r, g, b = (rgb[..., i] / total for i in range(3))
    return 2 * g - r - b


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def building_features(pre: np.ndarray, post: np.ndarray, bounds: tuple[int, int, int, int], mask: np.ndarray) -> dict:
    """Measures for one footprint. `pre`/`post` are float RGB in [0, 1]; `mask` is the window's binary mask."""
    height, width = pre.shape[:2]
    x1, y1, x2, y2 = bounds
    mask = mask.astype(bool)
    if not mask.any():
        return dict.fromkeys(FEATURES, float("nan"))
    # Ring: dilate inside a padded window, clipped to the image.
    px1, py1, px2, py2 = max(x1 - RING, 0), max(y1 - RING, 0), min(x2 + RING, width), min(y2 + RING, height)
    padded = np.zeros((py2 - py1, px2 - px1), dtype=bool)
    padded[y1 - py1:y2 - py1, x1 - px1:x2 - px1] = mask
    ring = binary_dilation(padded, iterations=RING) & ~padded
    vegetation = (float(excess_green(post[py1:py2, px1:px2])[ring].mean() - excess_green(pre[py1:py2, px1:px2])[ring].mean())
                  if ring.any() else float("nan"))
    a, b = pre[y1:y2, x1:x2], post[y1:y2, x1:x2]
    color = float(np.linalg.norm(b[mask] - a[mask], axis=-1).mean() / np.sqrt(3))
    structure = float("nan")
    if min(mask.shape) >= 3 and mask.sum() >= 9:          # too thin a footprint has no measurable structure
        ga, gb = gradient_magnitude(a.mean(-1))[mask], gradient_magnitude(b.mean(-1))[mask]
        ga, gb = ga - ga.mean(), gb - gb.mean()
        energy_a, energy_b = (ga ** 2).sum(), (gb ** 2).sum()
        flat_a, flat_b = energy_a < 1e-9, energy_b < 1e-9
        if flat_a and flat_b:
            structure = 0.                                   # both textureless: nothing changed
        elif flat_a or flat_b:
            structure = 1.                                   # edges appeared or vanished entirely
        else:
            structure = float(1 - (ga * gb).sum() / np.sqrt(energy_a * energy_b))
    return {"vegetation_change": vegetation, "color_change": color, "structure_change": structure}


def compute(export_dir: Path, predictions_csv: Path, manifest: Path, data_dir: Path) -> list[dict]:
    """One row per labeled building with a saved prediction: event, labels, predicted severity, measures."""
    _, path, _ = verified_export(export_dir)
    predicted = {}
    with predictions_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            predicted[row["building_uid"]] = row
    images = {}
    with manifest.open(newline="") as stream:
        for row in csv.DictReader(stream):
            images[row["scene_id"]] = (row["pre_image"], row["post_image"])
    rows = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            size = (record["height"], record["width"])
            pre_path, post_path = images[record["scene_id"]]
            pre = np.asarray(Image.open(data_dir / pre_path).convert("RGB"), dtype=np.float32) / 255
            post = np.asarray(Image.open(data_dir / post_path).convert("RGB"), dtype=np.float32) / 255
            for building in record["reference_buildings"]:
                prediction = predicted.get(building["reference_uid"])
                if prediction is None:
                    continue                      # buildings without a damage label are not scored
                window = decode_window(building["mask_window_rle"], size)
                rows.append({"scene_id": record["scene_id"], "event": record["event"],
                             "building_uid": building["reference_uid"], "area": window.area,
                             "true_label": prediction["reference_label"], "predicted_label": prediction["predicted_label"],
                             "predicted_severity": float(prediction["severity_index"]),
                             **building_features(pre, post, window.bounds, window.pixels)})
    return rows


def binned(values: np.ndarray, target: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin centers, mean target, count) over quantile bins of `values`; NaNs dropped."""
    keep = np.isfinite(values) & np.isfinite(target)
    values, target = values[keep], target[keep]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    centers = np.array([np.median(values[index == i]) for i in range(len(edges) - 1)])
    means = np.array([target[index == i].mean() for i in range(len(edges) - 1)])
    counts = np.bincount(index, minlength=len(edges) - 1)
    return centers, means, counts


def severity_of(label: str) -> float:
    return DAMAGE_LABELS.index(label) / (len(DAMAGE_LABELS) - 1)
