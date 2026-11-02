"""Alignment of a drone orthomosaic (post-event) to a satellite pre-event reference on a shared 0.5 m UTM grid.

Matching ignores buildings (`building_mask`) and keeps matches below a displacement prior; a RANSAC affine
(`fit_affine`) removes the global offset; a smooth residual field from locally consistent matches (`consistent`,
`local_field` / `pipeline_field`) removes most of the rest. `nearest_field` turns per-building ground-truth shifts into a
dense field (perfect alignment); `warp` resamples an image with a field. Pixel coordinates are (column, row) on the grid.
"""

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from disaster_lens.drone import crasar

RES = 0.5  # m/px, the alignment grid
PRIOR_M = 20.0  # max plausible displacement; raw georeference errors are below ~8 m
CELL = 4  # px, coarse grid for dense fields


def grid_bounds(day: int) -> tuple[float, float, float, float]:
    """Bounds (UTM) of a Mexico Beach drone mosaic: the alignment grid for that day."""
    with rasterio.open(crasar.image_path(day, "UAS")) as r:
        return transform_bounds(r.crs, crasar.CRS, *r.bounds)


def building_mask(polys, shifts, bounds, shape, margin_m: float) -> np.ndarray:
    """True on footprints (vertices + per-building shift, metres) dilated by `margin_m`, on the RES grid."""
    tf = from_origin(bounds[0], bounds[3], RES, RES)
    geoms = [Polygon(p + s) for p, s in zip(polys, shifts) if len(p) >= 3]
    m = rasterize([(g, 1) for g in geoms if g.is_valid], out_shape=shape, transform=tf, dtype="uint8").astype(bool)
    return binary_dilation(m, iterations=max(1, int(round(margin_m / RES))))


def prepare(img: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE, the matcher input."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def match_sift(ref: np.ndarray, drone: np.ndarray, m_ref: np.ndarray, m_drone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + ratio test inside the masks (the classical baseline)."""
    sift = cv2.SIFT_create(nfeatures=20000)
    k1, d1 = sift.detectAndCompute(ref, m_ref.astype(np.uint8) * 255)
    k2, d2 = sift.detectAndCompute(drone, m_drone.astype(np.uint8) * 255)
    if d1 is None or d2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    a = np.array([k1[m.queryIdx].pt for m in good]); b = np.array([k2[m.trainIdx].pt for m in good])
    return a, b


def fit_affine(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC affine from reference px `a` to drone px `b` (3 px = 1.5 m threshold); (2x3 matrix, inlier mask)."""
    A, inl = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=10000, confidence=0.999)
    return A, inl.ravel().astype(bool)


def apply_affine(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ A[:, :2].T + A[:, 2]


def consistent(a: np.ndarray, r: np.ndarray, radius_px: float = 100, tol_px: float = 3.0) -> np.ndarray:
    """Keep matches whose residual is within tol of the median residual of their neighbours."""
    tree = cKDTree(a); keep = np.zeros(len(a), bool)
    for i, nb in enumerate(tree.query_ball_point(a, radius_px)):
        if len(nb) >= 5:
            keep[i] = np.hypot(*(r[i] - np.median(r[nb], 0))) < tol_px
    return keep


def local_field(a: np.ndarray, r: np.ndarray, q: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-weighted mean residual `r` (at points `a`) evaluated at query points `q`."""
    tree = cKDTree(a); out = np.zeros_like(q)
    for i, nb in enumerate(tree.query_ball_point(q, 3 * sigma_px)):
        if not nb:
            continue
        w = np.exp(-((a[nb] - q[i]) ** 2).sum(1) / (2 * sigma_px**2))
        out[i] = (w[:, None] * r[nb]).sum(0) / w.sum()
    return out


def tps_field(a: np.ndarray, r: np.ndarray, q: np.ndarray, block_px: int = 50, smoothing: float = 50.0) -> np.ndarray:
    """Smoothing thin-plate spline through block-median residuals, evaluated at `q`."""
    keys = np.floor(a / block_px).astype(int)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    centers = np.array([np.median(a[inv == k], 0) for k in range(len(uniq))])
    vals = np.array([np.median(r[inv == k], 0) for k in range(len(uniq))])
    counts = np.bincount(inv)
    centers, vals = centers[counts >= 3], vals[counts >= 3]
    return RBFInterpolator(centers, vals, kernel="thin_plate_spline", smoothing=smoothing)(q)


def nearest_field(centroids_px: np.ndarray, disp_px: np.ndarray, shape) -> np.ndarray:
    """Dense displacement (px) on the pre grid: each pixel takes its nearest building's displacement."""
    h, w = shape
    gy, gx = np.mgrid[0:h:CELL, 0:w:CELL]
    _, idx = cKDTree(centroids_px).query(np.c_[gx.ravel(), gy.ravel()])
    coarse = disp_px[idx].reshape(gy.shape + (2,)).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_NEAREST)


def pipeline_field(matches: Path, shape, sigma_m: float = 25.0) -> np.ndarray:
    """Our correction as a dense map: drone px = affine(pre px) + Gaussian-smoothed local residual."""
    z = np.load(matches); a, b = z["a"], z["b"]
    A, _ = fit_affine(a, b)
    r = b - apply_affine(A, a)
    keep = np.hypot(*r.T) < 12
    a, r = a[keep], r[keep]
    ok = consistent(a, r); a, r = a[ok], r[ok]
    h, w = shape
    gh, gw = (h + CELL - 1) // CELL, (w + CELL - 1) // CELL
    num = np.zeros((gh, gw, 2)); den = np.zeros((gh, gw))
    ci = np.clip((a / CELL).astype(int), 0, [gw - 1, gh - 1])
    np.add.at(num, (ci[:, 1], ci[:, 0]), r); np.add.at(den, (ci[:, 1], ci[:, 0]), 1)
    s = sigma_m / RES / CELL
    num = np.dstack([gaussian_filter(num[..., k], s) for k in range(2)]); den = gaussian_filter(den, s)
    local = np.where(den[..., None] > 1e-6, num / np.maximum(den, 1e-6)[..., None], 0).astype(np.float32)
    local = cv2.resize(local, (w, h), interpolation=cv2.INTER_LINEAR)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    mapped = np.dstack([A[0, 0] * gx + A[0, 1] * gy + A[0, 2], A[1, 0] * gx + A[1, 1] * gy + A[1, 2]])
    return (mapped + local - np.dstack([gx, gy])).astype(np.float32)


def warp(img: np.ndarray, disp_px: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, gx + disp_px[..., 0], gy + disp_px[..., 1], cv2.INTER_LINEAR, borderValue=0)
