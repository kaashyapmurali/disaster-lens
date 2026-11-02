"""Score drone -> satellite alignment against CRASAR ground truth.

    uv run --group drone python -m disaster_lens.drone.evaluate matches outputs/drone/runs/align-mb-03
    uv run --group drone python -m disaster_lens.drone.evaluate ground-truth

`matches`: per pair (drone day x pre-event image) and matcher, the per-building error (median / 90th percentile, m) of
no correction, one RANSAC affine, affine + Gaussian local correction at 25 / 50 / 100 m, and affine + thin-plate spline.
`ground-truth`: the raw georeference error of each drone mosaic from its alignment annotations, and what remains after
one global translation or affine (the non-uniform part a local correction must handle).
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_points

from disaster_lens.drone import crasar
from disaster_lens.drone.alignment import (
    RES,
    apply_affine,
    consistent,
    fit_affine,
    grid_bounds,
    local_field,
    tps_field,
)

CRASAR_ROOT = Path("datasets/raw/drone/crasar")


def evaluate(run_dir: Path) -> None:
    print(f"{'pair':22s} {'matcher':9s} {'kept':>6s} {'inl':>6s} |" + "".join(f" {c:>14s}" for c in
          ("raw", "affine", "+local 25m", "+local 50m", "+local 100m", "+tps")) + "   (median / p90, m)")
    for f in sorted(run_dir.glob("*.npz")):
        stem, matcher = f.stem.rsplit("_", 1)
        day = int(stem[3]); pre = stem.split("_pre")[1]
        bounds = grid_bounds(day)
        bld, p_ref, p_drone = crasar.drone_and_reference(day, pre)
        to_px = lambda P: np.c_[(P[:, 0] - bounds[0]) / RES, (bounds[3] - P[:, 1]) / RES]
        to_m = lambda Q: np.c_[bounds[0] + Q[:, 0] * RES, bounds[3] - Q[:, 1] * RES]
        z = np.load(f); a, b = z["a"], z["b"]
        q = to_px(p_ref)
        res = {"raw": crasar.errors(p_ref, p_drone)}
        A, inl = fit_affine(a, b)
        res["affine"] = crasar.errors(to_m(apply_affine(A, q)), p_drone)
        r = b - apply_affine(A, a)  # residuals after the global fit
        loose = np.hypot(*r.T) < 12  # 6 m: generous gate for local structure
        a2, r2 = a[loose], r[loose]
        ok = consistent(a2, r2)
        a3, r3 = a2[ok], r2[ok]
        base = apply_affine(A, q)
        for s in (25, 50, 100):
            res[f"+local {s}m"] = crasar.errors(to_m(base + local_field(a3, r3, q, s / RES)), p_drone)
        res["+tps"] = crasar.errors(to_m(base + tps_field(a3, r3, q)), p_drone)
        cells = "".join(f" {np.median(v):6.2f}/{np.percentile(v, 90):5.2f}" for v in res.values())
        print(f"{stem:22s} {matcher:9s} {len(a):6d} {int(inl.sum()):6d} |{cells}", flush=True)


def per_building_shift(adj: np.ndarray, polys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Building centroids (px) and their mean vertex shift (px) from the nearest-vector field."""
    starts, vec = adj[:, 0, :], adj[:, 1, :] - adj[:, 0, :]
    cents, shifts = [], []
    for p in polys:
        d = ((p[:, None, :] - starts[None, :, :]) ** 2).sum(-1)
        shifts.append(vec[d.argmin(1)].mean(0)); cents.append(p.mean(0))
    return np.array(cents), np.array(shifts)


def residual_after_fit(cents: np.ndarray, shifts: np.ndarray, kind: str) -> np.ndarray:
    target = cents + shifts
    if kind == "translation":
        return shifts - shifts.mean(0)
    A = np.c_[cents, np.ones(len(cents))]
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return target - A @ coef


def ground_truth() -> None:
    """Raw georeference error of every CRASAR drone mosaic downloaded, and what one global fit would leave."""
    print(f"{'orthomosaic':44s} {'bldgs':>5s} {'vectors':>7s} {'gsd':>6s} | {'raw median':>10s} {'p90':>6s} {'max':>6s} | "
          f"{'-trans med':>10s} {'p90':>6s} | {'-affine med':>11s} {'p90':>6s} | pixel/geo check")
    for ann in sorted(glob.glob(str(CRASAR_ROOT / "*" / "annotations" / "UAS" / "building_alignment_adjustments" / "*.json"))):
        name = Path(ann).name.replace(".json", "")
        split = Path(ann).parts[-5]
        image = CRASAR_ROOT / split / "imagery" / "UAS" / name
        bda = Path(ann.replace("building_alignment_adjustments", "building_damage_assessment"))
        if not image.exists() or not bda.exists():
            continue
        adj = np.array(json.load(open(ann)), dtype=float)
        buildings = json.load(open(bda))
        polys = [np.array([[p["x"], p["y"]] for p in b["pixels"]], dtype=float) for b in buildings if b.get("pixels")]
        with rasterio.open(image) as r:
            gsd = abs(r.transform.a)
            # consistency: lon/lat of the first polygon, projected with the image transform, vs its stored pixels
            b0 = next(b for b in buildings if b.get("pixels"))
            xs, ys = warp_points("EPSG:4326", r.crs, [q["lon"] for q in b0["EPSG:4326"]], [q["lat"] for q in b0["EPSG:4326"]])
            col_row = np.array([~r.transform * (x, y) for x, y in zip(xs, ys)])
            check = np.abs(col_row - np.array([[q["x"], q["y"]] for q in b0["pixels"]])).max()
        if len(adj) == 0:
            print(f"{name[:44]:44s} {len(polys):5d} {0:7d}  no adjustment vectors"); continue
        cents, shifts = per_building_shift(adj, polys)
        raw = np.hypot(*shifts.T) * gsd
        tr = np.hypot(*residual_after_fit(cents, shifts, "translation").T) * gsd
        af = np.hypot(*residual_after_fit(cents, shifts, "affine").T) * gsd
        q = lambda a, p: np.percentile(a, p)
        print(f"{name[:44]:44s} {len(polys):5d} {len(adj):7d} {gsd:6.3f} | {q(raw,50):9.2f}m {q(raw,90):5.2f}m {raw.max():5.2f}m | "
              f"{q(tr,50):9.2f}m {q(tr,90):5.2f}m | {q(af,50):10.2f}m {q(af,90):5.2f}m | {check:.1f}px")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("matches", help="Score a run folder of matches.")
    m.add_argument("run_dir", type=Path)
    sub.add_parser("ground-truth", help="Raw georeference error per drone mosaic.")
    args = parser.parse_args()
    evaluate(args.run_dir) if args.command == "matches" else ground_truth()


if __name__ == "__main__":
    main()
