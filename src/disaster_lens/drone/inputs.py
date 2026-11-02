"""Build the Mexico Beach inputs: alignment pairs (for the Modal matcher) and damage-model tiles (scenarios A, B, C).

    uv run --group drone python -m disaster_lens.drone.inputs alignment   # outputs/drone/inputs/mexico-beach/*.npz
    uv run --group drone python -m disaster_lens.drone.inputs damage --matches outputs/drone/runs/align-mb-03
                                                                          # outputs/drone/inputs/mexico-beach-damage/

Alignment pair: pre-event satellite + drone on the drone mosaic's 0.5 m UTM grid, with matching masks (valid pixels of
both images minus buildings: reference footprints at their true position + 3 m; drone footprints at their raw position
+ 10 m, the expected misalignment). Damage tiles: 1024 px (512 m) pre-event tiles with the post-event image warped onto
the pre frame: A drone with ground-truth alignment, B drone with the pipeline's alignment, D (scenario C in the report)
the post-event satellite image with ground-truth alignment; plus boxes, building ids and drone-view labels.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from disaster_lens.drone import crasar
from disaster_lens.drone.alignment import (
    RES,
    building_mask,
    grid_bounds,
    nearest_field,
    pipeline_field,
    warp,
)
from disaster_lens.drone.geo import read_geotiff

INPUTS = Path("outputs/drone/inputs")
PRE_DATE = "2018-07-28"  # pre-event reference for the damage tiles
TILE = 1024
LABELS = {"no damage": "no-damage", "minor damage": "minor-damage", "major damage": "major-damage", "destroyed": "destroyed"}


def export(day: int, pre_date: str, out: Path) -> Path:
    bounds = grid_bounds(day)
    bld, p_ref, _ = crasar.drone_and_reference(day, pre_date)
    ref = read_geotiff(crasar.image_path(day, "SATELLITE", crasar.PRE[pre_date]), bounds, crasar.CRS, RES)
    drone = read_geotiff(crasar.image_path(day, "UAS"), bounds, crasar.CRS, RES)
    valid = ~binary_dilation(~((ref.sum(2) > 0) & (drone.sum(2) > 0)), iterations=4)
    m_ref = valid & ~building_mask(bld.polygons, p_ref - bld.centroids, bounds, valid.shape, 3.0)
    m_drone = valid & ~building_mask(bld.polygons, np.zeros_like(bld.centroids), bounds, valid.shape, 10.0)
    path = out / f"day{day}_pre{pre_date}.npz"
    np.savez_compressed(path, ref=ref, drone=drone, m_ref=m_ref, m_drone=m_drone, bounds=np.array(bounds), res=RES)
    print(f"{path.name}: {ref.shape[1]}x{ref.shape[0]} px, matchable {m_ref.mean():.0%} (ref) / {m_drone.mean():.0%} (drone), "
          f"{path.stat().st_size / 1e6:.0f} MB", flush=True)
    return path


def build(day: int, run_dir: Path, out: Path) -> None:
    bounds = grid_bounds(day)
    z = np.load(INPUTS / "mexico-beach" / f"day{day}_pre{PRE_DATE}.npz")
    pre, drone = z["ref"], z["drone"]
    h, w = pre.shape[:2]
    bld, p_ref, p_drone = crasar.drone_and_reference(day, PRE_DATE)
    post_sat_path = crasar.image_path(day, "SATELLITE", crasar.POST)
    p_post = crasar.true_positions(bld, post_sat_path, crasar.annotation(day, post_sat_path, "building_alignment_adjustments", "SATELLITE"))
    post_sat = read_geotiff(post_sat_path, bounds, crasar.CRS, RES)
    to_px = lambda P: np.c_[(P[:, 0] - bounds[0]) / RES, (bounds[3] - P[:, 1]) / RES]
    m2px = lambda D: np.c_[D[:, 0] / RES, -D[:, 1] / RES]
    c_ref = to_px(p_ref)
    post = {
        "A": warp(drone, nearest_field(c_ref, m2px(p_drone - p_ref), (h, w))),
        "B": warp(drone, pipeline_field(run_dir / f"day{day}_pre{PRE_DATE}_roma.npz", (h, w))),
        "D": warp(post_sat, nearest_field(c_ref, m2px(p_post - p_ref), (h, w))),
    }
    # boxes: footprint vertices at their true pre position, in pre-grid px
    boxes, keep = [], []
    for i, (poly, c, label) in enumerate(zip(bld.polygons, bld.centroids, bld.labels)):
        if label not in LABELS:
            continue
        v = to_px(poly + (p_ref[i] - c))
        boxes.append([v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()]); keep.append(i)
    boxes = np.array(boxes)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    tiles = {k: [] for k in ("pre", "A", "B", "D")}
    rec = {"boxes": [], "tile": [], "building_id": [], "label": []}
    for ty in range(0, h, TILE):
        for tx in range(0, w, TILE):
            inside = (centers[:, 0] >= tx) & (centers[:, 0] < tx + TILE) & (centers[:, 1] >= ty) & (centers[:, 1] < ty + TILE)
            if not inside.any():
                continue
            t = len(tiles["pre"])
            for k, img in (("pre", pre), *post.items()):
                tile = np.zeros((TILE, TILE, 3), np.uint8)
                part = img[ty:ty + TILE, tx:tx + TILE]
                tile[:part.shape[0], :part.shape[1]] = part
                tiles[k].append(tile)
            for j in np.where(inside)[0]:
                bx = boxes[j] - [tx, ty, tx, ty]
                bx = np.clip(bx, 0, TILE - 1)
                if bx[2] - bx[0] < 1 or bx[3] - bx[1] < 1:
                    continue
                rec["boxes"].append(bx); rec["tile"].append(t)
                rec["building_id"].append(bld.ids[keep[j]]); rec["label"].append(LABELS[bld.labels[keep[j]]])
    path = out / f"tiles_day{day}.npz"
    np.savez_compressed(path, **{k: np.stack(v) for k, v in tiles.items()}, boxes=np.array(rec["boxes"], np.float32),
                        tile=np.array(rec["tile"]), building_id=np.array(rec["building_id"]), label=np.array(rec["label"]))
    print(f"{path.name}: {len(tiles['pre'])} tiles, {len(rec['label'])} buildings, {path.stat().st_size / 1e6:.0f} MB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("alignment", help="Alignment pairs for every drone day x pre-event image.")
    damage = sub.add_parser("damage", help="Damage-model tiles for scenarios A, B, C.")
    damage.add_argument("--matches", type=Path, required=True, help="Run folder with the RoMa matches.")
    args = parser.parse_args()
    if args.command == "alignment":
        out = INPUTS / "mexico-beach"; out.mkdir(parents=True, exist_ok=True)
        for day in crasar.DAYS:
            for pre in crasar.PRE:
                export(day, pre, out)
    else:
        out = INPUTS / "mexico-beach-damage"; out.mkdir(parents=True, exist_ok=True)
        for day in crasar.DAYS:
            build(day, args.matches, out)


if __name__ == "__main__":
    main()
