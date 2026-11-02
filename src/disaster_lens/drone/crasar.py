"""CRASAR-U-DROIDs, Mexico Beach (Hurricane Michael): images, footprints and true positions per image (UTM 16N).

Each image I (drone, satellite, crewed) has CRASAR alignment vectors in its own pixel coordinates; a footprint vertex
moves by the nearest vector (README "Alignment Adjustments"). Converting vectors to map coordinates gives, per
building b, its true position on image I: P_I(b) = centroid(b) + d_I(b).

Alignment task: a drone (post) image and a satellite pre-event reference, both georeferenced. An estimated mapping
T: reference map coords -> drone map coords places reference positions on the drone image. Error per building:
    e(b) = | T(P_ref(b)) - P_drone(b) |   (metres; T = identity is the raw-georeference baseline)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_points

BASE = Path("datasets/raw/drone/crasar/test")  # relative to the repository root (see datasets/SOURCES.md)
CRS = "EPSG:32616"  # UTM 16N, metres (the drone mosaics' CRS)
DAYS = {1: "10132018-MexicoBeach.geo.tif", 2: "10142018-MexicoBeach.geo.tif"}
PRE = {"2017-03-11": "10400100291A5A00", "2018-07-28": "1050010011549F00"}  # Maxar pre-event scenes
POST = "104001004384D900"  # Maxar post-event scene, 2018-10-12


@dataclass
class Buildings:
    ids: list[str]
    labels: list[str]
    centroids: np.ndarray  # (n, 2) UTM metres, raw (unadjusted) footprint centroid
    polygons: list[np.ndarray]  # raw footprint vertices, UTM metres


def image_path(day: int, source: str, scene: str | None = None) -> Path:
    name = DAYS[day]
    if source == "UAS":
        return BASE / "imagery" / "UAS" / name
    folder = "SATELLITE" if source == "SATELLITE" else "CREWED"
    tail = f"{scene}.tif.geo.tif" if source == "SATELLITE" else "20181011a_RGB.geo.tif"
    return BASE / "imagery" / folder / f"{name}_{tail}"


def annotation(day: int, image: Path, kind: str, source: str) -> Path:
    return BASE / "annotations" / source / kind / f"{image.name}.json"


def buildings(day: int) -> Buildings:
    """Microsoft footprints as labelled on the drone image (the same building set appears on every view)."""
    data = json.load(open(annotation(day, image_path(day, "UAS"), "building_damage_assessment", "UAS")))
    ids, labels, polys = [], [], []
    for b in data:
        lon = [p["lon"] for p in b["EPSG:4326"]]; lat = [p["lat"] for p in b["EPSG:4326"]]
        x, y = warp_points("EPSG:4326", CRS, lon, lat)
        ids.append(b["building_id"]); labels.append(b["label"]); polys.append(np.c_[x, y])
    return Buildings(ids, labels, np.array([p[:-1].mean(0) if np.allclose(p[0], p[-1]) else p.mean(0) for p in polys]), polys)


def vectors_m(image: Path, adjustment_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """CRASAR adjustment vectors for `image`, as (starts, displacements) in UTM metres."""
    adj = np.array(json.load(open(adjustment_json)), dtype=float).reshape(-1, 2, 2)
    with rasterio.open(image) as r:
        def to_utm(px):
            xs, ys = zip(*[r.transform * (c, rr) for c, rr in px])
            return np.c_[warp_points(r.crs, CRS, list(xs), list(ys))]
        a, b = to_utm(adj[:, 0]), to_utm(adj[:, 1])
    return a, b - a


def true_positions(bld: Buildings, image: Path, adjustment_json: Path) -> np.ndarray:
    """P_I(b): footprint centroid + mean displacement of its vertices (nearest-vector field), UTM metres."""
    if not adjustment_json.exists() or len(json.load(open(adjustment_json))) == 0:
        return bld.centroids.copy()
    starts, disp = vectors_m(image, adjustment_json)
    out = []
    for poly, c in zip(bld.polygons, bld.centroids):
        nearest = ((poly[:, None, :] - starts[None]) ** 2).sum(-1).argmin(1)
        out.append(c + disp[nearest].mean(0))
    return np.array(out)


def drone_and_reference(day: int, pre_date: str) -> tuple[Buildings, np.ndarray, np.ndarray]:
    bld = buildings(day)
    drone = image_path(day, "UAS")
    ref = image_path(day, "SATELLITE", PRE[pre_date])
    p_drone = true_positions(bld, drone, annotation(day, drone, "building_alignment_adjustments", "UAS"))
    p_ref = true_positions(bld, ref, annotation(day, ref, "building_alignment_adjustments", "SATELLITE"))
    return bld, p_ref, p_drone


def errors(p_ref_mapped: np.ndarray, p_drone: np.ndarray) -> np.ndarray:
    return np.hypot(*(p_ref_mapped - p_drone).T)


def summary(e: np.ndarray) -> str:
    """Median / 90th percentile / max of per-building errors."""
    return f"median {np.median(e):5.2f} m · p90 {np.percentile(e, 90):5.2f} m · max {e.max():5.2f} m · n {len(e)}"


if __name__ == "__main__":
    for day in DAYS:
        for pre_date in PRE:
            bld, p_ref, p_drone = drone_and_reference(day, pre_date)
            print(f"day {day} drone vs pre {pre_date} | raw (no alignment): {summary(errors(p_ref, p_drone))}")
