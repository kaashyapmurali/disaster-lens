"""Phase B data: CRASAR-U-DROIDs Hurricane Ian (train + validation) as xBD-style training tiles, on Modal CPU.

    modal run --detach --env main --profile kaashyapmurali drone/modal_crasar_prep.py --limit 1   # smoke test
    modal run --detach --env main --profile kaashyapmurali drone/modal_crasar_prep.py             # all mosaics

One container per drone (UAS) mosaic: download the mosaic, its latest pre-event satellite image and annotations from
Hugging Face to the container's ephemeral disk; resample both onto a 0.5 m UTM grid; warp the drone image onto the
satellite frame with CRASAR's alignment adjustments (each pixel moves with its nearest building: perfect alignment);
cut 1024 px tiles (512 m) holding labelled buildings; write PNGs to the data volume
(/data/raw/crasar-ian/images/<scene>_{pre,post}_disaster.png); delete the download. Returns manifest rows (same
columns as the xBD manifests, source_split "crasar"); the local entrypoint writes them to
outputs/drone/inputs/crasar-ian/{train,validation}.csv for upload with `modal volume put`.
"""

import csv
import json
from pathlib import Path

import modal

HF = "https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs/resolve/main"
app = modal.App("disaster-lens-crasar-prep", include_source=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("rasterio==1.4.3", "numpy<2.3", "scipy", "shapely", "opencv-python-headless", "pillow", "requests")
         .env({"PYTHONUNBUFFERED": "1", "GDAL_CACHEMAX": "1024"})
         .add_local_file(Path(__file__).resolve(), "/root/modal_crasar_prep.py"))
data_volume = modal.Volume.from_name("disaster-lens-data", version=1)

RES, TILE, CELL = 0.5, 1024, 4
LABELS = {"no damage": "no-damage", "minor damage": "minor-damage", "major damage": "major-damage", "destroyed": "destroyed"}
COLUMNS = ["source_split", "scene_id", "building_uid", "event", "disaster_type", "match_status", "raw_damage_label",
           "damage_label", "pre_image", "post_image", "pre_annotation", "post_annotation", "pre_wkt_xy", "post_wkt_xy",
           "pre_capture_date", "post_capture_date", "pre_sensor", "post_sensor"]


def fetch(path: str, dest: Path) -> Path:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{HF}/{path}", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    return dest


@app.function(image=image, cpu=4, memory=16384, timeout=3 * 3600,
              volumes={"/data": data_volume})
def prepare(job: dict) -> list[dict]:
    import cv2
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.warp import transform as warp_points
    from scipy.spatial import cKDTree

    work = Path("/tmp/work"); work.mkdir(exist_ok=True)
    split, name, sat = job["split"], job["uas"], job["pre_sat"]
    uas = fetch(f"{split}/imagery/UAS/{name}", work / name)
    sat_path = fetch(f"{split}/imagery/SATELLITE/{sat}", work / sat)
    ann = lambda src, kind, img: json.loads(fetch(f"{split}/annotations/{src}/{kind}/{img}.json", work / f"{src}-{kind}-{img}.json").read_text())
    bda = ann("UAS", "building_damage_assessment", name)
    adj_uas, adj_sat = ann("UAS", "building_alignment_adjustments", name), ann("SATELLITE", "building_alignment_adjustments", sat)

    with rasterio.open(uas) as r:
        ll = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
    lon, lat = (ll[0] + ll[2]) / 2, (ll[1] + ll[3]) / 2
    crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)
    with rasterio.open(uas) as a, rasterio.open(sat_path) as b:
        ba, bb = transform_bounds(a.crs, crs, *a.bounds), transform_bounds(b.crs, crs, *b.bounds)
    bounds = (max(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), min(ba[3], bb[3]))
    if bounds[2] - bounds[0] < 50 or bounds[3] - bounds[1] < 50:
        print(f"{name}: no overlap with {sat}"); return []
    W, H = int((bounds[2] - bounds[0]) / RES), int((bounds[3] - bounds[1]) / RES)
    grid = from_origin(bounds[0], bounds[3], RES, RES)

    def read(path):
        with rasterio.open(path) as base:
            src_res = abs(base.transform.a) * (111_320 if base.crs.is_geographic else 1)
            level = None
            for i, f in enumerate(base.overviews(1)):
                if src_res * f <= RES * 0.75:
                    level = i
        with rasterio.open(path, **({"overview_level": level} if level is not None else {})) as src:
            out = np.zeros((3, H, W), np.uint8)
            bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]  # grayscale (+alpha) images: repeat band 1
            for k, band in enumerate(bands):
                reproject(rasterio.band(src, band), out[k], dst_transform=grid, dst_crs=crs,
                          resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
        return out.transpose(1, 2, 0)

    pre, drone = read(sat_path), read(uas)

    def true_shift(img_path, adj):
        """Per-building mean vertex displacement (metres, grid CRS) from CRASAR's nearest-vector field."""
        adj = np.array(adj, float).reshape(-1, 2, 2)
        if len(adj) == 0:
            return None
        with rasterio.open(img_path) as r:
            def to_m(px):
                xs, ys = zip(*[r.transform * (c, rr) for c, rr in px])
                return np.c_[warp_points(r.crs, crs, list(xs), list(ys))]
            s, e = to_m(adj[:, 0]), to_m(adj[:, 1])
        return s, e - s

    buildings = []
    for b in bda:
        if b.get("label") not in LABELS or not b.get("EPSG:4326"):
            continue
        x, y = warp_points("EPSG:4326", crs, [p["lon"] for p in b["EPSG:4326"]], [p["lat"] for p in b["EPSG:4326"]])
        buildings.append((b["building_id"], LABELS[b["label"]], b["label"], np.c_[x, y]))
    if not buildings:
        return []
    cents = np.array([p.mean(0) for *_, p in buildings])

    def positions(img_path, adj):
        field = true_shift(img_path, adj)
        if field is None:
            return cents.copy()
        starts, disp = field
        tree = cKDTree(starts)
        return np.array([c + disp[tree.query(p)[1]].mean(0) for (*_, p), c in zip(buildings, cents)])

    p_pre, p_uas = positions(sat_path, adj_sat), positions(uas, adj_uas)
    to_px = lambda P: np.c_[(P[:, 0] - bounds[0]) / RES, (bounds[3] - P[:, 1]) / RES]
    c_pre = to_px(p_pre)
    d_px = np.c_[(p_uas - p_pre)[:, 0] / RES, -(p_uas - p_pre)[:, 1] / RES]
    gy, gx = np.mgrid[0:H:CELL, 0:W:CELL]
    _, idx = cKDTree(c_pre).query(np.c_[gx.ravel(), gy.ravel()])
    field = cv2.resize(d_px[idx].reshape(gy.shape + (2,)).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    fy, fx = np.mgrid[0:H, 0:W].astype(np.float32)
    post = cv2.remap(drone, fx + field[..., 0], fy + field[..., 1], cv2.INTER_LINEAR, borderValue=0)
    del drone

    out_dir = Path("/data/raw/crasar-ian/images"); out_dir.mkdir(parents=True, exist_ok=True)
    stem = name.replace(".geo.tif", "").replace(".", "-")
    rows, valid = [], (pre.sum(2) > 0) & (post.sum(2) > 0)
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            polys = []
            for i, (uid, label, raw, poly) in enumerate(buildings):
                v = to_px(poly + (p_pre[i] - cents[i])) - [tx, ty]
                cx, cy = v.mean(0)
                if not (0 <= cx < TILE and 0 <= cy < TILE):
                    continue
                inside = valid[min(H - 1, int(cy + ty)), min(W - 1, int(cx + tx))]
                if inside:
                    polys.append((uid, label, raw, np.clip(v, 0, TILE - 1)))
            if not polys:
                continue
            scene = f"ian-{stem}-{ty // TILE:02d}-{tx // TILE:02d}"
            for phase, img in (("pre", pre), ("post", post)):
                tile = np.zeros((TILE, TILE, 3), np.uint8)
                part = img[ty:ty + TILE, tx:tx + TILE]; tile[:part.shape[0], :part.shape[1]] = part
                Image.fromarray(tile).save(out_dir / f"{scene}_{phase}_disaster.png", optimize=True)
            for uid, label, raw, v in polys:
                ring = ", ".join(f"{x:.2f} {y:.2f}" for x, y in np.vstack([v, v[:1]]))
                wkt = f"POLYGON (({ring}))"
                rows.append({"source_split": "crasar", "scene_id": scene, "building_uid": uid, "event": "hurricane-ian",
                             "disaster_type": "wind", "match_status": "matched", "raw_damage_label": raw,
                             "damage_label": label, "pre_image": f"raw/crasar-ian/images/{scene}_pre_disaster.png",
                             "post_image": f"raw/crasar-ian/images/{scene}_post_disaster.png", "pre_annotation": "",
                             "post_annotation": "", "pre_wkt_xy": wkt, "post_wkt_xy": wkt,
                             "pre_capture_date": job["pre_date"], "post_capture_date": job["uas_date"],
                             "pre_sensor": "MAXAR", "post_sensor": "sUAS", "_split": "validation" if split == "valid" else "train"})
    data_volume.commit()
    for f in work.iterdir():
        f.unlink()
    print(f"{name}: {W}x{H} px grid, {len({r['scene_id'] for r in rows})} tiles, {len(rows)} buildings", flush=True)
    return rows


@app.local_entrypoint()
def main(limit: int = 0, only: str = ""):
    from datetime import datetime
    stats = list(csv.DictReader(open(Path(__file__).resolve().parents[1] / "datasets/raw/drone/crasar/statistics.csv")))
    jobs = []
    for r in stats:
        if r["Event"] != "Hurricane Ian" or r["Source"] != "sUAS":
            continue
        pre = [x for x in stats if x["Orthomosaic"].startswith(r["Orthomosaic"] + "_") and x["Source"] == "Satellite"
               and x["Pre/Post Event"] == "PRE"]
        if not pre:
            continue
        latest = max(pre, key=lambda x: datetime.strptime(x["Date (mm/dd/yyy)"], "%m/%d/%Y"))
        split = {"Train": "train", "Validation": "valid", "Valid": "valid"}[r["Train/Test"]]
        jobs.append({"split": split, "uas": r["Orthomosaic"], "pre_sat": latest["Orthomosaic"],  # satellite file name = full "<uas>_<scene>.tif.geo.tif"
                     "pre_date": latest["Date (mm/dd/yyy)"], "uas_date": r["Date (mm/dd/yyy)"]})
    jobs.sort(key=lambda j: j["uas"])
    if only:
        jobs = [j for j in jobs if j["uas"] == only]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} mosaics: {sum(j['split'] == 'train' for j in jobs)} train, {sum(j['split'] == 'valid' for j in jobs)} validation")
    rows = [row for result in prepare.map(jobs, return_exceptions=True) if isinstance(result, list) for row in result]
    out = Path(__file__).resolve().parents[1] / "outputs" / "drone" / "inputs" / ("crasar-ian" if not only else "crasar-ian-rerun")
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        part = [r for r in rows if r["_split"] == split]
        with open(out / f"{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore"); w.writeheader(); w.writerows(part)
        print(f"{split}: {len(part)} buildings in {len({r['scene_id'] for r in part})} tiles -> {out / (split + '.csv')}")
