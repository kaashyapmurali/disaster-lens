"""Preview sheets for the aerial-alignment demo: drone / crewed / satellite / xBD views of the same ground.

    uv run --group drone python drone/previews.py <site> <out.jpg>

Sites: palu-<n> (OpenAerialMap mosaic n), kilauea-<n> (CRASAR UAS mosaic n), mexico-beach-<n> (CRASAR day n).
Every source is resampled onto one UTM grid. Row 1: whole drone extent; row 2: a 200 m detail box at 0.25 m/px.
Outlines: xBD pre-event footprints (Palu, Kilauea) or CRASAR footprints + their alignment arrows (Mexico Beach).
"""

import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.warp import transform as warp_points

from disaster_lens.drone.geo import read_geotiff, utm_crs

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "raw"
DRONE = ROOT / "drone"
PANEL = 560  # px per overview panel
DETAIL_M, DETAIL_RES = 200.0, 0.25  # detail box size (m) and resolution (m/px)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


# ---------- sources: each returns (array HxWx3 uint8, transform, crs) at a resolution close to `res` ----------


def xbd_scenes(event: str):
    """xBD scenes of an event with their lng/lat bounding boxes (from post labels)."""
    for label in glob.glob(str(ROOT / "*" / "labels" / f"{event}_*_post_disaster.json")):
        d = json.load(open(label))
        nums = [float(x) for f in d["features"]["lng_lat"] for x in re.findall(r"-?\d+\.\d+", f["wkt"])]
        if nums:
            yield Path(label), (min(nums[0::2]), min(nums[1::2]), max(nums[0::2]), max(nums[1::2]))


def xbd_affine(label: Path) -> Affine | None:
    """Pixel -> lng/lat affine fitted from the polygon vertices given in both systems."""
    d = json.load(open(label))
    px, ll = [], []
    for fxy, fll in zip(d["features"]["xy"], d["features"]["lng_lat"]):
        a = [float(v) for v in re.findall(r"-?\d+\.\d+", fxy["wkt"])]
        b = [float(v) for v in re.findall(r"-?\d+\.\d+", fll["wkt"])]
        if len(a) == len(b):
            px += list(zip(a[0::2], a[1::2])); ll += list(zip(b[0::2], b[1::2]))
    if len(px) < 6:
        return None
    A = np.c_[np.array(px), np.ones(len(px))]
    coef, *_ = np.linalg.lstsq(A, np.array(ll), rcond=None)
    return Affine(coef[0, 0], coef[1, 0], coef[2, 0], coef[0, 1], coef[1, 1], coef[2, 1])


def read_xbd(event: str, phase: str, dst_bounds, dst_crs: CRS, res: float):
    """Mosaic of the xBD `phase` (pre/post) tiles of `event` on the target grid; also returns footprints (lng/lat)."""
    width = int(round((dst_bounds[2] - dst_bounds[0]) / res))
    height = int(round((dst_bounds[3] - dst_bounds[1]) / res))
    dst_transform = from_origin(dst_bounds[0], dst_bounds[3], res, res)
    ll_bounds = transform_bounds(dst_crs, "EPSG:4326", *dst_bounds)
    out = np.zeros((3, height, width), dtype=np.uint8)
    footprints, n = [], 0
    for post_label, bb in xbd_scenes(event):
        if not (bb[0] < ll_bounds[2] and bb[2] > ll_bounds[0] and bb[1] < ll_bounds[3] and bb[3] > ll_bounds[1]):
            continue
        label = Path(str(post_label).replace("_post_disaster", f"_{phase}_disaster"))
        image = label.parent.parent / "images" / label.name.replace(".json", ".png")
        aff = xbd_affine(label)
        if aff is None or not image.exists():
            continue
        arr = np.asarray(Image.open(image).convert("RGB")).transpose(2, 0, 1)
        tile = np.zeros_like(out)
        for b in range(3):
            reproject(arr[b], tile[b], src_transform=aff, src_crs="EPSG:4326", dst_transform=dst_transform,
                      dst_crs=dst_crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
        empty = out.sum(0) == 0
        out[:, empty] = tile[:, empty]
        n += 1
        if phase == "pre":
            for f in json.load(open(label))["features"]["lng_lat"]:
                v = [float(x) for x in re.findall(r"-?\d+\.\d+", f["wkt"])]
                footprints.append(list(zip(v[0::2], v[1::2])))
    return out.transpose(1, 2, 0), footprints, n


# ---------- drawing ----------

def to_px(lonlat, dst_bounds, dst_crs, res):
    xs, ys = warp_points("EPSG:4326", dst_crs, [p[0] for p in lonlat], [p[1] for p in lonlat])
    return [((x - dst_bounds[0]) / res, (dst_bounds[3] - y) / res) for x, y in zip(xs, ys)]


def outline(img: Image.Image, polys, dst_bounds, dst_crs, res, color, width=1):
    draw = ImageDraw.Draw(img)
    for poly in polys:
        pts = to_px(poly, dst_bounds, dst_crs, res)
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=color, width=width)


def arrows(img: Image.Image, segs, dst_bounds, dst_crs, res, color=(255, 230, 0)):
    draw = ImageDraw.Draw(img)
    for (a, b) in segs:
        pa, pb = to_px([a, b], dst_bounds, dst_crs, res)
        draw.line([pa, pb], fill=color, width=2)
        ang = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        for s in (-0.5, 0.5):
            draw.line([pb, (pb[0] - 6 * math.cos(ang + s), pb[1] - 6 * math.sin(ang + s))], fill=color, width=2)


def sheet(rows, title: str, note: str, out: Path):
    """rows: list of list of (PIL image, caption). Images in a row share a size."""
    pad, cap, head = 10, 34, 46
    widths = [sum(im.width for im, _ in r) + pad * (len(r) + 1) for r in rows]
    W = max(widths)
    H = head + sum(r[0][0].height + cap + pad for r in rows) + 30
    canvas = Image.new("RGB", (W, H), (248, 246, 240))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 12), title, fill=(26, 27, 26), font=font(20))
    y = head
    for r in rows:
        x = pad
        for im, caption in r:
            canvas.paste(im, (x, y))
            d.text((x, y + im.height + 6), caption, fill=(60, 60, 55), font=font(14))
            x += im.width + pad
        y += r[0][0].height + cap + pad
    d.text((pad, H - 24), note, fill=(98, 99, 87), font=font(12))
    canvas.save(out, quality=88)
    print(f"{out}: {W}x{H}")


def fit(im: np.ndarray, size: int) -> Image.Image:
    img = Image.fromarray(im)
    img.thumbnail((size, size), Image.LANCZOS)
    return img


# ---------- sites ----------

def footprint_bounds(path: Path, crs: CRS):
    with rasterio.open(path) as r:
        return transform_bounds(r.crs, crs, *r.bounds)


def drone_site(drone: Path, event: str, title: str, out: Path, detail_center=None):
    with rasterio.open(drone) as r:
        ll = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
    crs = utm_crs((ll[0] + ll[2]) / 2, (ll[1] + ll[3]) / 2)
    b = footprint_bounds(drone, crs)
    res = max(b[2] - b[0], b[3] - b[1]) / PANEL
    d_img = read_geotiff(drone, b, crs, res)
    pre, fps, npre = read_xbd(event, "pre", b, crs, res)
    post, _, npost = read_xbd(event, "post", b, crs, res)
    # detail box: densest footprint area inside the drone mosaic (or given centre)
    if detail_center is None:
        valid = d_img.sum(2) > 0
        cents = [np.mean(p, axis=0) for p in fps]
        pts = to_px(cents, b, crs, res) if cents else []
        pts = [p for p in pts if 0 <= p[0] < valid.shape[1] and 0 <= p[1] < valid.shape[0] and valid[int(p[1]), int(p[0])]]
        if pts:
            P = np.array(pts); k = np.argmax([(np.hypot(*(P - p).T) < 40).sum() for p in P]); cx, cy = P[k]
        else:
            cy, cx = np.argwhere(valid).mean(0)
        detail_center = (b[0] + cx * res, b[3] - cy * res)
    h = DETAIL_M / 2
    db = (detail_center[0] - h, detail_center[1] - h, detail_center[0] + h, detail_center[1] + h)
    dd = read_geotiff(drone, db, crs, DETAIL_RES)
    dpre, _, _ = read_xbd(event, "pre", db, crs, DETAIL_RES)
    dpost, _, _ = read_xbd(event, "post", db, crs, DETAIL_RES)
    imgs = [Image.fromarray(a) for a in (d_img, pre, post)]
    for im in imgs[:2]:
        outline(im, fps, b, crs, res, (0, 220, 255))
    ImageDraw.Draw(imgs[0]).rectangle([((db[0] - b[0]) / res, (b[3] - db[3]) / res), ((db[2] - b[0]) / res, (b[3] - db[1]) / res)], outline=(255, 60, 60), width=2)
    det = [Image.fromarray(a) for a in (dd, dpre, dpost)]
    for im in det[:2]:
        outline(im, fps, db, crs, DETAIL_RES, (0, 220, 255), 2)
    rows = [[(imgs[0], f"Drone mosaic ({res:.1f} m/px shown)"), (imgs[1], f"xBD pre-event satellite ({npre} tiles)"), (imgs[2], f"xBD post-event satellite ({npost} tiles)")],
            [(det[0], f"Drone, {DETAIL_M:.0f} m box at {DETAIL_RES} m/px"), (det[1], "xBD pre, same box"), (det[2], "xBD post, same box")]]
    sheet(rows, title, "Cyan: xBD pre-event building footprints (from the xBD labels), drawn where the map coordinates put them. Red box: detail area.", out)


def mexico_beach(day: int, out: Path):
    name = {1: "10132018-MexicoBeach.geo.tif", 2: "10142018-MexicoBeach.geo.tif"}[day]
    base = DRONE / "crasar" / "test"
    drone = base / "imagery" / "UAS" / name
    with rasterio.open(drone) as r:
        ll = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
        drone_tf = r.transform; drone_crs = r.crs
    crs = utm_crs((ll[0] + ll[2]) / 2, (ll[1] + ll[3]) / 2)
    b = footprint_bounds(drone, crs)
    res = max(b[2] - b[0], b[3] - b[1]) / PANEL
    stats = {r["Orthomosaic"]: r for r in csv.DictReader(open(DRONE / "crasar" / "statistics.csv"))} if (DRONE / "crasar" / "statistics.csv").exists() else {}
    others = sorted((base / "imagery" / "CREWED").glob(f"{name}_*")) + sorted((base / "imagery" / "SATELLITE").glob(f"{name}_*"))
    polys = [[(p["lon"], p["lat"]) for p in bld["EPSG:4326"]] for bld in json.load(open(base / "annotations" / "UAS" / "building_damage_assessment" / f"{name}.json"))]
    adj = json.load(open(base / "annotations" / "UAS" / "building_alignment_adjustments" / f"{name}.json"))
    segs = []
    for (x0, y0), (x1, y1) in adj:
        (ax, ay), (bx, by) = drone_tf * (x0, y0), drone_tf * (x1, y1)
        lon, lat = warp_points(drone_crs, "EPSG:4326", [ax, bx], [ay, by])
        segs.append(((lon[0], lat[0]), (lon[1], lat[1])))
    # detail centre: densest polygon area
    cents = [np.mean(p, axis=0) for p in polys]
    xs, ys = warp_points("EPSG:4326", crs, [c[0] for c in cents], [c[1] for c in cents])
    P = np.c_[xs, ys]; k = np.argmax([(np.hypot(*(P - p).T) < 60).sum() for p in P])
    h = DETAIL_M / 2; db = (P[k, 0] - h, P[k, 1] - h, P[k, 0] + h, P[k, 1] + h)
    row1, row2 = [], []
    for src in [drone] + others:
        label = "Drone (UAS)" if src == drone else ("NOAA crewed aircraft" if "CREWED" in str(src) else "Satellite")
        meta = stats.get(src.name, {})
        when = f"{meta.get('Date (mm/dd/yyy)', '?')} {meta.get('Pre/Post Event', '')}".strip()
        im = Image.fromarray(read_geotiff(src, b, crs, res)); dm = Image.fromarray(read_geotiff(src, db, crs, DETAIL_RES))
        outline(im, polys, b, crs, res, (0, 220, 255)); outline(dm, polys, db, crs, DETAIL_RES, (0, 220, 255), 2)
        if src == drone:
            arrows(dm, segs, db, crs, DETAIL_RES)
            ImageDraw.Draw(im).rectangle([((db[0] - b[0]) / res, (b[3] - db[3]) / res), ((db[2] - b[0]) / res, (b[3] - db[1]) / res)], outline=(255, 60, 60), width=2)
        row1.append((im.resize((360, int(360 * im.height / im.width))), f"{label} · {when}"))
        row2.append((dm.resize((360, 360)), f"{label}, detail"))
    sheet([row1, row2], f"Mexico Beach (Hurricane Michael), CRASAR-U-DROIDs · drone day {day}",
          "Cyan: CRASAR building footprints (Microsoft). Yellow arrows (detail, drone): CRASAR alignment adjustments for the drone image. Red box: detail area.", out)


if __name__ == "__main__":
    site, out = sys.argv[1], Path(sys.argv[2])
    if site.startswith("palu-"):
        files = sorted((DRONE / "palu-oam").glob("*.tif"))
        f = files[int(site.split("-")[1])]
        drone_site(f, "palu-tsunami", f"Palu (tsunami/liquefaction) · OpenAerialMap drone: {f.stem}", out)
    elif site.startswith("kilauea-"):
        files = sorted(glob.glob(str(DRONE / "crasar" / "*" / "imagery" / "UAS" / "2018-05-18-*")))
        f = Path(files[int(site.split("-")[1])])
        drone_site(f, "lower-puna-volcano", f"Kilauea / Lower Puna · CRASAR drone: {f.name}", out)
    elif site.startswith("mexico-beach-"):
        mexico_beach(int(site.rsplit("-", 1)[1]), out)
