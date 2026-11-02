"""Drone section of the report (Mexico Beach, CRASAR-U-DROIDs): explorer images and outlines, alignment and damage tables.

    uv run --group drone python report/scripts/drone.py

Inputs (local, not in Git; produced as described in drone/README.md): datasets/raw/drone/crasar (imagery and
annotations), datasets/raw/drone/crasar-ian-sample (one Ian tile pair), outputs/drone/inputs/{mexico-beach,crasar-ian}
(0.5 m alignment pairs; Ian manifests), outputs/drone/runs/align-mb-03 (RoMa matches), outputs/drone/runs/damage-mb-0{2,3}
(damage predictions: xBD model / fine-tuned on Ian), outputs/modal/test-e2e-* (xBD test predictions).
Writes bundle/data/drone.json and bundle/images/drone-*.jpg (explorer, data cards, matches, close-ups, examples).
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from disaster_lens.drone import crasar as mexico
from disaster_lens.drone.alignment import (
    RES,
    apply_affine,
    consistent,
    fit_affine,
    grid_bounds,
    local_field,
    pipeline_field,
)
from disaster_lens.drone.geo import read_geotiff
from disaster_lens.drone.inputs import PRE_DATE

POST_SAT = mexico.POST
from paths import DATA, IMAGES

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
LABELS = {"no damage": "no-damage", "minor damage": "minor-damage", "major damage": "major-damage", "destroyed": "destroyed"}
DAY, SIZE, VIEW_RES = 1, 880, 0.3  # explorer: drone day 1 vs pre-event 2018-07-28, 880 px at 0.3 m (264 m)
RUNS = ROOT / "outputs" / "drone" / "runs"
INPUTS = ROOT / "outputs" / "drone" / "inputs"


# ------------------------------------------------------------------------------------------------ per-day geometry

class Day:
    """Footprints, true positions on each image, and the pipeline's correction for one drone day (pre 2018-07-28)."""

    def __init__(self, day: int):
        self.day, self.bounds = day, grid_bounds(day)
        self.bld, self.p_ref, self.p_drone = mexico.drone_and_reference(day, PRE_DATE)
        post = mexico.image_path(day, "SATELLITE", POST_SAT)
        self.p_post = mexico.true_positions(self.bld, post, mexico.annotation(day, post, "building_alignment_adjustments", "SATELLITE"))
        z = np.load(INPUTS / "mexico-beach" / f"day{day}_pre{PRE_DATE}.npz")
        self.field = pipeline_field(RUNS / "align-mb-03" / f"day{day}_pre{PRE_DATE}_roma.npz", z["m_ref"].shape)
        self.index = {bid: i for i, bid in enumerate(self.bld.ids)}

    def moved(self, points_m: np.ndarray) -> np.ndarray:
        b = self.bounds
        px = np.c_[(points_m[:, 0] - b[0]) / RES, (b[3] - points_m[:, 1]) / RES]
        ix = np.clip(np.round(px).astype(int), 0, [self.field.shape[1] - 1, self.field.shape[0] - 1])
        d = self.field[ix[:, 1], ix[:, 0]]
        return points_m + np.c_[d[:, 0] * RES, -d[:, 1] * RES]

    def polygon(self, i: int, where: str) -> np.ndarray:
        poly, c = self.bld.polygons[i], self.bld.centroids[i]
        ref = poly + (self.p_ref[i] - c)
        return {"ref": ref, "aligned": self.moved(ref), "truth": poly + (self.p_drone[i] - c),
                "post": poly + (self.p_post[i] - c)}[where]


def crop(path: Path, centre: np.ndarray, size_m: float, res: float) -> tuple[np.ndarray, tuple]:
    h = size_m / 2
    view = (centre[0] - h, centre[1] - h, centre[0] + h, centre[1] + h)
    return read_geotiff(path, view, mexico.CRS, res), view


def to_px(points: np.ndarray, view: tuple, res: float) -> list[list[float]]:
    return [[round(float((x - view[0]) / res), 1), round(float((view[3] - y) / res), 1)] for x, y in points]


def save(img: np.ndarray, name: str, size: int | None = None) -> str:
    from PIL import Image
    im = Image.fromarray(img)
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    im.save(IMAGES / name, quality=88, optimize=True)
    return name


def data_cards(explorer_view: tuple) -> dict:
    from PIL import Image
    pre = np.asarray(Image.open(IMAGES / "drone-explorer-pre.jpg").convert("RGB"))
    post = np.asarray(Image.open(IMAGES / "drone-explorer-post.jpg").convert("RGB"))
    s = (SIZE - 448) // 2
    save(pre[s:s + 448, s:s + 448], "drone-card-mb-pre.jpg"); save(post[s:s + 448, s:s + 448], "drone-card-mb-post.jpg")
    sample = ROOT / "datasets/raw/drone/crasar-ian-sample/ian-1001-Ft-Myers-Beach-Boone-01-01"
    for phase in ("pre", "post"):
        a = np.asarray(Image.open(f"{sample}_{phase}_disaster.png").convert("RGB"))
        save(a[288:736, 288:736], f"drone-card-ian-{phase}.jpg")
    mb = recall_table(RUNS / "damage-mb-03" / "predictions.csv")["A"]["support"]
    ian = {}
    for split in ("train", "validation"):
        rows = list(csv.DictReader(open(INPUTS / "crasar-ian" / f"{split}.csv")))
        ian[split] = {"buildings": len(rows), "tiles": len({r["scene_id"] for r in rows}),
                      "classes": [sum(r["damage_label"] == c for r in rows) for c in CLASSES]}
    return {"mexico_beach": {"classes": mb, "buildings": sum(mb)}, "ian": ian}


def matches_image(explorer_view: tuple) -> dict:
    """The explorer window on the 0.5 m alignment grid: buildings masked, RoMa matches kept / rejected."""
    from PIL import Image, ImageDraw
    z = np.load(INPUTS / "mexico-beach" / f"day{DAY}_pre{PRE_DATE}.npz")
    b = grid_bounds(DAY)
    c0, r0 = int((explorer_view[0] - b[0]) / RES), int((b[3] - explorer_view[3]) / RES)
    n = int((explorer_view[2] - explorer_view[0]) / RES)
    ref, mask = z["ref"][r0:r0 + n, c0:c0 + n].astype(float), z["m_ref"][r0:r0 + n, c0:c0 + n]
    img = np.where(mask[..., None], ref, ref * 0.3).astype(np.uint8)
    m = np.load(RUNS / "align-mb-03" / f"day{DAY}_pre{PRE_DATE}_roma.npz")
    a, bb = m["a"], m["b"]
    A, _ = fit_affine(a, bb)
    r = bb - apply_affine(A, a); keep = np.hypot(*r.T) < 12
    ok = np.zeros(len(a), bool); idx = np.where(keep)[0]; ok[idx[consistent(a[keep], r[keep])]] = True
    inside = (a[:, 0] >= c0) & (a[:, 0] < c0 + n) & (a[:, 1] >= r0) & (a[:, 1] < r0 + n)
    scale = 2  # draw on a 2x canvas for crisp dots
    im = Image.fromarray(img).resize((n * scale, n * scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(0)
    for i in rng.permutation(np.where(inside)[0])[:900]:
        x, y = (a[i, 0] - c0) * scale, (a[i, 1] - r0) * scale
        colour = (60, 207, 110) if ok[i] else (255, 79, 216)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colour, outline=(20, 22, 18))
    im.save(IMAGES / "drone-matches.jpg", quality=88, optimize=True)
    return {"matches": int(inside.sum()), "kept": int((inside & ok).sum()),
            "matchable_share": round(float(mask.mean()), 3)}


def closeups(days: dict[int, Day]) -> list[dict]:
    """Four drone crops (30 m, 0.1 m/px) with the footprint at its GPS, aligned and true position, by fixed rule."""
    d = days[DAY]
    gps = np.hypot(*(d.p_ref - d.p_drone).T)
    aligned = np.array([np.hypot(*(d.moved(d.p_ref[i:i + 1])[0] - d.p_drone[i])) for i in range(len(gps))])
    labelled = np.array([LABELS.get(lbl) is not None for lbl in d.bld.labels])
    fixed = np.where(labelled & (aligned < 1.0))[0]
    picks = list(fixed[np.argsort(-gps[fixed])][:2])
    order = np.argsort(gps); median = [i for i in order[len(order) // 2:] if labelled[i] and i not in picks][0]
    worst = [i for i in np.argsort(-aligned) if labelled[i] and i not in picks + [median]][0]
    out = []
    for k, (i, why) in enumerate(zip(picks + [median, worst], ["large GPS error, fixed", "large GPS error, fixed",
                                                                 "typical GPS error", "largest error left after alignment"])):
        img, view = crop(mexico.image_path(DAY, "UAS"), d.p_drone[i], 30.0, 0.1)
        out.append({"image": save(img, f"drone-close-{k}.jpg"), "size": img.shape[0], "why": why,
                    "gps_m": round(float(gps[i]), 1), "aligned_m": round(float(aligned[i]), 1),
                    "label": LABELS.get(d.bld.labels[i]),
                    "gps": to_px(d.polygon(i, "ref"), view, 0.1), "aligned": to_px(d.polygon(i, "aligned"), view, 0.1),
                    "truth": to_px(d.polygon(i, "truth"), view, 0.1)})
    return out


def examples(days: dict[int, Day]) -> list[dict]:
    """Per class: the most confident fix by fine-tuning, and the most confident remaining mistake (scenario B)."""
    before = predictions(RUNS / "damage-mb-02" / "predictions.csv", "B")
    after = predictions(RUNS / "damage-mb-03" / "predictions.csv", "B")
    out = []
    for c in CLASSES:
        keys = [k for k, v in after.items() if v["label"] == c and k in before]
        fixed = sorted((k for k in keys if after[k]["pred"] == c and before[k]["pred"] != c), key=lambda k: -after[k]["confidence"])
        missed = sorted((k for k in keys if after[k]["pred"] != c), key=lambda k: -after[k]["confidence"])
        for kind, pool in (("fixed by fine-tuning", fixed), ("still wrong", missed)):
            if not pool:
                continue
            day, bid = pool[0]
            d = days[day]; i = d.index[bid]
            crops = {}
            for name, path, where, centre in (
                    ("pre", mexico.image_path(day, "SATELLITE", mexico.PRE[PRE_DATE]), "ref", d.p_ref[i]),
                    ("drone", mexico.image_path(day, "UAS"), "aligned", d.moved(d.p_ref[i:i + 1])[0]),
                    ("post", mexico.image_path(day, "SATELLITE", POST_SAT), "post", d.p_post[i])):
                img, view = crop(path, centre, 40.0, 0.15)
                crops[name] = {"image": save(img, f"drone-ex-{c}-{kind[:5]}-{name}.jpg"),
                               "outline": to_px(d.polygon(i, where), view, 0.15)}
            out.append({"label": c, "kind": kind, "before": before[(day, bid)]["pred"], "after": after[(day, bid)]["pred"],
                        "confidence": after[(day, bid)]["confidence"], "size": img.shape[0], "crops": crops})
    return out


def predictions(path: Path, condition: str, day: int | None = None) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(open(path)):
        if r["condition"] == condition and (day is None or int(r["day"]) == day):
            probs = [float(r[f"p_{c}"]) for c in CLASSES]
            out[(int(r["day"]), r["building_id"])] = {"label": r["label"], "pred": CLASSES[int(np.argmax(probs))],
                                                      "confidence": round(max(probs), 3)}
    return out


def recall_table(path: Path) -> dict:
    out = {}
    for cond in ("A", "B", "D"):
        cm = Counter((v["label"], v["pred"]) for v in predictions(path, cond).values())
        rec, f1 = [], []
        for c in CLASSES:
            n = sum(cm[(c, p)] for p in CLASSES); tp = cm[(c, c)]; pp = sum(cm[(t, c)] for t in CLASSES)
            r_, p_ = (tp / n if n else 0.0), (tp / pp if pp else 0.0)
            rec.append(round(r_, 4)); f1.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
        out[cond] = {"recall": rec, "macro_f1": round(sum(f1) / 4, 4), "support": [sum(cm[(c, p)] for p in CLASSES) for c in CLASSES]}
    return out


def xbd_test_recall(run: str) -> dict:
    """Known-footprint recall per class and macro-F1 on the xBD test split."""
    cm = Counter()
    for r in csv.DictReader(open(ROOT / "outputs" / "modal" / run / "reference_predictions.csv")):
        if r["reference_label"] in CLASSES:
            cm[(r["reference_label"], max(CLASSES, key=lambda c: float(r[f"probability_{c}"])))] += 1
    rec, f1 = [], []
    for c in CLASSES:
        n = sum(cm[(c, p)] for p in CLASSES); tp = cm[(c, c)]; pp = sum(cm[(t, c)] for t in CLASSES)
        r_, p_ = tp / n, (tp / pp if pp else 0.0)
        rec.append(round(r_, 4)); f1.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
    return {"recall": rec, "macro_f1": round(sum(f1) / 4, 4), "buildings": sum(cm.values())}


def alignment_table() -> list[dict]:
    """Per pair: median / p90 building error (m) for GPS placement, LightGlue / RoMa affine, RoMa + local, best smooth."""
    from scipy.spatial import cKDTree
    rows = []
    for day in mexico.DAYS:
        for pre in mexico.PRE:
            bounds = grid_bounds(day)
            bld, p_ref, p_drone = mexico.drone_and_reference(day, pre)
            to_px = lambda P: np.c_[(P[:, 0] - bounds[0]) / RES, (bounds[3] - P[:, 1]) / RES]
            to_m = lambda Q: np.c_[bounds[0] + Q[:, 0] * RES, bounds[3] - Q[:, 1] * RES]
            q = to_px(p_ref)
            stat = lambda e: [round(float(np.median(e)), 2), round(float(np.percentile(e, 90)), 2)]
            row = {"day": day, "pre": pre, "buildings": len(q), "gps": stat(mexico.errors(p_ref, p_drone))}
            for matcher in ("lightglue", "roma"):
                z = np.load(RUNS / "align-mb-03" / f"day{day}_pre{pre}_{matcher}.npz")
                A, _ = fit_affine(z["a"], z["b"])
                row[f"{matcher}_affine"] = stat(mexico.errors(to_m(apply_affine(A, q)), p_drone))
                if matcher == "roma":
                    r = z["b"] - apply_affine(A, z["a"]); keep = np.hypot(*r.T) < 12
                    a2, r2 = z["a"][keep], r[keep]; ok = consistent(a2, r2)
                    row["roma_local"] = stat(mexico.errors(to_m(apply_affine(A, q) + local_field(a2[ok], r2[ok], q, 25 / RES)), p_drone))
            # best smooth correction: leave-one-out Gaussian smoothing of the true residuals of the best affine
            X = np.c_[p_ref, np.ones(len(p_ref))]; coef, *_ = np.linalg.lstsq(X, p_drone, rcond=None); aff = X @ coef
            R, tree, best = p_drone - aff, cKDTree(p_ref), np.zeros_like(p_ref)
            for i, nb in enumerate(tree.query_ball_point(p_ref, 75)):
                nb = [j for j in nb if j != i]
                if nb:
                    w = np.exp(-((p_ref[nb] - p_ref[i]) ** 2).sum(1) / (2 * 25**2)); best[i] = (w[:, None] * R[nb]).sum(0) / w.sum()
            row["best_smooth"] = stat(mexico.errors(aff + best, p_drone))
            rows.append(row)
    return rows


def explorer() -> dict:
    from PIL import Image
    bounds = grid_bounds(DAY)
    bld, p_ref, p_drone = mexico.drone_and_reference(DAY, PRE_DATE)
    preds = predictions(RUNS / "damage-mb-03" / "predictions.csv", "B", DAY)
    labels = [LABELS.get(lbl) for lbl in bld.labels]
    # window: most damage classes with >= 3 buildings (a mixed scene), then most labelled buildings
    span = SIZE * VIEW_RES
    best, centre = (-1, -1), None
    for cx in np.arange(p_ref[:, 0].min() + span / 2, p_ref[:, 0].max() - span / 2, 20):
        for cy in np.arange(p_ref[:, 1].min() + span / 2, p_ref[:, 1].max() - span / 2, 20):
            inside = (np.abs(p_ref[:, 0] - cx) < span / 2 - 10) & (np.abs(p_ref[:, 1] - cy) < span / 2 - 10)
            counts = Counter(labels[i] for i in np.where(inside)[0] if labels[i])
            score = (sum(v >= 3 for v in counts.values()), sum(counts.values()))
            if score > best:
                best, centre = score, (cx, cy)
    x0, y1 = centre[0] - span / 2, centre[1] + span / 2
    view = (x0, y1 - span, x0 + span, y1)
    pre = read_geotiff(mexico.image_path(DAY, "SATELLITE", mexico.PRE[PRE_DATE]), view, mexico.CRS, VIEW_RES)
    drone = read_geotiff(mexico.image_path(DAY, "UAS"), view, mexico.CRS, VIEW_RES)
    for name, img in (("pre", pre), ("post", drone)):
        Image.fromarray(img).save(IMAGES / f"drone-explorer-{name}.jpg", quality=88, optimize=True)
    # our correction, as a dense field on the 0.5 m alignment grid (pre px -> drone px displacement)
    z = np.load(INPUTS / "mexico-beach" / f"day{DAY}_pre{PRE_DATE}.npz")
    field = pipeline_field(RUNS / "align-mb-03" / f"day{DAY}_pre{PRE_DATE}_roma.npz", z["m_ref"].shape)

    def moved(points_m: np.ndarray) -> np.ndarray:
        px = np.c_[(points_m[:, 0] - bounds[0]) / RES, (bounds[3] - points_m[:, 1]) / RES]
        ix = np.clip(np.round(px).astype(int), 0, [field.shape[1] - 1, field.shape[0] - 1])
        d = field[ix[:, 1], ix[:, 0]]
        return points_m + np.c_[d[:, 0] * RES, -d[:, 1] * RES]

    to_view = lambda P: [[round(float((x - x0) / VIEW_RES), 1), round(float((y1 - y) / VIEW_RES), 1)] for x, y in P]
    buildings, errs_gps, errs_aligned = [], [], []
    for i, (poly, c) in enumerate(zip(bld.polygons, bld.centroids)):
        ref_poly = poly + (p_ref[i] - c)
        if np.abs(ref_poly.mean(0) - [centre[0], centre[1]]).max() > span / 2 - 10:
            continue
        aligned = moved(ref_poly)
        errs_gps.append(float(np.hypot(*(p_ref[i] - p_drone[i]))))
        errs_aligned.append(float(np.hypot(*(moved(p_ref[i:i + 1])[0] - p_drone[i]))))
        p = preds.get((DAY, bld.ids[i]))
        label = labels[i]
        buildings.append({"gps": to_view(ref_poly), "aligned": to_view(aligned),
                          "truth": to_view(poly + (p_drone[i] - c)), "label": label,
                          "pred": p["pred"] if p else None, "confidence": p["confidence"] if p else None,
                          "outcome": (None if not p or not label else "correct" if p["pred"] == label else "wrong")})
    rated = [b for b in buildings if b["outcome"]]
    summary = {"buildings": len(buildings), "rated": len(rated),
               "correct": sum(b["outcome"] == "correct" for b in rated),
               "predicted": {c: sum(b["pred"] == c for b in buildings) for c in CLASSES},
               "gps_error_median_m": round(float(np.median(errs_gps)), 2),
               "aligned_error_median_m": round(float(np.median(errs_aligned)), 2)}
    print(f"explorer: {len(buildings)} buildings ({len(rated)} rated); median error GPS {summary['gps_error_median_m']} m "
          f"-> aligned {summary['aligned_error_median_m']} m")
    return {"size": SIZE, "resolution_m": VIEW_RES, "day": DAY, "pre_date": PRE_DATE, "summary": summary, "buildings": buildings,
            "_view": [float(v) for v in view]}


def main() -> None:
    explorer_data = explorer()
    view = tuple(explorer_data.pop("_view"))
    days = {day: Day(day) for day in mexico.DAYS}
    out = {
        "explorer": explorer_data,
        "data_cards": data_cards(view),
        "matches": matches_image(view),
        "closeups": closeups(days),
        "examples": examples(days),
        "alignment": alignment_table(),
        "damage": {"mexico_beach": {"xbd_model": recall_table(RUNS / "damage-mb-02" / "predictions.csv"),
                                    "fine_tuned": recall_table(RUNS / "damage-mb-03" / "predictions.csv")},
                   "xbd_test": {"xbd_model": xbd_test_recall("test-e2e-vit-finetuned"),
                                "fine_tuned": xbd_test_recall("test-e2e-phaseb-ian-01")}},
    }
    (DATA / "drone.json").write_text(json.dumps(out, indent=1))
    print(f"drone.json: {len(out['alignment'])} alignment rows, explorer {out['explorer']['summary']}")


if __name__ == "__main__":
    main()
