"""Scene imagery and outlines for the report bundle (CPU; reads local model outputs and xBD test imagery).

    uv run --group notebook python report/scripts/scenes.py

Writes report/bundle/:
  data/explorer.json        the main explorer scene: detected outlines (class, confidence, outcome) and references
  data/scenes.json          explorer scene list (first scene + per disaster type the two scenes with the most damage)
  data/examples.json        one example crop per disaster type (with found / correct counts)
  data/dataset.json         split sizes and class balance
  data/example-building.json class probabilities of one correctly rated destroyed house
  scenes/<scene>.json       compact outlines per explorer scene (fetched by the website when selected)
  images/                   scene crops, overlays, 448 px example tiles, architecture parts, card cover
  media/scene-explorer.gif  before/after → buildings found → damage, for the README
Scene choices follow fixed rules (no hand-picking), so failure cases are included.
"""

import csv
import json
import re
from collections import Counter, defaultdict

import contourpy
import numpy as np
from paths import (
    CLASSES,
    COLORS,
    DAMAGE_RUN,
    DATA,
    DATASETS,
    IMAGES,
    MEDIA,
    NAMES,
    ROOT,
    SCENES,
    TEST_EXPORT,
    TEST_IMAGES,
    TEST_MANIFEST,
    write_json,
)
from PIL import Image, ImageDraw, ImageFont
from shapely import wkt
from shapely.geometry import Polygon

from disaster_lens.detection.mask_metrics import decode_window, verified_export

SCORE_THRESHOLD = 0.5
MAIN_SCENE, MAIN_CROP = "socal-fire_00000937", (72, 0, 952, 880)   # 880 px square above the no-imagery strip
TYPES = [("fire", "Wildfire"), ("flooding", "Flood"), ("wind", "Hurricane wind"), ("tsunami", "Tsunami"),
         ("earthquake", "Earthquake"), ("volcano", "Volcano")]
EXPLORER_TYPES = ("fire", "flooding", "wind", "tsunami", "earthquake")
EXAMPLE_WINDOW, EXPLORER_WINDOW, EXPLORER_PER_TYPE = 448, 768, 2
EVENTS = {"santa-rosa-wildfire": "Santa Rosa wildfire", "socal-fire": "Southern California wildfire",
          "hurricane-harvey": "Hurricane Harvey", "hurricane-florence": "Hurricane Florence",
          "midwest-flooding": "Midwest flooding", "hurricane-matthew": "Hurricane Matthew",
          "hurricane-michael": "Hurricane Michael", "palu-tsunami": "Palu tsunami",
          "mexico-earthquake": "Mexico earthquake", "guatemala-volcano": "Fuego volcano"}
PROBABILITIES = [f"probability_{c}" for c in CLASSES]
SCENE_ID = re.compile(rb'"scene_id":\s*"([^"]+)"')


class ExportIndex:
    """Random access to the scenes of a verified detector export (one byte offset per scene)."""

    def __init__(self, export_dir) -> None:
        _, self.path, _ = verified_export(export_dir)
        self.offsets: dict[str, int] = {}
        with self.path.open("rb") as stream:
            while line := stream.readline():
                self.offsets[SCENE_ID.search(line, 0, 400).group(1).decode()] = stream.tell() - len(line)

    def record(self, scene_id: str) -> dict:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[scene_id])
            return json.loads(stream.readline())


EXPORT = ExportIndex(TEST_EXPORT)


# --- outlines -------------------------------------------------------------------------------------------------------

def mask_polygon(pixels: np.ndarray, x0: int, y0: int, tolerance: float = .8) -> Polygon | None:
    """Largest outline of a binary window, in native pixel coordinates (pixel edges), lightly simplified."""
    lines = contourpy.contour_generator(z=np.pad(pixels.astype(np.float32), 1)).lines(.5)
    polygons = [Polygon(np.asarray(line) + [x0 - .5, y0 - .5]) for line in lines if len(line) >= 4]
    polygons = [p.buffer(0) for p in polygons if p.area > 0]
    if not polygons:
        return None
    best = max(polygons, key=lambda p: p.area)
    if best.geom_type == "MultiPolygon":
        best = max(best.geoms, key=lambda p: p.area)
    return best.simplify(tolerance, preserve_topology=True)


def to_crop(polygon: Polygon, crop: tuple[int, int, int, int]) -> list[list[float]] | None:
    x1, y1, x2, y2 = crop
    clipped = polygon.intersection(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
    if clipped.is_empty or clipped.area < 4:
        return None
    if clipped.geom_type != "Polygon":
        clipped = max(getattr(clipped, "geoms", [clipped]), key=lambda g: g.area)
        if clipped.geom_type != "Polygon":
            return None
    return [[round(x - x1, 1), round(y - y1, 1)] for x, y in list(clipped.exterior.coords)[:-1]]


def scene_data(scene: str, crop: tuple[int, int, int, int]) -> dict:
    """Detected outlines (predicted class, confidence, outcome vs reference) and reference outlines inside `crop`."""
    damage, truth = {}, {}
    with (DAMAGE_RUN / "detection_predictions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scene_id"] == scene:
                damage[row["building_id"]] = [float(row[c]) for c in PROBABILITIES]
    with (DAMAGE_RUN / "reference_predictions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scene_id"] == scene:
                truth[row["building_uid"]] = row["reference_label"]
    record = EXPORT.record(scene)
    detections = []
    for d in record["detections"]:
        if d["score"] < SCORE_THRESHOLD:
            continue
        window = decode_window(d["mask_window_rle"], (record["height"], record["width"]))
        polygon = mask_polygon(window.pixels, window.bounds[0], window.bounds[1])
        points = to_crop(polygon, crop) if polygon is not None else None
        if points is None:
            continue
        probabilities = damage[d["building_id"]]
        predicted = CLASSES[int(np.argmax(probabilities))]
        matched = d["matched_reference_uid_mask_iou50"]
        reference = truth.get(matched, "unlabeled") if matched else None
        outcome = "extra" if reference in (None, "unlabeled") else "correct" if reference == predicted else "wrong"
        detections.append({"points": points, "label": predicted, "confidence": round(max(probabilities), 3),
                           "score": round(d["score"], 3), "reference": reference, "outcome": outcome, "uid": matched})
    labels = json.loads((TEST_IMAGES / "labels" / f"{scene}_post_disaster.json").read_text())
    found_uids = {d.pop("uid") for d in detections} - {None}
    references = []
    for feature in labels["features"]["xy"]:
        uid = feature["properties"]["uid"]
        points = to_crop(wkt.loads(feature["wkt"]), crop)
        if points is not None:
            references.append({"points": points, "label": truth.get(uid, "unlabeled"), "found": uid in found_uids})
    found = [d for d in detections if d["reference"] not in (None, "unlabeled")]
    return {
        "scene": scene, "event": labels["metadata"]["disaster"], "crop": list(crop), "size": crop[2] - crop[0],
        "score_threshold": SCORE_THRESHOLD, "detections": detections, "references": references,
        "summary": {"references": len(references), "detections": len(detections), "matched_labeled": len(found),
                    "correct": sum(d["label"] == d["reference"] for d in found),
                    "predicted": {k: sum(d["label"] == k for d in detections) for k in CLASSES},
                    "outcomes": {k: sum(d["outcome"] == k for d in detections) for k in ("correct", "wrong", "extra")},
                    "missed": sum(not r["found"] and r["label"] != "unlabeled" for r in references)},
    }


def crop_image(phase: str, scene: str, crop: tuple[int, int, int, int]) -> Image.Image:
    with Image.open(TEST_IMAGES / "images" / f"{scene}_{phase}_disaster.png") as image:
        return image.convert("RGB").crop(crop)


def _hex(color: str, alpha: int) -> tuple[int, int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), alpha


def overlay(image: Image.Image, items: list[dict], color_of, *, scale: float, fill: int = 70, width: int = 2) -> Image.Image:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for item in items:
        color = color_of(item)
        if color is not None:
            draw.polygon([(x * scale, y * scale) for x, y in item["points"]], fill=_hex(color, fill),
                         outline=_hex(color, 255), width=width)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def rounded(points: list[list[float]]) -> list[list[float]]:
    """Outline coordinates to 0.5 px: plenty for display, and keeps the scene files small."""
    return [[round(x * 2) / 2, round(y * 2) / 2] for x, y in points]


def save(image: Image.Image, name: str) -> None:
    image.save(IMAGES / name, quality=86, optimize=True, progressive=True)


# --- selection rules -----------------------------------------------------------------------------------------------

def building_centres() -> tuple[dict, dict, dict]:
    centres: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
    kind, event = {}, {}
    with TEST_MANIFEST.open(newline="") as stream:
        for row in csv.DictReader(stream):
            c = wkt.loads(row["post_wkt_xy"]).centroid
            centres[row["scene_id"]].append((c.x, c.y, row["damage_label"] != "no-damage"))
            kind[row["scene_id"]], event[row["scene_id"]] = row["disaster_type"], row["event"]
    return centres, kind, event


def densest_window(points: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """The size × size window (32 px stride) with the most building centres, damaged buildings counting 4×."""
    best, crop = -1.0, None
    for y in range(0, 1024 - size + 1, 32):
        for x in range(0, 1024 - size + 1, 32):
            inside = (points[:, 0] >= x) & (points[:, 0] < x + size) & (points[:, 1] >= y) & (points[:, 1] < y + size)
            weight = float((inside * np.where(points[:, 2] > 0, 4, 1)).sum())
            if weight > best:
                best, crop = weight, (x, y, x + size, y + size)
    return crop


def ranked(centres: dict, kind: dict, disaster: str, exclude: set[str]) -> list[str]:
    return sorted((s for s in centres if kind[s] == disaster and s not in exclude),
                  key=lambda s: (-sum(d for *_, d in centres[s]), s))


# --- outputs -------------------------------------------------------------------------------------------------------

def explorer_scenes(centres: dict, kind: dict) -> list[dict]:
    chosen = [(MAIN_SCENE, MAIN_CROP, "fire")]
    for disaster in EXPLORER_TYPES:
        for scene in ranked(centres, kind, disaster, {MAIN_SCENE})[:EXPLORER_PER_TYPE]:
            chosen.append((scene, densest_window(np.array(centres[scene]), EXPLORER_WINDOW), disaster))
    summaries = []
    for scene, crop, disaster in chosen:
        data = scene_data(scene, crop)
        for phase in ("pre", "post"):
            save(crop_image(phase, scene, crop), f"scene-{scene}-{phase}.jpg")
        outcomes = ("correct", "wrong", "extra")
        compact = {"size": data["size"],
                   "detections": [{"p": rounded(d["points"]), "l": CLASSES.index(d["label"]), "c": d["confidence"],
                                   "s": d["score"], "o": outcomes.index(d["outcome"]),
                                   "r": CLASSES.index(d["reference"]) if d["reference"] in CLASSES else -1}
                                  for d in data["detections"]],
                   "missed": [rounded(r["points"]) for r in data["references"]
                              if not r["found"] and r["label"] != "unlabeled"]}
        write_json(SCENES / f"{scene}.json", compact, compact=True)
        summary = data["summary"]
        summaries.append({"id": scene, "title": dict(TYPES)[disaster], "event": EVENTS[data["event"]],
                          "size": data["size"], "detections": summary["detections"], "matched": summary["matched_labeled"],
                          "correct": summary["correct"], "missed": summary["missed"], "predicted": summary["predicted"],
                          "outcomes": summary["outcomes"]})
    return summaries


def examples(centres: dict, kind: dict, event: dict) -> dict:
    out = {}
    for disaster, title in TYPES:
        scene = ranked(centres, kind, disaster, set())[0]
        crop = densest_window(np.array(centres[scene]), EXAMPLE_WINDOW)
        data = scene_data(scene, crop)
        pre, post = crop_image("pre", scene, crop), crop_image("post", scene, crop)
        save(pre, f"data-{disaster}-pre.jpg")
        save(post, f"data-{disaster}-post.jpg")
        save(overlay(post, data["detections"], lambda d: COLORS[d["label"]], scale=1.0), f"data-{disaster}-predicted.jpg")
        save(overlay(post, data["references"], lambda r: COLORS[r["label"]] if r["label"] != "unlabeled" else None,
                     scale=1.0), f"data-{disaster}-reference.jpg")
        s = data["summary"]
        out[disaster] = {"title": title, "scene": scene, "event": event[scene], "crop": list(crop),
                         "matched": s["matched_labeled"], "correct": s["correct"], "references": s["references"],
                         "detections": s["detections"], "missed": s["missed"]}
    return out


def main_scene_parts(data: dict) -> None:
    """Architecture-figure parts (440 px), the full-size damage overlay and the 16:9 card cover."""
    pre, post = crop_image("pre", MAIN_SCENE, MAIN_CROP), crop_image("post", MAIN_SCENE, MAIN_CROP)
    small = 440
    scale = small / data["size"]
    pre_s, post_s = pre.resize((small, small), Image.LANCZOS), post.resize((small, small), Image.LANCZOS)
    save(pre_s, "arch-pre.jpg")
    save(post_s, "arch-post.jpg")
    save(overlay(pre_s, data["detections"], lambda _: COLORS["building"], scale=scale, fill=55), "arch-found.jpg")
    save(overlay(post_s, data["detections"], lambda d: COLORS[d["label"]], scale=scale), "arch-damage.jpg")
    damaged = overlay(post, data["detections"], lambda d: COLORS[d["label"]], scale=1.0, width=3)
    save(damaged, "explorer-damage.jpg")
    width, height, gap = 1920, 1080, 16
    panel = (width - gap) // 2
    cover = Image.new("RGB", (width, height), (255, 255, 255))
    for x, image in ((0, pre), (panel + gap, damaged)):
        scaled = image.resize((height, height), Image.LANCZOS)
        left = (height - panel) // 2
        cover.paste(scaled.crop((left, 0, left + panel, height)), (x, 0))
    save(cover, "cover.jpg")
    animation(data, pre, post)


def example_building() -> list[float]:
    """Class probabilities of the main scene's most confident correctly rated destroyed house."""
    truth = {}
    with (DAMAGE_RUN / "reference_predictions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scene_id"] == MAIN_SCENE:
                truth[row["building_uid"]] = row["reference_label"]
    best = None
    with (DAMAGE_RUN / "detection_predictions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scene_id"] != MAIN_SCENE or float(row["score"]) < SCORE_THRESHOLD:
                continue
            p = [float(row[c]) for c in PROBABILITIES]
            correct_destroyed = truth.get(row["matched_reference_uid"]) == "destroyed" and int(np.argmax(p)) == 3
            if correct_destroyed and (best is None or p[3] > best[3]):
                best = p
    return best


def dataset_numbers() -> dict:
    counts = {}
    for split, path in (("train+validation", DATASETS / "processed/train_buildings.csv"), ("test", TEST_MANIFEST)):
        labels, scenes, by_type = Counter(), set(), Counter()
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                labels[row["damage_label"]] += 1
                scenes.add(row["scene_id"])
                by_type[row["disaster_type"]] += 1
        counts[split] = {"buildings": sum(labels.values()), "scenes": len(scenes), "labels": dict(labels),
                         "by_type": dict(by_type)}
    return counts


# --- README animation ----------------------------------------------------------------------------------------------

def _font(size: int) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _caption(frame: Image.Image, step: str, text: str, legend: list[tuple[str, str]] | None = None) -> Image.Image:
    width, height = frame.size
    canvas = Image.new("RGBA", (width, height + 78), (16, 44, 54, 255))
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, height + 12), step, font=_font(11), fill=(169, 211, 182, 255))
    draw.text((16, height + 30), text, font=_font(14), fill=(237, 244, 239, 255))
    x = 16
    for label, color in legend or []:
        draw.rectangle((x, height + 57, x + 10, height + 67), fill=_hex(color, 255))
        draw.text((x + 16, height + 55), label, font=_font(11), fill=(205, 218, 214, 255))
        x += 26 + int(draw.textlength(label, font=_font(11))) + 10
    return canvas.convert("RGB")


def animation(data: dict, pre: Image.Image, post: Image.Image, size: int = 520) -> None:
    """Before/after swipe → buildings found → damage, as a GIF whose palette reserves the exact class colours."""
    pre, post = pre.resize((size, size), Image.LANCZOS), post.resize((size, size), Image.LANCZOS)
    scale = size / data["size"]
    frames, durations = [], []
    frames.append(_caption(pre, "1 / 3  ·  BEFORE / AFTER", "Before the fire")); durations.append(1300)
    for step in range(1, 9):
        x = int(size * step / 8)
        frame = pre.copy()
        frame.paste(post.crop((0, 0, x, size)), (0, 0))
        ImageDraw.Draw(frame).line([(x, 0), (x, size)], fill=(255, 255, 255), width=2)
        frames.append(_caption(frame, "1 / 3  ·  BEFORE / AFTER", "After the fire")); durations.append(90)
    frames.append(_caption(post, "1 / 3  ·  BEFORE / AFTER", "After the fire")); durations.append(1300)
    found = overlay(post, data["detections"], lambda _: COLORS["building"], scale=scale, fill=40)
    frames.append(_caption(found, "2 / 3  ·  BUILDINGS FOUND", f"{len(data['detections'])} buildings detected"))
    durations.append(1800)
    damaged = overlay(post, data["detections"], lambda d: COLORS[d["label"]], scale=scale)
    frames.append(_caption(damaged, "3 / 3  ·  DAMAGE", "Predicted damage per building",
                           [(NAMES[k], COLORS[k]) for k in CLASSES]))
    durations.append(3400)
    reserved = [_hex(c, 255)[:3] for c in COLORS.values()] + [(16, 44, 54), (237, 244, 239), (169, 211, 182),
                                                             (205, 218, 214), (255, 255, 255)]
    mosaic = Image.new("RGB", (size, len(frames) * (frames[0].height // 4)))
    for i, frame in enumerate(frames):
        mosaic.paste(frame.resize((size, frame.height // 4)), (0, i * (frame.height // 4)))
    imagery = mosaic.quantize(colors=256 - len(reserved), method=Image.MEDIANCUT).getpalette()[:3 * (256 - len(reserved))]
    palette = Image.new("P", (1, 1))
    palette.putpalette(imagery + [channel for color in reserved for channel in color])
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    MEDIA.mkdir(parents=True, exist_ok=True)
    quantized[0].save(MEDIA / "scene-explorer.gif", save_all=True, append_images=quantized[1:], duration=durations,
                      loop=0, optimize=True)


def main() -> None:
    for folder in (IMAGES, SCENES, MEDIA, DATA):
        folder.mkdir(parents=True, exist_ok=True)
    main_data = scene_data(MAIN_SCENE, MAIN_CROP)
    write_json(DATA / "explorer.json", main_data)
    main_scene_parts(main_data)
    centres, kind, event = building_centres()
    write_json(DATA / "scenes.json", explorer_scenes(centres, kind))
    write_json(DATA / "examples.json", examples(centres, kind, event))
    write_json(DATA / "dataset.json", dataset_numbers())
    write_json(DATA / "example-building.json", example_building())
    print(f"scenes: {len(list(SCENES.iterdir()))} explorer scenes, {len(list(IMAGES.iterdir()))} images, GIF; "
          f"source {ROOT.name}")


if __name__ == "__main__":
    main()
