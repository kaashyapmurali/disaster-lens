"""Strict PRE-footprint rasterization and local annotation-only mask audit.

No image decoding, dense mask stacks, raw-label edits, post-event inputs, or remote operations.
Strict geometry is default; make-valid-polygonal is an explicit audited opt-in.
Rasterization marks pixels whose centers intersect the clipped footprint;
interior rings remain holes and separate building instances may overlap.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely import box, get_coordinates, intersects_xy, make_valid, union_all, wkt
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException
from shapely.validation import explain_validity

from disaster_lens.data.building_index import SCHEMA_VERSION
from disaster_lens.data.manifest import index_features


class MaskGeometryError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class MaskWindow:
    bounds: tuple[int, int, int, int]  # [x1,y1,x2,y2), native pixel indices
    pixels: np.ndarray  # uint8 local window, not a full-image mask
    repair: dict | None = None  # Only populated when the explicit repair policy changes geometry.

    @property
    def area(self) -> int:
        return int(self.pixels.sum())


GEOMETRY_POLICIES = ("strict", "make-valid-polygonal")


def _geometry_parts(geometry: BaseGeometry) -> Iterator[BaseGeometry]:
    if geometry.geom_type in ("GeometryCollection", "MultiPolygon", "MultiLineString", "MultiPoint"):
        for component in geometry.geoms:
            yield from _geometry_parts(component)
    else:
        yield geometry


def resolve_footprint(text: str | None, policy: str = "strict") -> tuple[BaseGeometry, dict | None]:
    if policy not in GEOMETRY_POLICIES:
        raise ValueError(f"Choose a geometry policy from {GEOMETRY_POLICIES}.")
    try:
        geometry = wkt.loads(text)
    except (GEOSException, TypeError, ValueError) as error:
        raise MaskGeometryError("unparseable_geometry") from error
    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise MaskGeometryError("non_polygon")
    if geometry.is_empty or not np.isfinite(get_coordinates(geometry)).all():
        raise MaskGeometryError("empty_or_nonfinite_geometry")
    repair = None
    if not geometry.is_valid:
        if policy == "strict":
            raise MaskGeometryError("invalid_geometry", explain_validity(geometry))
        fixed = make_valid(geometry, method="linework", keep_collapsed=True)
        components = list(_geometry_parts(fixed))
        polygons = [part for part in components if part.geom_type == "Polygon" and part.area > 0]
        effective = union_all(polygons)
        if effective.is_empty or not effective.is_valid or effective.geom_type not in ("Polygon", "MultiPolygon"):
            raise MaskGeometryError("repair_has_no_valid_polygonal_area")
        repair = {
            "policy": policy, "method": "linework", "keep_collapsed": True,
            "original_validity": explain_validity(geometry),
            "original_wkt_sha256": hashlib.sha256(geometry.wkt.encode()).hexdigest(),
            "make_valid_output_type": fixed.geom_type, "polygon_components": len(polygons),
            "non_polygon_components_excluded_from_area_mask": [
                part.geom_type for part in components if part.geom_type != "Polygon"],
            "effective_wkt_sha256": hashlib.sha256(effective.wkt.encode()).hexdigest(),
        }
        geometry = effective
    if not math.isfinite(geometry.area) or geometry.area <= 0:
        raise MaskGeometryError("zero_or_nonfinite_area_geometry")
    return geometry, repair


def rasterize_footprint_window(
    text: str | None, width: int, height: int, *, geometry_policy: str = "strict",
) -> MaskWindow:
    """Reject unusable masks; only the explicit opt-in policy repairs geometry."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Raster dimensions must be positive integers.")
    geometry, repair = resolve_footprint(text, geometry_policy)
    clipped = geometry.intersection(box(0., 0., float(width), float(height)))
    if clipped.is_empty or clipped.area <= 0:
        raise MaskGeometryError("outside_or_zero_area_after_clipping")
    min_x, min_y, max_x, max_y = clipped.bounds
    x1, y1 = max(0, math.floor(min_x)), max(0, math.floor(min_y))
    x2, y2 = min(width, math.ceil(max_x)), min(height, math.ceil(max_y))
    xs = np.arange(x1, x2, dtype=np.float64) + .5
    ys = np.arange(y1, y2, dtype=np.float64) + .5
    pixels = intersects_xy(clipped, xs[None, :], ys[:, None]).astype(np.uint8)
    if not pixels.any():
        raise MaskGeometryError("zero_rasterized_pixels")
    return MaskWindow((x1, y1, x2, y2), pixels, repair)


def audit_instance_targets(
    data_dir: Path, index_dir: Path, output: Path, *, geometry_policy: str = "strict",
) -> dict:
    """Audit only assigned source-training polygons, one small window at a time."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    raw_index = (index_dir / "scenes.jsonl").read_bytes()
    digest = hashlib.sha256(raw_index).hexdigest()
    report = json.loads((index_dir / "report.json").read_text())
    if report["schema_version"] != SCHEMA_VERSION or report["scenes_sha256"] != digest:
        raise ValueError("Unsupported or changed localization index.")
    if geometry_policy not in GEOMETRY_POLICIES:
        raise ValueError("Unknown geometry policy.")
    counts = {split: Counter() for split in ("train", "validation")}
    problems = []
    repairs = []
    largest = {split: {"dense_uint8_mask_bytes": 0} for split in counts}
    largest_compact = {split: {"stored_window_bytes": 0} for split in counts}
    seen: set[str] = set()
    for line in raw_index.splitlines():
        row = json.loads(line)
        if row["source_split"] != "train" or row["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Use the source-training localization index only.")
        scene = row["scene_id"]
        if scene in seen:
            raise ValueError("Duplicate scene in localization index.")
        seen.add(scene)
        split = row["split"]
        if split == "unassigned":
            continue
        if split not in counts:
            raise ValueError("Unexpected localization partition.")
        expected = Path("raw/train/labels") / f"{scene}_pre_disaster.json"
        path = data_dir / row["pre_annotation"]
        if Path(row["pre_annotation"]) != expected or not path.resolve().is_relative_to(data_dir.resolve()):
            raise ValueError("Expected an assigned PRE source-training annotation inside data_dir.")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["annotation_sha256"]:
            raise ValueError(f"Annotation changed since indexing: {scene}")
        features = index_features(json.loads(raw), path)
        if set(features) != {b["reference_uid"] for b in row["buildings"]}:
            raise ValueError(f"Annotation/index UID mismatch: {scene}")
        width, height = row["width"], row["height"]
        counts[split]["scenes"] += 1
        nbytes = len(features) * width * height
        if nbytes > largest[split]["dense_uint8_mask_bytes"]:
            largest[split] = {"scene_id": scene, "buildings": len(features),
                              "dense_uint8_mask_bytes": nbytes,
                              "float32_target_copy_bytes": 4 * nbytes}
        compact_bytes = 0
        for uid, feature in sorted(features.items()):
            counts[split]["buildings"] += 1
            try:
                mask = rasterize_footprint_window(feature.get("wkt"), width, height, geometry_policy=geometry_policy)
            except MaskGeometryError as error:
                counts[split][f"blocked:{error.reason}"] += 1
                problems.append({"split": split, "scene_id": scene, "reference_uid": uid,
                                 "reason": error.reason, "detail": str(error)})
            else:
                counts[split]["usable_masks"] += 1
                counts[split]["foreground_pixels"] += mask.area
                compact_bytes += mask.pixels.nbytes
                if mask.repair is not None:
                    counts[split]["repaired_buildings"] += 1
                    repairs.append({"split": split, "scene_id": scene, "reference_uid": uid,
                                    "annotation_sha256": row["annotation_sha256"], **mask.repair})
        counts[split]["compact_stored_window_bytes"] += compact_bytes
        if compact_bytes > largest_compact[split]["stored_window_bytes"]:
            largest_compact[split] = {"scene_id": scene, "buildings": len(features),
                                      "stored_window_bytes": compact_bytes}
    result = {
        "schema_version": 1, "index_sha256": digest,
        "counts": {split: dict(sorted(values.items())) for split, values in counts.items()},
        "problems": problems, "largest_dense_target_stacks": largest,
        "largest_compact_targets": largest_compact,
        "policy": "valid effective polygon geometry under geometry_policy, clipped to image; pixel-center intersection including boundary",
        "geometry_policy": geometry_policy,
        "geometry_implementation": {"shapely": shapely.__version__, "geos": shapely.geos_version_string},
        "repairs": bool(repairs), "repair_records": repairs,
        "target_drops": False, "raw_annotation_edits": False,
        "warning": "Window audit does not allocate dense mask stacks or establish Mask R-CNN GPU feasibility.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--index-dir", type=Path, default=Path("datasets/processed/localization/seed42-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/instance-mask-audit-v1.json"))
    parser.add_argument("--geometry-policy", choices=GEOMETRY_POLICIES, default="strict")
    args = parser.parse_args()
    result = audit_instance_targets(args.data_dir, args.index_dir, args.output, geometry_policy=args.geometry_policy)
    print(json.dumps({key: value for key, value in result.items() if key not in ("problems", "repair_records")}, indent=2))
    print(f"Problem annotations: {len(result['problems'])}; raw data unchanged. Audit: {args.output}")


if __name__ == "__main__":
    main()
