"""Index pre-event imagery (xBD train or test source) for single-class building detection.

Reads pre-event annotations and image sizes only (no post imagery or damage labels). The train source keeps the
existing scene assignments (unassigned scenes are audited, not assigned); the test source is one split.
"""

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image
from shapely import wkt
from shapely.errors import GEOSException
from shapely.validation import explain_validity

from disaster_lens.data.manifest import index_features

SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_assignments(path: Path) -> dict[str, str]:
    """scene_id -> train | validation, for scenes of the xBD training source."""
    assignments: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scene, split = row["scene_id"], row["split"]
            if not scene or scene in assignments or split not in ("train", "validation"):
                raise ValueError(f"Invalid or duplicate scene assignment: {row}")
            assignments[scene] = split
    if not assignments:
        raise ValueError("Empty scene assignments.")
    return assignments


def building_bounds(text: str | None, width: int, height: int) -> dict:
    """Audit geometry without repairing it; positive finite polygon bounds suffice.

    Invalid topology is flagged but does not disqualify a bounding-box target.
    Unparseable, non-polygon, nonfinite, degenerate, or wholly off-grid geometry
    has no usable box. Such annotations remain in the index, and the loader
    refuses their scenes rather than silently teaching false background.
    """
    result: dict = {
        "box": None, "raw_box": None, "geometry_valid": False,
        "geometry_issue": None, "box_issue": None, "clipped": False,
    }
    try:
        shape = wkt.loads(text)
    except (GEOSException, TypeError, ValueError):
        result["box_issue"] = "unparseable_geometry"
        return result
    result["geometry_valid"] = bool(shape.is_valid)
    result["geometry_issue"] = None if shape.is_valid else explain_validity(shape)
    if shape.geom_type not in ("Polygon", "MultiPolygon"):
        result["box_issue"] = "non_polygon"
        return result
    bounds = shape.bounds
    if shape.is_empty or len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
        result["box_issue"] = "empty_or_nonfinite_bounds"
        return result
    result["raw_box"] = list(bounds)
    x1, y1, x2, y2 = bounds
    if x2 <= x1 or y2 <= y1:
        result["box_issue"] = "degenerate_bounds"
        return result
    clipped = [max(0., min(float(width), x1)), max(0., min(float(height), y1)),
               max(0., min(float(width), x2)), max(0., min(float(height), y2))]
    result["clipped"] = clipped != list(bounds)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        result["box_issue"] = "outside_image"
        return result
    result["box"] = clipped
    return result


def build_localization_index(data_dir: Path, assignments_path: Path | None, output_dir: Path, *,
                             source_split: str = "train") -> dict:
    """Publish scenes.jsonl + report.json atomically; never overwrite outputs.

    The xBD training source is indexed with the train/validation scene assignments. The xBD test source is
    indexed as a single split named "test" (no assignments).
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if source_split not in ("train", "test") or (assignments_path is None) != (source_split == "test"):
        raise ValueError("Index the train source with scene assignments, or the test source without them.")
    assignments_hash = file_sha256(assignments_path) if assignments_path else None
    assignments = read_assignments(assignments_path) if assignments_path else {}
    root = data_dir / "raw" / source_split
    annotations = sorted((root / "labels").glob("*_pre_disaster.json"))
    if not annotations:
        raise FileNotFoundError(f"No pre-event annotations in {root / 'labels'}")
    counts = ({split: Counter() for split in ("train", "validation", "unassigned")} if source_split == "train"
              else {"test": Counter()})
    seen_scenes: set[str] = set()
    uid_scenes: dict[str, str] = {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".localization-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "index"
        staging.mkdir()
        with (staging / "scenes.jsonl").open("w", encoding="utf-8") as stream:
            for annotation in annotations:
                scene = annotation.name.removesuffix("_pre_disaster.json")
                split = assignments.get(scene, "unassigned") if source_split == "train" else "test"
                seen_scenes.add(scene)
                image_path = root / "images" / f"{scene}_pre_disaster.png"
                for source in (annotation, image_path):
                    if not source.is_file() or not source.resolve().is_relative_to(data_dir.resolve()):
                        raise FileNotFoundError(f"Expected source inside data_dir: {source}")
                raw_annotation = annotation.read_bytes()
                record = json.loads(raw_annotation)
                features = index_features(record, annotation)
                with Image.open(image_path) as image:
                    width, height = image.size
                buildings = []
                for uid, feature in sorted(features.items()):
                    previous = uid_scenes.setdefault(uid, scene)
                    if previous != scene:
                        raise ValueError(f"Building UID {uid} spans scenes {previous}/{scene}.")
                    bounds = building_bounds(feature.get("wkt"), width, height)
                    buildings.append({"reference_uid": uid, **bounds})
                    counts[split]["buildings"] += 1
                    counts[split]["usable_boxes"] += bounds["box"] is not None
                    counts[split]["invalid_geometry"] += not bounds["geometry_valid"]
                    counts[split]["clipped_boxes"] += bounds["clipped"]
                    if bounds["box_issue"]:
                        counts[split][f"box_issue:{bounds['box_issue']}"] += 1
                unusable = any(building["box"] is None for building in buildings)
                counts[split]["scenes"] += 1
                counts[split]["empty_scenes"] += not buildings
                counts[split]["scenes_with_unusable_boxes"] += unusable
                metadata = record.get("metadata", {})
                row = {
                    "schema_version": SCHEMA_VERSION, "source_split": source_split,
                    "scene_id": scene, "split": split,
                    "pre_image": image_path.relative_to(data_dir).as_posix(),
                    "pre_annotation": annotation.relative_to(data_dir).as_posix(),
                    "annotation_sha256": hashlib.sha256(raw_annotation).hexdigest(),
                    "width": width, "height": height,
                    "event": metadata.get("disaster", scene.rsplit("_", 1)[0]),
                    "buildings": buildings,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        missing = assignments.keys() - seen_scenes
        if missing:
            raise ValueError(f"Assigned scenes missing pre annotations: {sorted(missing)[:5]}")
        if assignments_path and file_sha256(assignments_path) != assignments_hash:
            raise ValueError("Scene assignments changed while indexing.")
        report = {
            "schema_version": SCHEMA_VERSION,
            "source_split": source_split,
            "scene_assignments": str(assignments_path) if assignments_path else None,
            "scene_assignments_sha256": assignments_hash,
            "scenes_sha256": file_sha256(staging / "scenes.jsonl"),
            "counts": {split: dict(sorted(values.items())) for split, values in counts.items()},
            "policy": {
                "inputs": "source-training pre-event annotations and image dimensions only",
                "targets": "one building class; no severity or post-UID-match filtering",
                "coordinates": "continuous xyxy bounds clipped to native image edges",
                "invalid_topology": "retain finite positive polygon bounds; flag, never repair",
                "unusable_box": "retain annotation; loader refuses scene, never silently drops target",
                "unassigned_scenes": "audit only; excluded from train/validation loader",
                "empty_scenes": "retain; absence of annotations does not prove complete negative labels",
                "image_preprocessing": "loader returns native RGB float32 [0,1]; detector owns transforms",
            },
            "limitations": [
                "Existing within-event scene assignments; geographic overlap remains unaudited.",
                "Scenes absent from damage-derived assignments are not automatically added.",
                "Reference UIDs are evaluation metadata, not model inputs or production IDs.",
                "Annotation hashes are recorded; image pixels are not hashed. Treat source data as immutable.",
                "This indexes localization targets; it performs no detection or registration.",
            ],
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--source-split", choices=("train", "test"), default="train",
                        help="xBD source to index; test is indexed as one split without assignments.")
    parser.add_argument("--assignments", type=Path, default=None,
                        help="Scene assignments (train source; default datasets/processed/splits/seed42/scene_assignments.csv).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default datasets/processed/localization/seed42-v1 (train) or .../test-v1 (test).")
    args = parser.parse_args()
    training = args.source_split == "train"
    assignments = args.assignments or (Path("datasets/processed/splits/seed42/scene_assignments.csv") if training else None)
    output = args.output_dir or Path("datasets/processed/localization") / ("seed42-v1" if training else "test-v1")
    report = build_localization_index(args.data_dir, assignments, output, source_split=args.source_split)
    print(json.dumps(report["counts"], indent=2))
    print(f"Saved localization index: {output}.")


if __name__ == "__main__":
    main()
