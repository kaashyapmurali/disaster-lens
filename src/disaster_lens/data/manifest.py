"""Index building annotations without generating crops or choosing training targets.

    uv run python -m disaster_lens.data.manifest --split train

Paths in the CSV are relative to the dataset directory. Only matched buildings
with one of the four known severity labels enter the supervised manifest. Raw
annotations are unchanged; exclusions are counted before any train/validation split.
"""

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

DAMAGE_LABELS = ("no-damage", "minor-damage", "major-damage", "destroyed")
FIELDS = (
    "source_split", "scene_id", "building_uid", "event", "disaster_type",
    "match_status", "raw_damage_label", "damage_label",
    "pre_image", "post_image", "pre_annotation", "post_annotation",
    "pre_wkt_xy", "post_wkt_xy", "pre_capture_date", "post_capture_date",
    "pre_sensor", "post_sensor",
)


def index_features(record: dict, source: Path) -> dict[str, dict]:
    """Match by annotation UID, never by feature-list order."""
    indexed: dict[str, dict] = {}
    for feature in record["features"]["xy"]:
        if feature["properties"].get("feature_type") != "building":
            continue
        uid = feature["properties"]["uid"]
        if not uid or uid in indexed:
            raise ValueError(f"Missing or duplicate building UID {uid!r} in {source}")
        indexed[uid] = feature
    return indexed


def iter_buildings(data_dir: Path, split: str) -> Iterator[dict[str, str]]:
    """Yield one row per scene/building UID, including unmatched annotations."""
    root = data_dir / "raw" / split
    label_dir = root / "labels"
    scene_ids = sorted({
        path.name.removesuffix(f"_{phase}_disaster.json")
        for phase in ("pre", "post")
        for path in label_dir.glob(f"*_{phase}_disaster.json")
    })
    if not scene_ids:
        raise FileNotFoundError(f"No scene annotations found in {label_dir}")

    for scene_id in scene_ids:
        paths = {
            phase: {
                "image": root / "images" / f"{scene_id}_{phase}_disaster.png",
                "annotation": label_dir / f"{scene_id}_{phase}_disaster.json",
            }
            for phase in ("pre", "post")
        }
        for phase_paths in paths.values():
            for path in phase_paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete scene {scene_id}: {path}")
        records = {
            phase: json.loads(paths[phase]["annotation"].read_text())
            for phase in ("pre", "post")
        }
        features = {
            phase: index_features(records[phase], paths[phase]["annotation"])
            for phase in ("pre", "post")
        }
        metadata = {phase: records[phase]["metadata"] for phase in ("pre", "post")}
        for uid in sorted(features["pre"].keys() | features["post"].keys()):
            pre = features["pre"].get(uid)
            post = features["post"].get(uid)
            raw_label = post["properties"].get("subtype", "") if post else ""
            status = "matched" if pre and post else "missing_pre" if post else "missing_post"
            row = {
                "source_split": split,
                "scene_id": scene_id,
                "building_uid": uid,
                "event": metadata["post"].get("disaster", scene_id.rsplit("_", 1)[0]),
                "disaster_type": metadata["post"].get("disaster_type", "unknown"),
                "match_status": status,
                "raw_damage_label": raw_label or "",
                "damage_label": raw_label if raw_label in DAMAGE_LABELS else "unknown",
                "pre_wkt_xy": pre["wkt"] if pre else "",
                "post_wkt_xy": post["wkt"] if post else "",
            }
            for phase in ("pre", "post"):
                for kind, path in paths[phase].items():
                    row[f"{phase}_{kind}"] = path.relative_to(data_dir).as_posix()
                row[f"{phase}_capture_date"] = metadata[phase].get("capture_date", "")
                row[f"{phase}_sensor"] = metadata[phase].get("sensor", "")
            yield row


def build_manifest(data_dir: Path, split: str, output: Path) -> Counter[str]:
    """Write matched, labeled examples atomically; never overwrite a manifest."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    counts: Counter[str] = Counter()
    try:
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in iter_buildings(data_dir, split):
                counts["source_rows"] += 1
                if row["damage_label"] not in DAMAGE_LABELS:
                    counts["excluded:unknown_damage"] += 1
                    continue
                if row["match_status"] != "matched":
                    counts["excluded:unmatched_building"] += 1
                    continue
                writer.writerow(row)
                counts["rows"] += 1
                counts[f"label:{row['damage_label']}"] += 1
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()
    output = args.data_dir / "processed" / f"{args.split}_buildings.csv"
    counts = build_manifest(args.data_dir, args.split, output)
    print(f"Manifest: {output}")
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count:,}")
    print("No crops, numeric targets, or train/validation assignments were generated.")


if __name__ == "__main__":
    main()
