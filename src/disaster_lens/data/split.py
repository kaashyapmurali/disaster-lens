"""Create an approximately stratified 80/20 split, grouping buildings by scene.

    uv run python -m disaster_lens.data.split --seed 42

Only the filtered source-training manifest is read. This is within-event
validation, not held-out-event evaluation or a verified spatial separation.
"""

import argparse
import csv
import json
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from disaster_lens.data.manifest import DAMAGE_LABELS


KEYS = ("scene_id", "building_uid", "event", "disaster_type", "damage_label")


def read_examples(manifest: Path) -> list[dict[str, str]]:
    """Load split-relevant columns only; leave large polygon strings on disk."""
    examples = []
    seen = set()
    uid_scenes: dict[str, str] = {}
    scene_metadata: dict[str, tuple[str, str]] = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row["source_split"] != "train"
                or row["match_status"] != "matched"
                or row["damage_label"] not in DAMAGE_LABELS
            ):
                raise ValueError("Use a source-training manifest of matched, known-label buildings.")
            example = {key: row[key] for key in KEYS}
            scene, uid = example["scene_id"], example["building_uid"]
            if not all(example.values()) or (scene, uid) in seen:
                raise ValueError(f"Missing fields or duplicate example: {scene}/{uid}")
            seen.add((scene, uid))
            if uid in uid_scenes and uid_scenes[uid] != scene:
                raise ValueError(f"Building UID {uid} spans scenes; use larger leakage groups.")
            uid_scenes[uid] = scene
            metadata = (example["event"], example["disaster_type"])
            if scene in scene_metadata and scene_metadata[scene] != metadata:
                raise ValueError(f"Inconsistent event metadata in scene {scene}")
            scene_metadata[scene] = metadata
            examples.append(example)
    if not examples:
        raise ValueError("Manifest has no labeled buildings.")
    return examples


def choose_assignments(
    examples: list[dict[str, str]], seed: int,
) -> tuple[dict[str, str], int, float, list[str]]:
    """Choose among five folds using counts only, never model performance.

    Favor 20% validation coverage in each event/hazard/damage stratum, each
    aggregate category, total buildings, and total scenes. Exact balance is not
    generally possible because a scene can contain multiple damage classes.
    """
    groups = np.array([row["scene_id"] for row in examples])
    strata = np.array([
        "|".join(row[key] for key in ("event", "disaster_type", "damage_label"))
        for row in examples
    ])
    category_arrays = [
        strata,
        *[np.array([row[key] for row in examples])
          for key in ("event", "disaster_type", "damage_label")],
    ]
    encoded = []
    for values in category_arrays:
        _, indices = np.unique(values, return_inverse=True)
        encoded.append((indices, np.bincount(indices)))
    total_scenes = len(set(groups))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        for fold, (_, val_indices) in enumerate(splitter.split(np.zeros(len(groups)), strata, groups)):
            fractions = [
                float(np.mean(np.abs(np.bincount(indices[val_indices], minlength=len(total)) / total - 0.2)))
                for indices, total in encoded
            ]
            val_scenes = set(groups[val_indices])
            score = (
                sum(fractions)
                + abs(len(val_indices) / len(examples) - 0.2)
                + abs(len(val_scenes) / total_scenes - 0.2)
            )
            candidates.append((score, fold, val_scenes))
    score, fold, val_scenes = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    assignments = {str(scene): "validation" if scene in val_scenes else "train" for scene in sorted(set(groups))}
    return assignments, fold, score, [str(warning.message) for warning in captured]


def summarize(examples: list[dict[str, str]], assignments: dict[str, str]) -> dict:
    """Record marginal/joint counts and rare-stratum limitations."""
    report: dict = {"counts": {}, "strata": []}
    for split in ("train", "validation"):
        rows = [row for row in examples if assignments[row["scene_id"]] == split]
        report["counts"][split] = {
            "buildings": len(rows),
            "scenes": len({row["scene_id"] for row in rows}),
            **{key: dict(sorted(Counter(row[key] for row in rows).items()))
               for key in ("event", "disaster_type", "damage_label")},
        }
    joint: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    support: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in examples:
        key = (row["event"], row["disaster_type"], row["damage_label"])
        joint[key][assignments[row["scene_id"]]] += 1
        support[key].add(row["scene_id"])
    for key, counts in sorted(joint.items()):
        total = sum(counts.values())
        report["strata"].append({
            "event": key[0], "disaster_type": key[1], "damage_label": key[2],
            "total": total, "supporting_scenes": len(support[key]),
            "train": counts["train"], "validation": counts["validation"],
            "validation_fraction": counts["validation"] / total,
            "present_in_both": bool(counts["train"] and counts["validation"]),
        })
    return report


def create_split(manifest: Path, output_dir: Path, seed: int = 42) -> dict:
    """Write assignments, disjoint building manifests, and a distribution report."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    examples = read_examples(manifest)
    assignments, fold, score, split_warnings = choose_assignments(examples, seed)
    report = summarize(examples, assignments)
    report.update({
        "input_manifest": str(manifest), "seed": seed,
        "method": "StratifiedGroupKFold on event/disaster_type/damage_label; scene groups",
        "n_folds": 5, "selected_validation_fold": fold, "selection_score": score,
        "target_validation_fraction": 0.2, "sklearn_version": sklearn.__version__,
        "warnings": split_warnings,
        "limitations": [
            "Within-event validation; events are represented in both splits where feasible.",
            "Scene IDs and building UIDs are disjoint; geographic overlap and repeated physical buildings with different UIDs have not been audited.",
            "Rare strata may be absent from one split; inspect strata in this report.",
        ],
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".split-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "split"
        staging.mkdir()
        with (staging / "scene_assignments.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("scene_id", "split"))
            writer.writerows(sorted(assignments.items()))
        with manifest.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            assert reader.fieldnames is not None
            with (staging / "train.csv").open("w", newline="", encoding="utf-8") as train_file, \
                 (staging / "validation.csv").open("w", newline="", encoding="utf-8") as val_file:
                writers = {"train": csv.DictWriter(train_file, reader.fieldnames),
                           "validation": csv.DictWriter(val_file, reader.fieldnames)}
                for writer in writers.values():
                    writer.writeheader()
                for row in reader:
                    writers[assignments[row["scene_id"]]].writerow(row)
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/processed/train_buildings.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output_dir or args.manifest.parent / "splits" / f"seed{args.seed}"
    report = create_split(args.manifest, output_dir, args.seed)
    print(f"Saved split to {output_dir}")
    for split, counts in report["counts"].items():
        print(f"{split}: {counts['scenes']:,} scenes; {counts['buildings']:,} buildings")
        print(f"  damage labels: {counts['damage_label']}")
    for stratum in report["strata"]:
        if not stratum["present_in_both"]:
            print(f"WARNING: stratum absent from one split: {stratum}")
    print("See report.json for event, disaster-type, and joint distributions.")
    print("Test/holdout were not read or modified. Spatial-overlap audit remains pending.")


if __name__ == "__main__":
    main()
