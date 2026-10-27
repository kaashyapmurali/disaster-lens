"""Build the report bundle: numbers, scene imagery, the website's data file and a manifest.

    uv run --group notebook python report/scripts/build.py                 # metrics + scenes (CPU)
    uv run --group notebook --group vit python report/scripts/build.py --features   # also redo the feature images
    uv run --group notebook python report/scripts/build.py --registration  # also redo the keypoint example (uvx)

The bundle (report/bundle/) is committed. The README figures are rendered from it (report/figures/render.mjs) and the
website copies it with its own sync command, pinned to a commit of this repository.
"""

import argparse
import csv
import datetime
import json
import subprocess
import sys

from paths import BUNDLE, CLASSES, DATA, ROOT, sha256, write_json

SCHEMA_VERSION = 1
MODEL = "Fine-tuned DINOv3 ViT-L"


def rounded(points: list[list[float]]) -> list[list[float]]:
    return [[round(x * 2) / 2, round(y * 2) / 2] for x, y in points]


def site_data() -> dict:
    """Everything the website report components read, assembled from bundle/data (no new computation)."""
    def load(name: str):
        return json.loads((DATA / name).read_text())

    flow, aggregate, explorer = load("test-flow.json"), load("aggregate-metrics.json"), load("explorer.json")
    confusion = flow["footprints_confusion"]
    with (DATA / "change-measures.csv").open(newline="") as stream:
        measures = [{"event": r["event_label"], "measure": r["measure"], "x": round(float(r["bin_center"]), 4),
                     "predicted": round(float(r["mean_predicted_severity"]), 4),
                     "reference": round(float(r["mean_reference_severity"]), 4)} for r in csv.DictReader(stream)]
    return {
        "schema_version": SCHEMA_VERSION,
        "classes": CLASSES,
        "recall": load("class-recall.json"),
        "precision_known": {c: confusion[i][i] / sum(row[i] for row in confusion) for i, c in enumerate(CLASSES)},
        "flow": flow,
        "detector": {"test": aggregate["test_detector"],
                     "validation": [{k: d[k] for k in ("label", "ap", "recall", "precision", "small")}
                                    for d in aggregate["detectors"]]},
        "no_change": load("no-change.json"),
        "change_measures": measures,
        "examples": load("examples.json"),
        "dataset": load("dataset.json"),
        "example_building": load("example-building.json"),
        "matches": load("matches.json"),
        "explorer": {"size": explorer["size"], "summary": explorer["summary"],
                     "detections": [{"points": rounded(d["points"]), "label": d["label"], "confidence": d["confidence"],
                                     "score": d["score"], "outcome": d["outcome"], "reference": d["reference"]}
                                    for d in explorer["detections"]],
                     "missed": [{"points": rounded(r["points"]), "label": r["label"]} for r in explorer["references"]
                                if not r["found"] and r["label"] != "unlabeled"]},
        "scenes": load("scenes.json"),
    }


def manifest() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file() and p.name != "manifest.json")
    dirty = bool(git("status", "--porcelain", "--", ".", ":!report/bundle"))
    return {"schema_version": SCHEMA_VERSION, "project": "disaster-lens",
            "built_at": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
            "source_commit": git("rev-parse", "HEAD"), "source_dirty": dirty,
            "note": "source_commit is the commit the bundle was built from; commit the bundle itself afterwards.",
            "files": {str(p.relative_to(BUNDLE)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}}


def run(script: str, *, uvx: bool = False) -> None:
    command = (["uvx", "--quiet", "--with", "opencv-python-headless", "--with", "numpy", "python"] if uvx
               else [sys.executable])
    subprocess.run([*command, str(ROOT / "report/scripts" / script)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", action="store_true", help="Recompute the DINOv3 feature images (heavy).")
    parser.add_argument("--registration", action="store_true", help="Recompute the keypoint-matching example (uvx).")
    args = parser.parse_args()
    run("metrics.py")
    run("scenes.py")
    if args.features:
        run("features.py")
    if args.registration:
        run("registration.py", uvx=True)
    for required in ("images/features-change.jpg", "images/matches-masked.jpg", "data/matches.json"):
        if not (BUNDLE / required).is_file():
            raise FileNotFoundError(f"{required} is missing; run with --features / --registration once.")
    write_json(BUNDLE / "site.json", site_data(), compact=True)
    result = manifest()
    write_json(BUNDLE / "manifest.json", result)
    size = sum(f["bytes"] for f in result["files"].values())
    print(f"bundle: {len(result['files'])} files, {size / 1e6:.1f} MB, built from {result['source_commit'][:8]}"
          + (" (uncommitted changes)" if result["source_dirty"] else ""))


if __name__ == "__main__":
    main()
