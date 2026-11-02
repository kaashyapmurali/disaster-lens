"""Build the phase-B fine-tuning split: Hurricane Ian drone tiles plus an equal share of xBD training scenes.

    uv run python drone/make_split.py   # -> outputs/drone/inputs/splits/crasar-ian-mix-v1/{train,validation}.csv

train.csv: every Ian training building (outputs/drone/inputs/crasar-ian/train.csv, from drone/modal_crasar_prep.py,
Kennedy Green excluded: its pre-event image is panchromatic) plus whole xBD training scenes drawn at random (seed 42)
until they hold at least as many buildings. validation.csv: the Ian validation tiles (model selection). Upload with
`modal volume put disaster-lens-data <dir> /processed/splits/crasar-ian-mix-v1`.
"""

import collections
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IAN = ROOT / "outputs/drone/inputs/crasar-ian"
XBD = ROOT / "datasets/processed/splits/seed42/train.csv"
OUT = ROOT / "outputs/drone/inputs/splits/crasar-ian-mix-v1"


def main() -> None:
    ian_train = [r for r in csv.DictReader(open(IAN / "train.csv")) if "Kennedy-Green" not in r["scene_id"]]
    ian_val = list(csv.DictReader(open(IAN / "validation.csv")))
    fields = csv.DictReader(open(XBD)).fieldnames
    by_scene = collections.defaultdict(list)
    for r in csv.DictReader(open(XBD)):
        by_scene[r["scene_id"]].append(r)
    scenes = sorted(by_scene)
    random.Random(42).shuffle(scenes)
    picked, n = [], 0
    for s in scenes:
        if n >= len(ian_train):
            break
        picked.append(s); n += len(by_scene[s])
    xbd_rows = [r for s in sorted(picked) for r in by_scene[s]]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", ian_train + xbd_rows), ("validation", ian_val)):
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(f"{name}: {len(rows)} buildings in {len({r['scene_id'] for r in rows})} scenes")


if __name__ == "__main__":
    main()
