"""Locations of the report pipeline's inputs (local, not in Git) and of the committed bundle."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "report/bundle"
DATA, IMAGES, SCENES, MEDIA = (BUNDLE / name for name in ("data", "images", "scenes", "media"))

# Local inputs: model outputs and xBD imagery (see report/README.md for how they are produced).
DATASETS = ROOT / "datasets"
TEST_IMAGES = DATASETS / "raw/test"
TEST_MANIFEST = DATASETS / "processed/test_buildings.csv"
VALIDATION_MANIFEST = DATASETS / "processed/splits/seed42/validation.csv"
TEST_EXPORT = ROOT / "outputs/modal/segmenter-crops-l4-02/test_predictions_tile512"
VALIDATION_EXPORT = ROOT / "outputs/modal/segmenter-crops-l4-02/validation_predictions_tile512"
DAMAGE_RUN = ROOT / "outputs/modal/test-e2e-vit-finetuned"                   # fine-tuned ViT, test split
NO_CHANGE_RUN = ROOT / "outputs/modal/test-e2e-vit-finetuned-no-change"       # pre-event image given as both dates
VALIDATION_PREDICTIONS = ROOT / "outputs/modal/vit-finetune-bf16-l4-20261020-01/validation_predictions/predictions.csv"
CHANGE_CACHE = ROOT / "outputs/analysis/report/change_features.csv"           # per-building measures, stays local
DINOV3 = DATASETS / "pretrained/dinov3-vitl16-sat-timm-ce148bd5"

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
COLORS = {"building": "#35d0ff", "no-damage": "#3ccf6e", "minor-damage": "#ffd23f",
          "major-damage": "#ff8a2a", "destroyed": "#ff3b4e", "unlabeled": "#b9c4c9"}
NAMES = {"no-damage": "No damage", "minor-damage": "Minor", "major-damage": "Major", "destroyed": "Destroyed",
         "unlabeled": "Not rated"}


def write_json(path: Path, value, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, separators=(",", ":")) if compact else json.dumps(value, indent=1)
    path.write_text(text + "\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
