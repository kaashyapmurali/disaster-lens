"""Native-resolution PRE images and single-class torchvision detection targets."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from disaster_lens.data.building_index import SCHEMA_VERSION, file_sha256


@dataclass
class LocalizationSample:
    scene_id: str
    image: Tensor
    target: dict[str, Tensor]
    reference_uids: list[str]  # Evaluation joins only; never passed to the detector.


class LocalizationDataset(Dataset[LocalizationSample]):
    """Load one existing development partition; never silently drop targets.

    Images are native RGB float32 [0,1], NOT ImageNet-normalized. Faster R-CNN
    owns resizing and normalization, and returns boxes in native coordinates.
    Class 1 is building; class 0 is reserved for detector background.
    Unassigned scenes cannot be selected. Empty annotated scenes are retained.
    """

    def __init__(self, index_dir: Path, data_dir: Path, split: str) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("Select train, validation or test; unassigned scenes require a split decision.")
        self.data_dir = data_dir
        report = json.loads((index_dir / "report.json").read_text())
        raw_index = (index_dir / "scenes.jsonl").read_bytes()
        self.index_sha256 = hashlib.sha256(raw_index).hexdigest()
        if report["schema_version"] != SCHEMA_VERSION or self.index_sha256 != report["scenes_sha256"]:
            raise ValueError("Unsupported or changed localization index; rebuild in a new directory.")
        self.scenes: list[dict] = []
        seen: set[str] = set()
        seen_uids: set[str] = set()
        for line in raw_index.splitlines():
            row = json.loads(line)
            scene = row["scene_id"]
            if (row["schema_version"] != SCHEMA_VERSION or row["source_split"] not in ("train", "test")
                    or row["split"] not in ("train", "validation", "unassigned", "test")
                    or (row["source_split"] == "test") != (row["split"] == "test")
                    or not scene or scene in seen):
                raise ValueError("Invalid or duplicate scene (or source/split mismatch) in localization index.")
            seen.add(scene)
            for building in row["buildings"]:
                uid = building["reference_uid"]
                if not uid or uid in seen_uids:
                    raise ValueError(f"Missing or repeated reference UID: {scene}/{uid}")
                seen_uids.add(uid)
            if row["split"] != split:
                continue
            width, height = row["width"], row["height"]
            if (not isinstance(width, int) or not isinstance(height, int)
                    or width <= 0 or height <= 0):
                raise ValueError(f"Invalid image dimensions: {scene}")
            self._source_path(row["pre_image"], scene, "images", ".png", row["source_split"])
            annotation = self._source_path(row["pre_annotation"], scene, "labels", ".json", row["source_split"])
            if file_sha256(annotation) != row["annotation_sha256"]:
                raise ValueError(f"Pre annotation changed since indexing: {scene}")
            for building in row["buildings"]:
                box = building["box"]
                if box is None or building["box_issue"]:
                    raise ValueError(f"Unusable target in {scene}/{building['reference_uid']}; inspect index audit.")
                tensor = torch.tensor(box, dtype=torch.float64)
                if (tensor.shape != (4,) or not torch.isfinite(tensor).all()
                        or (tensor[:2] < 0).any() or tensor[2] > width or tensor[3] > height
                        or (tensor[2:] <= tensor[:2]).any()):
                    raise ValueError(f"Invalid indexed box in {scene}")
            self.scenes.append(row)
        if not self.scenes:
            raise ValueError(f"No scenes for localization split {split}")
        self.scenes.sort(key=lambda row: row["scene_id"])

    def _source_path(self, text: str, scene: str, folder: str, suffix: str, source: str = "train") -> Path:
        relative = Path(text)
        expected = Path("raw") / source / folder / f"{scene}_pre_disaster{suffix}"
        if relative != expected or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a pre-event xBD {source}-source {folder} path: {text}")
        path = self.data_dir / relative
        if not path.is_file() or not path.resolve().is_relative_to(self.data_dir.resolve()):
            raise FileNotFoundError(f"Expected source inside data_dir: {path}")
        return path

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> LocalizationSample:
        row = self.scenes[index]
        path = self._source_path(row["pre_image"], row["scene_id"], "images", ".png", row["source_split"])
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"Image dimensions changed since indexing: {row['scene_id']}")
            pixels = to_tensor(image.convert("RGB"))
        boxes = torch.tensor([b["box"] for b in row["buildings"]], dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return LocalizationSample(row["scene_id"], pixels, target,
                                  [b["reference_uid"] for b in row["buildings"]])


def collate_localization(samples: list[LocalizationSample]) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    """Keep variable-size images and variable-count targets as lists, not stacks."""
    return [sample.image for sample in samples], [sample.target for sample in samples]
