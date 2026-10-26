"""Scene datasets: each scene loaded once with all of its labeled buildings (post-only or paired dates)."""

import csv
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from shapely import wkt
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from disaster_lens.data.manifest import DAMAGE_LABELS

CLASS_TO_INDEX = {label: index for index, label in enumerate(DAMAGE_LABELS)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Building:
    uid: str
    box: tuple[float, float, float, float]
    label: int


@dataclass
class Scene:
    scene_id: str
    image_path: Path
    buildings: list[Building] = field(default_factory=list)


@dataclass
class SceneSample:
    scene_id: str
    image: Tensor
    boxes: Tensor
    labels: Tensor
    building_uids: list[str]


class SceneDataset(Dataset[SceneSample]):
    """Post-event scenes with pre-annotation boxes and known severity labels.

    No pre-event pixels are read. Building positions still come from supplied
    pre-event annotations: this assumes the paired image grids correspond.
    Images are resized to a square; box coordinates are rescaled consistently.
    No augmentation or black-pixel filtering is applied in this first baseline.
    """

    def __init__(self, manifest: Path, data_dir: Path, image_size: int = 1024) -> None:
        if image_size < 64 or image_size % 32:
            raise ValueError("image_size must be a multiple of 32 and at least 64.")
        self.image_size = image_size
        grouped: dict[str, Scene] = {}
        seen = set()
        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if (
                    row["source_split"] not in ("train", "test")
                    or row["match_status"] != "matched"
                    or row["damage_label"] not in CLASS_TO_INDEX
                ):
                    raise ValueError("Expected matched, known-label xBD train/test examples.")
                scene_id, uid = row["scene_id"], row["building_uid"]
                if (scene_id, uid) in seen:
                    raise ValueError(f"Duplicate building: {scene_id}/{uid}")
                seen.add((scene_id, uid))
                image_path = data_dir / row["post_image"]
                scene = grouped.setdefault(scene_id, Scene(scene_id, image_path))
                if scene.image_path != image_path:
                    raise ValueError(f"Inconsistent post-image paths for scene {scene_id}")
                geometry = wkt.loads(row["pre_wkt_xy"])
                bounds = geometry.bounds
                if (
                    geometry.geom_type not in ("Polygon", "MultiPolygon")
                    or len(bounds) != 4
                    or not all(math.isfinite(value) for value in bounds)
                    or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]
                ):
                    raise ValueError(f"Invalid building bounds: {scene_id}/{uid}")
                scene.buildings.append(Building(uid, bounds, CLASS_TO_INDEX[row["damage_label"]]))
        self.scenes = [grouped[key] for key in sorted(grouped)]
        if not self.scenes:
            raise ValueError(f"Empty scene manifest: {manifest}")
        for scene in self.scenes:
            if not scene.image_path.is_file():
                raise FileNotFoundError(scene.image_path)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> SceneSample:
        scene = self.scenes[index]
        with Image.open(scene.image_path) as image:
            width, height = image.size
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            pixels = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        boxes = torch.tensor([building.box for building in scene.buildings], dtype=torch.float32)
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, width) * (self.image_size / width)
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, height) * (self.image_size / height)
        if not torch.all((boxes[:, 2:] - boxes[:, :2]) > 0):
            raise ValueError(f"Building box lies outside post-image bounds: {scene.scene_id}")
        return SceneSample(
            scene_id=scene.scene_id,
            image=pixels,
            boxes=boxes,
            labels=torch.tensor([b.label for b in scene.buildings], dtype=torch.long),
            building_uids=[b.uid for b in scene.buildings],
        )


@dataclass
class PairedSceneSample:
    scene_id: str
    building_uids: list[str]
    labels: Tensor
    pre_image: Tensor
    post_image: Tensor
    pre_boxes: Tensor
    post_boxes: Tensor


@dataclass
class PairedSceneBatch:
    scene_ids: list[str]
    building_uids: list[list[str]]
    labels: list[Tensor]
    pre_images: Tensor
    post_images: Tensor
    pre_boxes: list[Tensor]
    post_boxes: list[Tensor]


class PairedSceneDataset(Dataset[PairedSceneSample]):
    """Deterministic paired extraction with explicitly recorded RGB normalization.

    Reuse the validated building manifest index from SceneDataset. Both dates
    use pre-event footprint bounds; no post-event annotation geometry is used.
    Equal image dimensions are required, but do not establish true registration.
    There is no augmentation, black-pixel filtering, or learned preprocessing.
    """

    def __init__(
        self, manifest: Path, data_dir: Path, image_size: int = 1024, *,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        if (len(mean) != 3 or len(std) != 3
                or not all(math.isfinite(value) for value in (*mean, *std))
                or any(value <= 0 for value in std)):
            raise ValueError("RGB normalization requires three finite means and positive finite standard deviations.")
        with manifest.open("rb") as stream:
            initial_manifest_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        indexed = SceneDataset(manifest, data_dir, image_size)
        self.manifest = manifest
        self.data_dir = data_dir
        self.image_size = image_size
        self.mean, self.std = tuple(mean), tuple(std)
        self.scenes = indexed.scenes
        self.pre_paths: dict[str, Path] = {}
        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                raw_path = row["pre_image"].strip()
                relative = Path(raw_path)
                if not raw_path or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Pre-image paths must be nonempty and relative to data_dir.")
                path = data_dir / relative
                previous = self.pre_paths.setdefault(row["scene_id"], path)
                if previous != path:
                    raise ValueError(f"Inconsistent pre-image paths for {row['scene_id']}")
        for scene in self.scenes:
            for path in (self.pre_paths[scene.scene_id], scene.image_path):
                if not path.is_file() or not path.resolve().is_relative_to(data_dir.resolve()):
                    raise FileNotFoundError(f"Expected input image inside data_dir: {path}")
        with manifest.open("rb") as stream:
            self.manifest_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if self.manifest_sha256 != initial_manifest_sha256:
            raise ValueError("Manifest changed while indexing paired data; recreate the dataset.")

    def preprocessing_config(self) -> dict:
        return {
            "image_size": self.image_size, "resize": "PIL bilinear square",
            "color": "RGB", "mean": list(self.mean), "std": list(self.std),
            "boxes": "pre-event bounds, clipped to common input grid and rescaled",
            "augmentation": False, "black_pixel_filter": False,
        }

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> PairedSceneSample:
        scene = self.scenes[index]
        pixels = {}
        sizes = {}
        for phase, path in (("pre", self.pre_paths[scene.scene_id]), ("post", scene.image_path)):
            with Image.open(path) as image:
                sizes[phase] = image.size
                resized = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR,
                )
                pixels[phase] = TF.normalize(TF.to_tensor(resized), self.mean, self.std)
        if sizes["pre"] != sizes["post"]:
            raise ValueError(f"{scene.scene_id}: unequal source grids need explicit coordinate mapping.")
        width, height = sizes["pre"]
        boxes = torch.tensor([building.box for building in scene.buildings], dtype=torch.float32)
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, width) * (self.image_size / width)
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, height) * (self.image_size / height)
        if not torch.all((boxes[:, 2:] - boxes[:, :2]) > 0):
            raise ValueError(f"{scene.scene_id}: a building is outside the shared image grid.")
        return PairedSceneSample(
            scene_id=scene.scene_id, building_uids=[b.uid for b in scene.buildings],
            labels=torch.tensor([b.label for b in scene.buildings], dtype=torch.long),
            pre_image=pixels["pre"], post_image=pixels["post"],
            pre_boxes=boxes, post_boxes=boxes.clone(),
        )


def collate_pairs(samples: list[PairedSceneSample]) -> PairedSceneBatch:
    return PairedSceneBatch(
        scene_ids=[sample.scene_id for sample in samples],
        building_uids=[sample.building_uids for sample in samples],
        labels=[sample.labels for sample in samples],
        pre_images=torch.stack([sample.pre_image for sample in samples]),
        post_images=torch.stack([sample.post_image for sample in samples]),
        pre_boxes=[sample.pre_boxes for sample in samples],
        post_boxes=[sample.post_boxes for sample in samples],
    )
