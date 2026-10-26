"""Persist frozen paired features once, then read them without images/encoders.

    uv run python -m disaster_lens.damage.feature_cache --help

Each split has metadata.json and one tensor shard per scene. Existing compatible
shards are reused; interrupted builds resume. No trainable projections are saved.
"""

import argparse
import hashlib
import inspect
import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import PIL
import torch
import torchvision
from filelock import FileLock
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from disaster_lens.damage.features import PairedFeatureExtractor, PairedSceneFeatures
from disaster_lens.data.scenes import CLASS_TO_INDEX, SceneDataset
from disaster_lens.data.scenes import PairedSceneDataset, collate_pairs

CACHE_VERSION = 1


def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _state_digest(extractor: PairedFeatureExtractor) -> str:
    """Identify actual parameters AND buffers, not just a checkpoint filename."""
    digest = hashlib.sha256()
    state = dict(extractor.backbone.state_dict())
    # Native timm RoPE periods are non-persistent but still affect every feature.
    state.update((name, tensor) for name, tensor in extractor.backbone.named_buffers() if name not in state)
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cache_spec(dataset: PairedSceneDataset, extractor: PairedFeatureExtractor) -> dict:
    """Describe cache validity without decoding images or executing the encoder.

    Manifest/weights/code fingerprints are content-based. Image provenance uses
    relative path, byte size and mtime: treat source images as immutable. Pixel
    edits that preserve BOTH size and mtime require a new cache directory/version.
    These fingerprints are cache keys, not raw-download checksum verification.
    """
    if not extractor.freeze_backbone or any(p.requires_grad for p in extractor.parameters()):
        raise ValueError("Persistent caches require a frozen backbone.")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in extractor.parameters()):
        raise ValueError("This cache pipeline currently requires float32 encoder weights.")
    with dataset.manifest.open("rb") as stream:
        manifest_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if manifest_digest != dataset.manifest_sha256:
        raise ValueError("Manifest changed since indexing; recreate PairedSceneDataset.")
    indexed_buildings = hashlib.sha256()
    source_images = []
    for scene in dataset.scenes:
        if not scene.scene_id or any(not (c.isalnum() or c in "-_") for c in scene.scene_id):
            raise ValueError(f"Unsafe scene ID: {scene.scene_id}")
        indexed_buildings.update(json.dumps([
            scene.scene_id, [(b.uid, b.box, b.label) for b in scene.buildings],
        ]).encode())
        for path in (dataset.pre_paths[scene.scene_id], scene.image_path):
            stat = path.stat()
            source_images.append((path.resolve().relative_to(dataset.data_dir.resolve()).as_posix(),
                                  stat.st_size, stat.st_mtime_ns))
    classes = {SceneDataset, PairedSceneDataset, type(dataset), PairedFeatureExtractor, type(extractor.backbone)}
    classes.update(type(module) for module in extractor.modules()
                   if type(module).__module__.startswith("disaster_lens."))
    implementation = {
        f"{cls.__module__}.{cls.__qualname__}": inspect.getsource(cls)
        for cls in classes
    }
    spec = {
        "cache_version": CACHE_VERSION,
        "backbone": f"{type(extractor.backbone).__module__}.{type(extractor.backbone).__qualname__}",
        "encoder_state_sha256": _state_digest(extractor),
        "implementation_sha256": _json_digest(implementation),
        "manifest_sha256": manifest_digest,
        "building_index_sha256": indexed_buildings.hexdigest(),
        "image_provenance_sha256": _json_digest(source_images),
        "image_provenance_policy": "relative path + size + mtime_ns; immutable input images",
        "preprocessing": dataset.preprocessing_config(),
        "roi": {"size": extractor.roi_size, "sampling_ratio": 2, "aligned": True,
                "spatial_scale": extractor.backbone.spatial_scale},
        "building_dim": extractor.building_dim, "context_dim": extractor.context_dim,
        "dtype": "float32", "class_to_index": CLASS_TO_INDEX,
        "torch_version": str(torch.__version__), "torchvision_version": torchvision.__version__,
        "pillow_version": PIL.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    backbone_config = extractor.backbone.cache_config()
    if backbone_config:
        spec["backbone_config"] = backbone_config
    return spec


def _atomic_write(path: Path, value: dict, *, tensors: bool = False) -> None:
    """Never expose a half-written JSON document or tensor shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary_path = Path(stream.name)
            if tensors:
                torch.save(value, stream)
            else:
                stream.write((json.dumps(value, indent=2) + "\n").encode())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass
class CachedScene:
    scene_id: str
    building_uids: list[str]
    labels: Tensor
    features: PairedSceneFeatures


@dataclass
class CachedSceneBatch:
    scene_ids: list[str]
    building_uids: list[str]
    labels: Tensor
    features: PairedSceneFeatures


def _read_scene(path: Path, metadata: dict, entry: dict) -> CachedScene:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["cache_id"] != metadata["cache_id"] or payload["scene_id"] != entry["scene_id"]:
        raise ValueError(f"Shard identity mismatch: {path}")
    uids, labels = payload["building_uids"], payload["labels"]
    count = entry["buildings"]
    if len(uids) != count or len(set(uids)) != count or labels.shape != (count,):
        raise ValueError(f"Building/label count mismatch in {path}")
    if labels.dtype != torch.long or ((labels < 0) | (labels >= len(CLASS_TO_INDEX))).any():
        raise ValueError(f"Invalid cached labels in {path}")
    features = PairedSceneFeatures(**payload["features"])
    for name, tensor in features.as_dict().items():
        if name == "building_to_scene":
            valid = tensor.shape == (count,) and tensor.dtype == torch.long and not tensor.any()
        else:
            dim = metadata["spec"]["building_dim" if "building" in name else "context_dim"]
            valid = tensor.shape == (count if "building" in name else 1, dim) and tensor.dtype == torch.float32
        if not valid:
            raise ValueError(f"Invalid {name} shape/dtype in {path}")
    return CachedScene(payload["scene_id"], uids, labels, features)


def build_cache(
    dataset: PairedSceneDataset,
    extractor: PairedFeatureExtractor,
    cache_dir: Path,
    *,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 1,
    num_workers: int = 0,
    on_batch: Callable[[dict], None] | None = None,
) -> dict[str, int]:
    """Extract only missing scenes; refuse incompatible cache reuse.

    The file lock prevents concurrent writers and releases after process death.
    Compatibility is checked before any image decoding or backbone execution.
    """
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Use a positive batch_size and nonnegative num_workers.")
    spec = cache_spec(dataset, extractor)
    cache_id = _json_digest(spec)
    entries = [{"scene_id": scene.scene_id, "file": f"scenes/{scene.scene_id}.pt",
                "buildings": len(scene.buildings)} for scene in dataset.scenes]
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    with FileLock(str(cache_dir / ".writer.lock"), timeout=0):
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["spec"] != spec or metadata["scenes"] != entries:
                raise ValueError("Cache configuration or inputs changed. Choose a new cache directory.")
            if metadata["cache_id"] != cache_id:
                raise ValueError("Cache identity is inconsistent with its configuration.")
        else:
            if (cache_dir / "scenes").exists():
                raise ValueError("Unindexed cache shards exist; inspect the directory before rebuilding.")
            metadata = {"cache_id": cache_id, "spec": spec, "scenes": entries, "complete": False}
            _atomic_write(metadata_path, metadata)
        missing = [i for i, entry in enumerate(entries) if not (cache_dir / entry["file"]).is_file()]
        reused = len(entries) - len(missing)
        if metadata["complete"] and not missing:
            return {"written_scenes": 0, "reused_scenes": reused,
                    "buildings": sum(entry["buildings"] for entry in entries)}
        metadata["complete"] = False
        _atomic_write(metadata_path, metadata)
        # Validate committed shards before resuming a partial extraction.
        missing_set = set(missing)
        for i, entry in enumerate(entries):
            if i not in missing_set:
                cached = _read_scene(cache_dir / entry["file"], metadata, entry)
                source = dataset.scenes[i]
                if cached.building_uids != [b.uid for b in source.buildings] or cached.labels.tolist() != [b.label for b in source.buildings]:
                    raise ValueError(f"Cached building correspondence changed for {source.scene_id}")
        loader = DataLoader(Subset(dataset, missing), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_pairs)
        extractor.to(device).eval()
        entry_by_scene = {entry["scene_id"]: entry for entry in entries}
        written = 0
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            waiting_started = time.perf_counter()
            for batch in loader:
                ready_at = time.perf_counter()
                if on_batch is not None and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                transfer_started = time.perf_counter()
                pre_images = batch.pre_images.to(device)
                post_images = batch.post_images.to(device)
                if on_batch is not None and device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_started = time.perf_counter()
                features = extractor(
                    pre_images, post_images, batch.pre_boxes, batch.post_boxes,
                )
                if on_batch is not None and device.type == "cuda":
                    torch.cuda.synchronize(device)
                writing_started = time.perf_counter()
                offset = 0
                for index, scene_id in enumerate(batch.scene_ids):
                    count = len(batch.building_uids[index])
                    stop = offset + count
                    local = PairedSceneFeatures(
                        features.pre_building[offset:stop].detach().to(device="cpu", dtype=torch.float32, copy=True),
                        features.post_building[offset:stop].detach().to(device="cpu", dtype=torch.float32, copy=True),
                        features.pre_context[index:index+1].detach().to(device="cpu", dtype=torch.float32, copy=True),
                        features.post_context[index:index+1].detach().to(device="cpu", dtype=torch.float32, copy=True),
                        torch.zeros(count, dtype=torch.long),
                    )
                    if any(not tensor.isfinite().all() for tensor in local.as_dict().values()):
                        raise ValueError(f"Non-finite features for {scene_id}")
                    payload = {"cache_id": cache_id, "scene_id": scene_id,
                               "building_uids": batch.building_uids[index],
                               "labels": batch.labels[index].cpu(), "features": local.as_dict()}
                    _atomic_write(cache_dir / entry_by_scene[scene_id]["file"], payload, tensors=True)
                    offset = stop
                    written += 1
                    if written % 50 == 0 or written == len(missing):
                        print(f"Cached {written}/{len(missing)} new scenes; reused {reused}.", flush=True)
                finished_at = time.perf_counter()
                del features, pre_images, post_images
                if on_batch is not None:
                    on_batch({
                        "scene_ids": batch.scene_ids,
                        "buildings_per_scene": [len(uids) for uids in batch.building_uids],
                        "scenes": len(batch.scene_ids), "buildings": offset,
                        "data_wait_seconds": ready_at - waiting_started,
                        "image_transfer_seconds": forward_started - transfer_started,
                        "forward_seconds": writing_started - forward_started,
                        "feature_transfer_and_write_seconds": finished_at - writing_started,
                        "batch_seconds": finished_at - waiting_started,
                        "peak_gpu_allocated_bytes": (
                            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                        ),
                        "peak_gpu_reserved_bytes": (
                            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                        ),
                    })
                waiting_started = time.perf_counter()
        if cache_spec(dataset, extractor) != spec:
            raise ValueError("Inputs or encoder state changed during extraction; cache remains incomplete.")
        metadata["complete"] = True
        _atomic_write(metadata_path, metadata)
        return {"written_scenes": written, "reused_scenes": reused,
                "buildings": sum(entry["buildings"] for entry in entries)}


class CachedFeatureDataset(Dataset[CachedScene]):
    """Read frozen features without constructing an encoder or opening images."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.metadata = json.loads((cache_dir / "metadata.json").read_text())
        if not self.metadata["complete"]:
            raise ValueError("Feature cache is incomplete; resume extraction first.")
        spec = self.metadata["spec"]
        if spec["cache_version"] != CACHE_VERSION or self.metadata["cache_id"] != _json_digest(spec):
            raise ValueError("Unsupported or inconsistent cache metadata.")
        if spec["class_to_index"] != CLASS_TO_INDEX:
            raise ValueError("Cached label mapping differs from the current mapping.")
        self.building_dim, self.context_dim = spec["building_dim"], spec["context_dim"]
        entries = self.metadata["scenes"]
        if not entries or len({entry["scene_id"] for entry in entries}) != len(entries):
            raise ValueError("Cached scene index is empty or contains duplicate scene IDs.")
        if len({entry["file"] for entry in entries}) != len(entries):
            raise ValueError("Cached scene index contains duplicate shard paths.")
        for entry in entries:
            path = cache_dir / entry["file"]
            if not path.resolve().is_relative_to(cache_dir.resolve()) or not path.is_file():
                raise FileNotFoundError(f"Missing or invalid scene shard: {path}")

    def __len__(self) -> int:
        return len(self.metadata["scenes"])

    def __getitem__(self, index: int) -> CachedScene:
        entry = self.metadata["scenes"][index]
        return _read_scene(self.cache_dir / entry["file"], self.metadata, entry)


def collate_cached_scenes(samples: list[CachedScene]) -> CachedSceneBatch:
    """Rebuild batch-local scene indices while retaining one context per scene."""
    features = PairedSceneFeatures(
        torch.cat([sample.features.pre_building for sample in samples]),
        torch.cat([sample.features.post_building for sample in samples]),
        torch.cat([sample.features.pre_context for sample in samples]),
        torch.cat([sample.features.post_context for sample in samples]),
        torch.cat([torch.full((len(sample.labels),), i, dtype=torch.long) for i, sample in enumerate(samples)]),
    )
    return CachedSceneBatch(
        scene_ids=[sample.scene_id for sample in samples],
        building_uids=[uid for sample in samples for uid in sample.building_uids],
        labels=torch.cat([sample.labels for sample in samples]), features=features,
    )


def build_extraction_inputs(
    manifests: dict[str, Path], data_dir: Path, image_size: int, roi_size: int, *,
    encoder: str = "resnet18", encoder_dir: Path | None = None,
) -> tuple[dict[str, PairedSceneDataset], PairedFeatureExtractor]:
    """Datasets with the encoder's normalization, plus a frozen extractor (shared by caches and their checks)."""
    from disaster_lens.damage.classifier import build_encoder, normalization

    mean, std = normalization(encoder)
    datasets = {split: PairedSceneDataset(manifest, data_dir, image_size, mean=mean, std=std)
                for split, manifest in manifests.items()}
    if "train" in datasets and "validation" in datasets:
        ids = {split: {scene.scene_id for scene in data.scenes} for split, data in datasets.items()}
        uids = {split: {b.uid for scene in data.scenes for b in scene.buildings} for split, data in datasets.items()}
        if ids["train"] & ids["validation"] or uids["train"] & uids["validation"]:
            raise ValueError("Train/validation overlap detected.")
    backbone = build_encoder(encoder, weights_dir=encoder_dir)
    if encoder == "dinov3-vitl" and (backbone.patch_size != 16 or backbone.spatial_channels != 1024
                                     or len(backbone.encoder.blocks) != 24):
        raise ValueError("Expected a ViT-L/16 checkpoint (1024 channels, 24 transformer layers).")
    return datasets, PairedFeatureExtractor(backbone, roi_size=roi_size, freeze_backbone=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/features/resnet18-paired-v1"))
    parser.add_argument("--encoder", choices=("resnet18", "dinov3-vitl"), default="resnet18")
    parser.add_argument("--encoder-dir", type=Path, default=None,
                        help="Directory containing the local timm DINOv3 model.safetensors; never downloaded.")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--roi-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--profile-output", type=Path, default=None,
                        help="Optional JSON batch timings/memory; refuses an existing file.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (args.encoder == "dinov3-vitl") != (args.encoder_dir is not None):
        parser.error("--encoder-dir is required for dinov3-vitl and not used by resnet18.")
    if args.encoder == "dinov3-vitl":
        if args.cache_dir == Path("datasets/features/resnet18-paired-v1"):
            parser.error("Choose a separate --cache-dir for DINOv3; do not reuse the ResNet cache path.")
        if args.image_size % 16:
            parser.error("DINOv3 image dimensions must be divisible by 16.")
    if args.profile_output is not None and args.profile_output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.profile_output}")
    started = time.perf_counter()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    split_dir = args.split_dir or args.data_dir / "processed/splits/seed42"
    datasets, extractor = build_extraction_inputs(
        {split: split_dir / f"{split}.csv" for split in args.splits},
        args.data_dir, args.image_size, args.roi_size,
        encoder=args.encoder, encoder_dir=args.encoder_dir,
    )
    profile = {
        "complete": False, "batch_size_scenes": args.batch_size,
        "num_workers": args.num_workers, "device": str(device),
        "image_size": args.image_size, "roi_size": args.roi_size,
        "encoder": args.encoder,
        "precision": "float32; autocast disabled",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_bytes": (
            torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
        ),
        "splits": {},
    }
    for split, dataset in datasets.items():
        split_profile = {"batches": []}
        profile["splits"][split] = split_profile

        def record_batch(metrics: dict) -> None:
            split_profile["batches"].append(metrics)
            _atomic_write(args.profile_output, profile)
            print(f"Extraction batch {split}: {metrics['scenes']} pairs; "
                  f"{metrics['buildings']} buildings; {metrics['batch_seconds']:.3f}s; "
                  f"peak GPU allocated={metrics['peak_gpu_allocated_bytes']}", flush=True)

        split_started = time.perf_counter()
        summary = build_cache(dataset, extractor, args.cache_dir / split, device=device,
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              on_batch=record_batch if args.profile_output is not None else None)
        split_profile.update(cache_summary=summary, elapsed_seconds=time.perf_counter() - split_started)
        if args.profile_output is not None:
            _atomic_write(args.profile_output, profile)
        print(f"{split}: {json.dumps(summary)}", flush=True)
    if args.profile_output is not None:
        profile.update(complete=True, elapsed_seconds=time.perf_counter() - started)
        _atomic_write(args.profile_output, profile)


if __name__ == "__main__":
    main()
