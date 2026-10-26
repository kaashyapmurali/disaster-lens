"""Mask R-CNN building-footprint training and separate checkpoint validation export.

No downloads, remote scheduling, automatic retries/resume, or automatic export.
Use `python -m disaster_lens.detection.train {train,export} --help`.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import shapely
import torch
import torchvision
from torch.utils.data import DataLoader

from disaster_lens.detection.box_metrics import metrics_agree
from disaster_lens.detection.evaluate import evaluate_instances
from disaster_lens.detection.model import DetectorConfig, build_detector
from disaster_lens.detection.instance_data import InstanceSegmentationDataset
from disaster_lens.data.footprints import GEOMETRY_POLICIES
from disaster_lens.data.building_index import file_sha256
from disaster_lens.profiling import (
    StageTimer, batch_timing_summary, measurements, profiled_batches, start_measurement, sync,
)
from disaster_lens.detection.tiling import TiledInstancePredictor, TilingConfig
from disaster_lens.detection.crops import CropInstanceDataset
from disaster_lens.detection.data import LocalizationDataset, collate_localization

ARCHITECTURE = "pre-event-maskrcnn-resnet50-fpn-compact-v1"
# Source files that determine detector training/evaluation behaviour; their hashes are recorded per run.
IMPLEMENTATION_FILES = tuple(sorted(
    [f"detection/{path.name}" for path in (Path(__file__).resolve().parent).glob("*.py") if path.name != "__init__.py"]
    + ["data/building_index.py", "data/footprints.py", "profiling.py"]))


def _dataset(index_dir: Path, data_dir: Path, split: str, geometry_policy: str) -> InstanceSegmentationDataset:
    return InstanceSegmentationDataset(index_dir, data_dir, split, geometry_policy=geometry_policy)


def _json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _rng_snapshot(loader: DataLoader, device: torch.device) -> dict:
    """Capture future continuation state without consuming RNG or storing NumPy objects."""
    algorithm, keys, position, has_gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"algorithm": algorithm, "keys": keys.tolist(),
            "position": position, "has_gaussian": has_gaussian, "cached_gaussian": cached},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "train_loader_generator": loader.generator.get_state() if loader.generator is not None else None}


def _device(name: str) -> torch.device:
    if name not in ("cpu", "cuda"):
        raise ValueError("Localization supports explicit cpu or cuda devices.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback.")
    return torch.device(name)


def _loader(data: LocalizationDataset, batch_size: int, workers: int, *, seed: int | None = None) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=seed is not None,
                      generator=torch.Generator().manual_seed(seed) if seed is not None else None,
                      num_workers=workers, collate_fn=collate_localization)


def _smoke_subset(data: InstanceSegmentationDataset, limit: int, seed: int) -> None:
    ranked = sorted(data.scenes, key=lambda row: (-len(row["buildings"]), row["scene_id"]))
    selected = ranked[:1]
    invalid = next((row for row in sorted(data.scenes, key=lambda row: row["scene_id"])
                    if any(not building["geometry_valid"] for building in row["buildings"])
                    and row not in selected), None)
    if invalid is not None and len(selected) < limit:
        selected.append(invalid)
    remaining = [row for row in ranked if row not in selected]
    selected.extend(random.Random(seed).sample(remaining, min(limit - len(selected), len(remaining))))
    data.scenes = sorted(selected, key=lambda row: row["scene_id"])


def _validation_subset(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Deterministic event-balanced subset (round-robin over events, seeded within each event).

    Only scenes with reference buildings are eligible: a scene without any cannot support AP.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("--validation-subset must be a positive integer.")
    rows = [row for row in rows if row["buildings"]]
    by_event: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda row: row["scene_id"]):
        by_event.setdefault(row["event"], []).append(row)
    generator = random.Random(seed)
    for event_rows in by_event.values():
        generator.shuffle(event_rows)
    chosen: list[dict] = []
    while len(chosen) < min(limit, len(rows)):
        for event in sorted(by_event):
            if by_event[event] and len(chosen) < limit:
                chosen.append(by_event[event].pop())
    return sorted(chosen, key=lambda row: row["scene_id"])


def _data_identity(data: LocalizationDataset) -> dict:
    identity = {"index_sha256": data.index_sha256, "scene_ids": [row["scene_id"] for row in data.scenes],
                "buildings": sum(len(row["buildings"]) for row in data.scenes)}
    if isinstance(data, InstanceSegmentationDataset):
        identity.update(geometry_policy=data.geometry_policy, mask_representation=data.mask_representation)
    return identity


def _tensor_digest(values: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _smoke_fingerprints(model: torch.nn.Module) -> dict[str, str]:
    result = {
        "frozen": _tensor_digest((name, p) for name, p in model.named_parameters() if not p.requires_grad),
        "buffers": _tensor_digest(model.named_buffers()),
        "backbone": _tensor_digest((name, p) for name, p in model.backbone.named_parameters() if p.requires_grad),
        "heads": _tensor_digest((name, p) for name, p in model.named_parameters()
                                if p.requires_grad and name.startswith(("rpn.", "roi_heads."))),
    }
    for name in ("mask_head", "mask_predictor"):
        module = getattr(model.roi_heads, name, None)
        if module is not None:
            result[name] = _tensor_digest(module.named_parameters())
    return result


def train_epoch(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer,
) -> dict:
    """One optimizer step per scene batch; torchvision's sampled detection losses.

    Losses are NOT mean damage loss per building. Epoch reporting weights the
    batch losses by scene count. No autocast, accumulation, scheduler, or clipping.
    """
    model.train()
    totals: dict[str, float] = {}
    scene_count = 0
    building_counts = []
    started = start_measurement(device)
    profile: dict = {}
    for batch_index, ((images, targets), timing) in enumerate(profiled_batches(loader, profile), start=1):
        building_counts.append(sum(len(target["boxes"]) for target in targets))
        transfer_started = time.perf_counter()
        device_images = [image.to(device) for image in images]
        device_targets = [{key: target[key].to(device) for key in ("boxes", "labels", "masks") if key in target}
                          for target in targets]
        for original, transferred in zip(targets, device_targets, strict=True):
            if "compact_masks" in original:
                transferred["compact_masks"] = original["compact_masks"]  # Keep windows on CPU until positive ROI sampling.
        sync(device)
        timing["transfer_seconds"] = time.perf_counter() - transfer_started
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = model(device_images, device_targets)
        if not losses or any(loss.ndim != 0 or not torch.isfinite(loss) for loss in losses.values()):
            raise ValueError("Nonfinite or malformed detection losses.")
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite summed detection loss.")
        loss.backward()
        # Infinite max_norm validates finite gradient norms without clipping.
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      max_norm=float("inf"), error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        timing["forward_backward_step_seconds"] = time.perf_counter() - update_started
        reporting_started = time.perf_counter()
        for name, value in {**losses, "total": loss}.items():
            totals[name] = totals.get(name, 0.) + float(value.detach()) * len(images)
        scene_count += len(images)
        if batch_index % 100 == 0 or batch_index == len(loader):
            print(f"train: {batch_index}/{len(loader)} batches; {scene_count} scenes; "
                  f"loss={totals['total'] / scene_count:.4f}", flush=True)
        timing["reporting_seconds"] = time.perf_counter() - reporting_started
    if not scene_count:
        raise ValueError("Empty training loader.")
    return {
        "loss": totals["total"] / scene_count,
        "loss_components": {name: value / scene_count for name, value in totals.items() if name != "total"},
        "scenes": scene_count, "buildings": sum(building_counts), "optimizer_steps": len(building_counts),
        "batch_buildings_min": min(building_counts), "batch_buildings_max": max(building_counts),
        "batch_timing": batch_timing_summary(profile),
        **measurements(device, started, scene_count),
    }


def _restore_rng(state: dict | None, loader: DataLoader, device: torch.device, fallback_seed: int) -> bool:
    """Restore saved Python/NumPy/Torch/loader RNG state; else reseed deterministically.

    Returns True only for an actual restore. Checkpoints written before RNG capture was added
    cannot be replayed exactly; the fallback reseeds from `fallback_seed` and says so.
    """
    if state is None:
        random.seed(fallback_seed)
        np.random.seed(fallback_seed % 2**32)
        torch.manual_seed(fallback_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(fallback_seed)
        if loader.generator is not None:
            loader.generator.manual_seed(fallback_seed)
        return False
    python = state["python"]
    random.setstate((python[0], tuple(python[1]), python[2]))
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.array(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gaussian"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG state does not match the available device count.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if loader.generator is not None and state["train_loader_generator"] is not None:
        loader.generator.set_state(state["train_loader_generator"])
    return True


WHOLE_SCENE_TRAINING = {"mode": "whole_scene"}


def _load_parent(args: argparse.Namespace, detector, train_data, validation, protocols: dict) -> dict:
    """Load and verify a completed parent run for continuation; never modifies it."""
    run = args.resume_from
    path = run / f"{args.resume_checkpoint}.pt"
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != digest:
        raise ValueError("Parent checkpoint changed during loading.")
    config = checkpoint["config"]
    roundtrip = lambda value: json.loads(json.dumps(value))
    status = json.loads((run / "status.json").read_text())
    history = json.loads((run / "history.json").read_text())
    problems = []
    if roundtrip(config) != json.loads((run / "config.json").read_text()):
        problems.append("parent checkpoint/config.json disagree")
    if status.get("state") != "completed" or config.get("smoke_test"):
        problems.append("parent must be a completed non-smoke run")
    if config.get("architecture") != ARCHITECTURE:
        problems.append(f"architecture {config.get('architecture')} != {ARCHITECTURE}")
    if config["environment"]["torch"] != str(torch.__version__) or config["environment"]["torchvision"] != torchvision.__version__:
        problems.append("torch/torchvision versions differ")
    if roundtrip(asdict(detector)) != roundtrip(config["detector"]):
        problems.append("detector recipe (anchors/proposals/limits/thresholds) differs")
    saved = config["optimizer"]
    for key, value in (("momentum", args.momentum), ("weight_decay", args.weight_decay)):
        if saved[key] != value:
            problems.append(f"optimizer {key} {saved[key]} != {value}")
    for key, value in (("backbone_lr", args.backbone_lr), ("head_lr", args.head_lr)):
        if saved[key] != value and not args.allow_lr_change:
            problems.append(f"{key} {saved[key]} != {value} (pass --allow-lr-change to change it deliberately)")
    for key, value in (("batch_size", args.batch_size), ("validation_batch_size", args.validation_batch_size),
                       ("operating_threshold", args.operating_threshold)):
        if config[key] != value:
            problems.append(f"{key} {config[key]} != {value}")
    mask = config["mask_targets"]
    if (mask["geometry_policy"] != args.geometry_policy or mask["mask_threshold"] != args.mask_threshold
            or mask["shapely"] != shapely.__version__ or mask["geos"] != shapely.geos_version_string):
        problems.append("mask geometry policy/threshold/library versions differ")
    for split, data in (("train", train_data), ("validation", validation)):
        if roundtrip(_data_identity(data)) != roundtrip(config["data"][split]):
            problems.append(f"{split} data identity (index/scenes/buildings/policy) differs")
    if not args.allow_protocol_change:
        defaults = {"training_protocol": WHOLE_SCENE_TRAINING, "validation_protocol": _default_validation_protocol()}
        for key, default in defaults.items():
            if roundtrip(config.get(key, default)) != roundtrip(protocols[key]):
                problems.append(f"{key} differs from the parent (pass --allow-protocol-change to change it deliberately)")
    if problems:
        raise ValueError("Cannot continue; recipe/data mismatch: " + "; ".join(problems))
    return {"run": run, "checkpoint": checkpoint, "config": config, "name": args.resume_checkpoint,
            "sha256": digest, "epoch": checkpoint["epoch"],
            "best_ap": max(entry["validation"]["pooled"]["ap_50_95"] for entry in history),
            "history_epochs": [entry["epoch"] for entry in history]}


def _default_validation_protocol() -> dict:
    return {"mode": "whole_scene", "tiling": None, "subset_scene_ids": None}


def train(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    resuming = args.resume_from is not None
    if resuming == bool(args.fresh_training):
        raise ValueError("Choose exactly one: --fresh-training (new weights from the original pretrained "
                         "checkpoint; it does not continue a saved model) or --resume-from RUN_DIR "
                         "--additional-epochs N (continue a completed run).")
    if resuming:
        if (args.epochs is not None or args.additional_epochs is None or args.smoke_test or args.random_init
                or args.coco_weights or args.weights_sha256):
            raise ValueError("--resume-from needs --additional-epochs and forbids --epochs, --smoke-test, "
                             "--random-init, and pretrained-weight arguments (weights come from the parent).")
        epochs_requested = args.additional_epochs
    else:
        if args.additional_epochs is not None or args.allow_lr_change:
            raise ValueError("--additional-epochs/--allow-lr-change apply only to --resume-from.")
        epochs_requested = 5 if args.epochs is None else args.epochs
    positive = (epochs_requested, args.batch_size, args.validation_batch_size, args.backbone_lr, args.head_lr)
    if (any(not math.isfinite(value) or value <= 0 for value in positive)
            or args.num_workers < 0 or not 0 <= args.momentum < 1
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("Use positive epochs/batches/LRs, nonnegative workers/decay, and momentum in [0,1).")
    uses_crops = args.crop_size is not None
    if args.allow_protocol_change and not resuming:
        raise ValueError("--allow-protocol-change applies only to --resume-from.")
    if args.validation_subset is not None and args.smoke_test:
        raise ValueError("--validation-subset cannot be combined with --smoke-test.")
    if resuming:
        pass
    elif args.random_init:
        if not args.smoke_test or args.device != "cpu" or args.coco_weights or args.weights_sha256:
            raise ValueError("Random initialization is restricted to explicit CPU smoke tests without COCO weights.")
    elif (args.coco_weights is None or args.weights_sha256 is None
          or len(args.weights_sha256) != 64
          or any(char not in "0123456789abcdef" for char in args.weights_sha256)):
        raise ValueError("Training requires local COCO_V1 weights and their full lowercase SHA-256.")
    detector = DetectorConfig(
        min_size=args.image_size, max_size=args.image_size,
        anchor_sizes=tuple(args.anchor_sizes), trainable_backbone_layers=args.trainable_backbone_layers,
        rpn_pre_nms_top_n=args.pre_nms_proposals, rpn_post_nms_top_n=args.post_nms_proposals,
        detections_per_image=args.max_detections, score_threshold=args.score_floor,
        mask_threshold=args.mask_threshold,
    )
    if not detector.score_threshold <= args.operating_threshold <= 1:
        raise ValueError("Require score floor <= operating threshold <= 1.")
    device = _device(args.device)
    def persist_timings() -> None:
        if args.output_dir.is_dir():
            _json(args.output_dir / "timings.json", timings.snapshot())
    timings = StageTimer(on_change=persist_timings)
    with timings.measure("train_index_validation"):
        train_data = _dataset(args.index_dir, args.data_dir, "train", args.geometry_policy)
    with timings.measure("validation_index_validation"):
        validation = _dataset(args.index_dir, args.data_dir, "validation", args.geometry_policy)
    if (train_data.index_sha256 != validation.index_sha256
            or {row["scene_id"] for row in train_data.scenes} & {row["scene_id"] for row in validation.scenes}):
        raise ValueError("Train/validation index identities differ or scenes overlap.")
    if args.smoke_test:
        selector = _smoke_subset
        selector(train_data, 4, args.seed)
        selector(validation, 2, args.seed + 1)
        print("SMOKE ONLY: densest + optional repair case + seeded random subset; one epoch, not a quality benchmark.", flush=True)
    crop_data = None
    if uses_crops:
        crop_data = CropInstanceDataset(train_data, args.crop_size, args.crops_per_scene, args.seed,
                                        min_visible=args.crop_min_visible, resample_empty=args.crop_resample_empty)
    training_protocol = ({**crop_data.metadata(), "upsampling": args.image_size / args.crop_size,
                          "model_input_size": args.image_size} if crop_data else dict(WHOLE_SCENE_TRAINING))
    validation_rows = (_validation_subset(validation.scenes, args.validation_subset, args.seed + 1)
                       if args.validation_subset is not None else validation.scenes)
    tiling = None
    if args.validation_tile_size is not None:
        tiling = TilingConfig(tile_size=args.validation_tile_size, min_overlap=args.validation_tile_min_overlap,
                              tile_batch_size=args.validation_tile_batch_size, nms_threshold=detector.nms_threshold,
                              detections_per_image=detector.detections_per_image,
                              score_threshold=detector.score_threshold)
    validation_protocol = {
        "mode": "tiled" if tiling else "whole_scene", "tiling": tiling.metadata() if tiling else None,
        "subset_scene_ids": ([row["scene_id"] for row in validation_rows] if args.validation_subset is not None else None)}
    parent = (_load_parent(args, detector, train_data, validation,
                           {"training_protocol": training_protocol, "validation_protocol": validation_protocol})
              if resuming else None)
    epochs = 1 if args.smoke_test else epochs_requested
    start_epoch = parent["epoch"] + 1 if parent else 1
    last_epoch = start_epoch + epochs - 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source_root = Path(__file__).resolve().parents[1]
    config = {
        "architecture": ARCHITECTURE, "schema_version": 1, "run_id": uuid.uuid4().hex,
        "initialization_mode": "continuation" if parent else "fresh", "start_epoch": start_epoch,
        "additional_epochs": epochs,
        "lineage": ((parent["config"].get("lineage") or []) + [{
            "run_id": parent["config"]["run_id"], "checkpoint": parent["name"], "checkpoint_sha256": parent["sha256"],
            "epoch": parent["epoch"], "best_ap_50_95": parent["best_ap"]}]) if parent else [],
        "recipe_changes": ({**{key: {"parent": parent["config"]["optimizer"][key], "new": getattr(args, key)}
                               for key in ("backbone_lr", "head_lr")
                               if parent["config"]["optimizer"][key] != getattr(args, key)},
                            **{key: {"parent": parent["config"].get(key, default), "new": new}
                               for key, default, new in (("training_protocol", WHOLE_SCENE_TRAINING, training_protocol),
                                                         ("validation_protocol", _default_validation_protocol(), validation_protocol))
                               if json.loads(json.dumps(parent["config"].get(key, default))) != json.loads(json.dumps(new))}}
                           if parent else {}),
        "training_protocol": training_protocol, "validation_protocol": validation_protocol,
        "detector": asdict(detector), "recipe": detector.metadata(),
        "epochs": last_epoch, "batch_size": args.batch_size, "validation_batch_size": args.validation_batch_size,
        "optimizer": {"name": "SGD", "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
                      "momentum": args.momentum, "weight_decay": args.weight_decay},
        "selection": "maximum pooled mask AP_50_95, earliest epoch on exact ties",
        "mask_targets": {"geometry_policy": args.geometry_policy, "representation": "compact",
                         "mask_threshold": args.mask_threshold, "shapely": shapely.__version__,
                         "geos": shapely.geos_version_string, "rasterization": "native pixel-center intersection"},
        "operating_threshold": args.operating_threshold, "seed": args.seed,
        "precision": "float32", "tf32": False, "smoke_test": args.smoke_test,
        "smoke_selection": "densest, then invalid-geometry case if present, then seeded random",
        "profiling": "stage wall times plus synchronized transfer/compute and exposed loader-wait summaries",
        "expected_optimizer_steps_per_epoch": math.ceil(len(crop_data or train_data) / args.batch_size),
        "implementation_sha256": {relative: file_sha256(source_root / relative) for relative in IMPLEMENTATION_FILES},
        "data": {"train": _data_identity(train_data), "validation": _data_identity(validation)},
        "initialization": (
            {"type": "continuation", "weights_path": None, "pretrained_model": parent["config"]["initialization"]["pretrained_model"],
             "weights_sha256": parent["config"]["initialization"]["weights_sha256"],
             "parent_run_id": parent["config"]["run_id"], "parent_checkpoint_sha256": parent["sha256"]} if parent else
            {"type": "random-smoke-only" if args.random_init else "torchvision-COCO_V1",
             "weights_path": str(args.coco_weights) if args.coco_weights else None,
             "weights_sha256": args.weights_sha256,
             "pretrained_model": "maskrcnn-resnet50-fpn-coco-v1"}),
        "environment": {"torch": str(torch.__version__), "torchvision": torchvision.__version__,
                        "python": sys.version.split()[0], "device": str(device), "workers": args.num_workers,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "limitations": ["Within-event validation; geographic overlap unaudited.",
                        "Localization only, no image registration or damage integration.",
                        "Losses use torchvision anchor/ROI sampling, not uniform per-building weighting.",
                        "No scheduler, mixed precision, automatic resume, retry, or automatic export.",
                        "Image pixels assumed immutable; index and annotation identities checked."],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _json(args.output_dir / "config.json", config)
    _json(args.output_dir / "status.json", {"state": "running"})
    history = []
    persist_timings()
    try:
        with timings.measure("model_initialization_and_device_transfer"):
            builder = build_detector
            if parent:
                model = builder(detector)
                model.load_state_dict(parent["checkpoint"]["model_state_dict"], strict=True)
                model = model.to(device)
            else:
                model = builder(detector, coco_weights=args.coco_weights, expected_sha256=args.weights_sha256).to(device)
            sync(device)
        initial = None
        if args.smoke_test:
            with timings.measure("initial_smoke_fingerprints"):
                initial = _smoke_fingerprints(model)
        config["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        _json(args.output_dir / "config.json", config)
        with timings.measure("optimizer_and_loader_setup"):
            optimizer = torch.optim.SGD([
                {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr},
                {"params": list(model.rpn.parameters()) + list(model.roi_heads.parameters()), "lr": args.head_lr},
            ], momentum=args.momentum, weight_decay=args.weight_decay)
            if parent:
                optimizer.load_state_dict(parent["checkpoint"]["optimizer_state_dict"])
                for group, lr in zip(optimizer.param_groups, (args.backbone_lr, args.head_lr), strict=True):
                    group["lr"] = lr   # equal to the parent's unless --allow-lr-change was given
            train_loader = _loader(crop_data or train_data, args.batch_size, args.num_workers, seed=args.seed)
            validation_view = copy.copy(validation)
            validation_view.scenes = validation_rows
            validation_loader = _loader(validation_view, args.validation_batch_size, args.num_workers)
            validation_model = TiledInstancePredictor(model, tiling) if tiling else model
        if parent:
            config["rng_restored"] = _restore_rng(parent["checkpoint"].get("rng_state"), train_loader, device,
                                                  args.seed + parent["epoch"])
            config["rng_note"] = ("Python/NumPy/Torch/loader RNG restored from the parent checkpoint." if config["rng_restored"]
                                  else "Parent predates RNG capture: reseeded from seed + parent epoch; not an exact replay.")
            _json(args.output_dir / "config.json", config)
        best_ap = -1.
        for epoch in range(start_epoch, last_epoch + 1):
            if crop_data is not None:
                crop_data.set_epoch(epoch)
            with timings.measure(f"epoch_{epoch}_train"):
                training = train_epoch(model, train_loader, device, optimizer)
            with timings.measure(f"epoch_{epoch}_validation"):
                metrics = evaluate_instances(validation_model, validation_loader, device, config=detector,
                                                   operating_threshold=args.operating_threshold)
            ap = metrics["pooled"]["ap_50_95"]
            if ap is None:
                raise ValueError("Checkpoint selection requires validation building support.")
            entry = {"epoch": epoch, "train": training, "validation": metrics}
            history.append(entry)
            checkpoint = {"epoch": epoch, "config": config, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(), "validation": metrics,
                          "rng_state": _rng_snapshot(train_loader, device)}
            with timings.measure(f"epoch_{epoch}_last_checkpoint_write"):
                _checkpoint(args.output_dir / "last.pt", checkpoint)
            if ap > best_ap:
                best_ap = ap
                with timings.measure(f"epoch_{epoch}_best_checkpoint_write"):
                    _checkpoint(args.output_dir / "best.pt", checkpoint)
            _json(args.output_dir / "history.json", history)
            _json(args.output_dir / "status.json", {"state": "running", "completed_epoch": epoch, "best_ap_50_95": best_ap})
            print(f"Epoch {epoch}: loss={training['loss']:.4f}; AP50:95={ap:.4f}; "
                  f"AP50={metrics['pooled']['ap50']:.4f}", flush=True)
        if initial is not None:
            with timings.measure("final_smoke_fingerprints"):
                final = _smoke_fingerprints(model)
            checks = {
                "frozen_parameters_unchanged": initial["frozen"] == final["frozen"],
                "normalization_buffers_unchanged": initial["buffers"] == final["buffers"],
                "trainable_backbone_updated": initial["backbone"] != final["backbone"],
                "detection_heads_updated": initial["heads"] != final["heads"],
                "frozen_parameters_have_no_gradients": all(p.grad is None for p in model.parameters() if not p.requires_grad),
                "finite_losses_and_gradient_norms": True,  # enforced before every optimizer step
            }
            checks.update(mask_head_updated=initial["mask_head"] != final["mask_head"],
                          mask_predictor_updated=initial["mask_predictor"] != final["mask_predictor"])
            _json(args.output_dir / "smoke_checks.json", checks)
            if not all(checks.values()):
                raise RuntimeError(f"Localization smoke invariants failed: {checks}")
        _json(args.output_dir / "status.json", {
            "state": "completed", "completed_epoch": last_epoch, "best_ap_50_95": best_ap,
            **({"parent_best_ap_50_95": parent["best_ap"], "improves_on_parent": best_ap > parent["best_ap"]} if parent else {})})
        if parent and file_sha256(parent["run"] / f"{parent['name']}.pt") != parent["sha256"]:
            raise RuntimeError("Parent checkpoint changed during continuation; investigate before using this run.")
    except BaseException:
        _json(args.output_dir / "status.json", {"state": "failed", "completed_epochs": len(history)})
        raise
    return {"config": config, "history": history}


def export(args: argparse.Namespace) -> dict:
    tile_size = getattr(args, "tile_size", None)
    split = getattr(args, "split", "validation")
    if split not in ("validation", "test"):
        raise ValueError("Export the validation or test split.")
    suffix = "" if tile_size is None else f"_tile{tile_size}"
    destination = args.run_dir / ((f"{split}_predictions" if args.checkpoint_name == "best"
                                   else f"last_{split}_predictions") + suffix)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    failed_destination = args.run_dir / f"failed_{args.checkpoint_name}_{split}_predictions{suffix}"
    if failed_destination.exists():
        raise FileExistsError(f"Previous export diagnostics retained at {failed_destination}; inspect before any rerun.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Use positive batch size and nonnegative workers.")
    device = _device(args.device)
    checkpoint_path = args.run_dir / f"{args.checkpoint_name}.pt"
    digest = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if file_sha256(checkpoint_path) != digest:
        raise ValueError("Checkpoint changed during loading.")
    config = checkpoint["config"]
    run_config = json.loads((args.run_dir / "config.json").read_text())
    if json.loads(json.dumps(config)) != run_config or config["architecture"] != ARCHITECTURE:
        raise ValueError("Checkpoint/run configuration mismatch or unsupported architecture.")
    if (config["schema_version"] != 1 or config["environment"]["torchvision"] != torchvision.__version__
            or config["environment"]["torch"] != str(torch.__version__)):
        raise ValueError("Checkpoint schema or torch/torchvision versions differ; no silent recipe change.")
    mask_targets = config["mask_targets"]
    if mask_targets["shapely"] != shapely.__version__ or mask_targets["geos"] != shapely.geos_version_string:
        raise ValueError("Mask geometry implementation differs from the checkpoint.")
    validation = _dataset(args.index_dir, args.data_dir, split, mask_targets["geometry_policy"])
    if split == "validation":
        recorded = config["data"]["validation"]
        if validation.index_sha256 != recorded["index_sha256"]:
            raise ValueError("Validation index differs from training checkpoint.")
        if config["smoke_test"]:
            selected = set(recorded["scene_ids"])
            validation.scenes = [row for row in validation.scenes if row["scene_id"] in selected]
        if _data_identity(validation) != recorded:
            raise ValueError("Validation scene/building identities differ from checkpoint.")
    else:   # held-out test source: no recorded results to reproduce; never share scenes with training
        if config["smoke_test"]:
            raise ValueError("Test exports require a full (non-smoke) checkpoint.")
        recorded = _data_identity(validation)
        trained_on = set(config["data"]["train"]["scene_ids"]) | set(config["data"]["validation"]["scene_ids"])
        if trained_on & set(recorded["scene_ids"]):
            raise ValueError("Test scenes overlap the checkpoint's training/validation scenes.")
    settings = dict(config["detector"])
    settings["anchor_sizes"] = tuple(settings["anchor_sizes"])
    settings["aspect_ratios"] = tuple(settings["aspect_ratios"])
    detector = DetectorConfig(**settings)
    model = build_detector(detector)
    model.load_state_dict(checkpoint.pop("model_state_dict"), strict=True)
    # No optimizer is restored and original COCO initialization files are not needed.
    del checkpoint["optimizer_state_dict"]
    model.to(device)
    tiling = None
    protocol = config.get("validation_protocol") or _default_validation_protocol()
    if tile_size is None and (protocol["mode"] != "whole_scene" or protocol["subset_scene_ids"] is not None):
        raise ValueError("This checkpoint was validated with a tiled/subset protocol, so an untiled full export "
                         "cannot reproduce its recorded metrics; export with --tile-size (recorded in the run's "
                         "validation_protocol).")
    if tile_size is not None:
        if detector.mask_representation != "compact":
            raise ValueError("Tiled export requires the compact Mask R-CNN architecture.")
        tiling = TilingConfig(
            tile_size=tile_size, min_overlap=args.tile_min_overlap, tile_batch_size=args.tile_batch_size,
            nms_threshold=detector.nms_threshold, detections_per_image=detector.detections_per_image,
            score_threshold=detector.score_threshold)
        model = TiledInstancePredictor(model, tiling)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with tempfile.TemporaryDirectory(prefix=".localization-export-", dir=args.run_dir) as temporary:
        staging = Path(temporary) / "predictions"
        staging.mkdir()
        prediction_path = staging / "predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            metrics = evaluate_instances(model, _loader(validation, args.batch_size, args.num_workers), device,
                                               config=detector, operating_threshold=config["operating_threshold"],
                                               prediction_stream=stream, run_key=f"{config['run_id']}:{digest}")
        if split == "test":
            matched = True
            metrics["export_verification"] = "test_split_no_recorded_metrics"
        elif tiling is not None:
            # Tiled inference is a different predictor, so the checkpoint's recorded metrics are the
            # untiled baseline to compare against, not a reproduction target.
            matched = True
            metrics["export_verification"] = "tiled_inference_not_comparable_to_checkpoint_metrics"
            metrics["checkpoint_recorded_metrics"] = {
                "pooled": checkpoint["validation"]["pooled"], "box_pooled": checkpoint["validation"]["box_metrics"]["pooled"],
                "protocol": protocol}
        else:
            matched = (metrics_agree(metrics["pooled"], checkpoint["validation"]["pooled"])
                       and metrics_agree(metrics["box_metrics"]["pooled"], checkpoint["validation"]["box_metrics"]["pooled"]))
            metrics["export_verification"] = "matched_checkpoint_metrics" if matched else "failed_checkpoint_metrics"
        metrics["verification_tolerance"] = {"relative": 1e-10, "absolute": 1e-12, "counts_and_structure": "exact"}
        metrics["provenance"] = {
            "checkpoint_name": args.checkpoint_name, "checkpoint_sha256": digest, "epoch": checkpoint["epoch"],
            "run_id": config["run_id"], "index_sha256": validation.index_sha256, "split": split,
            "predictions_sha256": file_sha256(prediction_path), "smoke_test": config["smoke_test"],
            "detector": config["recipe"], "mask_targets": config.get("mask_targets"),
            "scene_ids": recorded["scene_ids"],
            "reference_buildings": recorded["buildings"], "device": str(device),
            "tiling": None if tiling is None else tiling.metadata(),
        }
        _json(staging / "metrics.json", metrics)
        if not matched:
            _json(staging / "comparison_failure.json", {
                "expected_primary": checkpoint["validation"]["pooled"], "actual_primary": metrics["pooled"],
                "expected_box": checkpoint["validation"].get("box_metrics", {}).get("pooled"),
                "actual_box": metrics.get("box_metrics", {}).get("pooled"),
            })
            staging.rename(failed_destination)
            raise ValueError(f"Reloaded checkpoint metrics differ; export not published. "
                             f"Diagnostics retained at {failed_destination}; no automatic retry.")
        staging.rename(destination)
    return metrics


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train, evaluate each epoch, save best/last; no automatic export.")
    exporting = commands.add_parser("export", help="Reload best/last and export validation detections separately.")
    for command in (training, exporting):
        command.add_argument("--data-dir", type=Path, default=Path("datasets"))
        command.add_argument("--index-dir", type=Path, default=Path("datasets/processed/localization/seed42-v1"))
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--num-workers", type=int, default=0)
        command.add_argument("--batch-size", type=int, default=1, help="Scenes per batch.")
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--fresh-training", action="store_true",
                          help="Start new weights from the original pretrained checkpoint (not a continuation).")
    training.add_argument("--resume-from", type=Path, help="Completed parent run directory to continue; never modified.")
    training.add_argument("--resume-checkpoint", choices=("best", "last"), default="last",
                          help="Parent checkpoint to continue from (last = most recent weights).")
    training.add_argument("--additional-epochs", type=int, help="Epochs to add when using --resume-from.")
    training.add_argument("--allow-lr-change", action="store_true",
                          help="Permit different learning rates than the parent; recorded under recipe_changes.")
    training.add_argument("--allow-protocol-change", action="store_true",
                          help="Permit a different training/validation protocol than the parent; recorded under recipe_changes.")
    training.add_argument("--crop-size", type=int, default=None,
                          help="Train on random crops of this size, upsampled by the model's own resize (e.g. 512 -> 2x).")
    training.add_argument("--crops-per-scene", type=int, default=1)
    training.add_argument("--crop-min-visible", type=float, default=.5,
                          help="Keep a building in a crop only if this fraction of its mask remains.")
    training.add_argument("--crop-resample-empty", type=int, default=5)
    training.add_argument("--validation-subset", type=int, default=None,
                          help="Validate each epoch on a seeded event-balanced subset of this many scenes.")
    training.add_argument("--validation-tile-size", type=int, default=None, help="Validate with overlapping tiles.")
    training.add_argument("--validation-tile-min-overlap", type=int, default=128)
    training.add_argument("--validation-tile-batch-size", type=int, default=3)
    training.add_argument("--geometry-policy", choices=GEOMETRY_POLICIES, default="make-valid-polygonal",
                          help="How to rasterize the few invalid annotation polygons (strict refuses them).")
    training.add_argument("--mask-threshold", type=float, default=.5)
    training.add_argument("--coco-weights", type=Path)
    training.add_argument("--weights-sha256")
    training.add_argument("--random-init", action="store_true", help="Only allowed with --smoke-test --device cpu.")
    training.add_argument("--smoke-test", action="store_true", help="One epoch, <=4 train/2 validation scenes; not quality evaluation.")
    training.add_argument("--epochs", type=int, default=None, help="Fresh training only (default 5).")
    training.add_argument("--validation-batch-size", type=int, default=1)
    training.add_argument("--image-size", type=int, default=1024)
    training.add_argument("--anchor-sizes", type=int, nargs=5, default=(16, 32, 64, 128, 256))
    training.add_argument("--trainable-backbone-layers", type=int, default=3)
    training.add_argument("--pre-nms-proposals", type=int, default=4000)
    training.add_argument("--post-nms-proposals", type=int, default=3000)
    training.add_argument("--max-detections", type=int, default=2500)
    training.add_argument("--score-floor", type=float, default=.001, help="Low retained-score floor for PR/AP evaluation.")
    training.add_argument("--operating-threshold", type=float, default=.5, help="Fixed precision/recall diagnostic, not a tuned threshold.")
    training.add_argument("--backbone-lr", type=float, default=.001)
    training.add_argument("--head-lr", type=float, default=.005)
    training.add_argument("--momentum", type=float, default=.9)
    training.add_argument("--weight-decay", type=float, default=.0005)
    training.add_argument("--seed", type=int, default=42)
    exporting.add_argument("--run-dir", type=Path, required=True)
    exporting.add_argument("--checkpoint-name", choices=("best", "last"), default="best")
    exporting.add_argument("--split", choices=("validation", "test"), default="validation",
                           help="test uses a test-source index (data.building_index --source-split test).")
    exporting.add_argument("--tile-size", type=int, default=None,
                           help="Optional overlapping-tile inference (writes validation_predictions_tile<N>/).")
    exporting.add_argument("--tile-min-overlap", type=int, default=128)
    exporting.add_argument("--tile-batch-size", type=int, default=3)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "train":
        train(args)
    else:
        metrics = export(args)
        print(json.dumps(metrics["pooled"], indent=2))


if __name__ == "__main__":
    main()
