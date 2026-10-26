"""Train and export damage classifiers on live images, for any encoder × inputs × trainable-blocks choice.

    python -m disaster_lens.damage.train_live train --encoder dinov3-vitl --inputs paired --trainable-blocks 4 ...
    python -m disaster_lens.damage.train_live export --run-dir RUN --data-dir DATA --manifest validation.csv

`--trainable-blocks 0` keeps the encoder frozen (only the head trains). The paired, frozen models can also be
trained much faster from cached features with `damage.feature_cache` + `damage.train_cached`. A paired head trained
that way can initialize live fine-tuning (`--init-head`), after checking that the live encoder, preprocessing and
ROI recipe match the features it was trained on.

Exports also load the two earlier checkpoint formats used for the reported models (the partially fine-tuned
DINOv3 ViT-L and the post-only ResNet-18 baseline).
"""

import argparse
import csv
import hashlib
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.nn import functional as F
from torch.utils.data import DataLoader

from disaster_lens.damage.classifier import (
    ENCODERS, INPUTS, DamageClassifier, build_encoder, build_head, normalization,
)
from disaster_lens.damage.export_cached import ORDINAL_PREDICTION_COLUMNS, PROBABILITY_COLUMNS, validate_prediction_rows
from disaster_lens.damage.feature_cache import _atomic_write, cache_spec
from disaster_lens.damage.features import PairedFeatureExtractor
from disaster_lens.damage.heads import TemporalFusionHead
from disaster_lens.damage.severity import SEVERITY_COLUMNS, severity_outputs
from disaster_lens.damage.train_cached import classification_metrics
from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.data.scenes import CLASS_TO_INDEX, PairedSceneDataset, collate_pairs

ARCHITECTURE = "live-damage-classifier-v1"
LEGACY_FINETUNED = "live-paired-partial-vit-finetune"
LEGACY_POST_ONLY = "resnet18-layer2-ROI-plus-layer4-global-linear"
SPEC_KEYS = ("backbone", "encoder_state_sha256", "manifest_sha256", "building_index_sha256", "preprocessing", "roi",
             "building_dim", "context_dim", "dtype", "class_to_index", "backbone_config", "float32_matmul_precision")


def _validate_precision(precision: str, device: torch.device) -> None:
    if precision not in ("float32", "bf16"):
        raise ValueError("precision must be 'float32' or 'bf16'.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Live training supports CPU or CUDA.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        with torch.cuda.device(device):
            if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError("Native BF16 is unavailable on the requested CUDA device; no precision fallback.")


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _dataset(manifest: Path, data_dir: Path, image_size: int, encoder: str) -> PairedSceneDataset:
    mean, std = normalization(encoder)
    return PairedSceneDataset(manifest, data_dir, image_size, mean=mean, std=std)


def _parameter_digest(module: torch.nn.Module, *, trainable: bool | None = None) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if trainable is not None and parameter.requires_grad != trainable:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _smoke_subset(data: PairedSceneDataset, limit: int, seed: int) -> None:
    ranked = sorted(data.scenes, key=lambda scene: (-len(scene.buildings), scene.scene_id))
    data.scenes = ranked[:1] + random.Random(seed).sample(ranked[1:], min(limit, len(ranked)) - 1)


def _validate_splits(train: PairedSceneDataset, validation: PairedSceneDataset) -> None:
    if {s.scene_id for s in train.scenes} & {s.scene_id for s in validation.scenes}:
        raise ValueError("Training and validation share scene IDs.")
    uids = []
    for data in (train, validation):
        values = [b.uid for scene in data.scenes for b in scene.buildings]
        if len(set(values)) != len(values):
            raise ValueError("Duplicate building UIDs within a split.")
        uids.append(set(values))
    if uids[0] & uids[1]:
        raise ValueError("Training and validation share building UIDs.")


def _data_spec(data: PairedSceneDataset) -> dict:
    return {"manifest_sha256": data.manifest_sha256, "preprocessing": data.preprocessing_config()}


def load_cached_head(backbone, head_checkpoint: Path, splits: dict[str, PairedSceneDataset], roi_size: int) -> tuple[dict, dict]:
    """A cross-entropy fusion head trained on cached features, verified against the live encoder recipe."""
    checkpoint = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
    source = checkpoint["config"]
    if (source["architecture"] != "cached-paired-temporal-fusion"
            or source.get("objective", "cross-entropy") != "cross-entropy" or source["class_to_index"] != CLASS_TO_INDEX):
        raise ValueError("--init-head must be a cached cross-entropy temporal-fusion head with the four original classes.")
    frozen = PairedFeatureExtractor(backbone, roi_size=roi_size, freeze_backbone=True)
    for split, data in splits.items():
        live, recorded = cache_spec(data, frozen), source["caches"][split]["spec"]
        if any(live.get(key) != recorded.get(key) for key in SPEC_KEYS):
            raise ValueError(f"Live {split} feature recipe differs from the head's cached features.")
    return checkpoint, {"init_head_sha256": _file_digest(head_checkpoint), "init_head_epoch": checkpoint["epoch"],
                        "init_head_cache_ids": {split: source["caches"][split]["cache_id"] for split in splits}}


def run_live_epoch(model: DamageClassifier, loader: DataLoader, device: torch.device,
                   optimizer: torch.optim.Optimizer | None = None, *, export_rows: bool = False,
                   precision: str = "float32") -> tuple[dict, list[dict[str, str]]]:
    """One update per scene batch (mean cross-entropy over its buildings); no feature cache."""
    training = optimizer is not None
    _validate_precision(precision, device)
    if training and export_rows:
        raise ValueError("Prediction export must run in evaluation mode without an optimizer.")
    model.train(training)
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict[str, str]] = []
    counts = []
    total_loss = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch in loader:
        labels = torch.cat(batch.labels).to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(batch.pre_images.to(device), batch.post_images.to(device), batch.pre_boxes)
            losses = F.cross_entropy(logits.float(), labels, reduction="none")
            if not logits.isfinite().all() or not losses.isfinite().all():
                raise ValueError(f"Non-finite logits/loss in scenes {batch.scene_ids}")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses.mean().backward()
                if any(p.grad is not None and not p.grad.isfinite().all() for p in model.parameters()):
                    raise ValueError(f"Non-finite gradients in scenes {batch.scene_ids}")
                optimizer.step()
        probabilities = logits.detach().float().softmax(dim=1).cpu()
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(probabilities.argmax(dim=1).tolist())
        counts.append(len(labels))
        total_loss += losses.detach().sum().item()
        if export_rows:
            scores = severity_outputs(probabilities.double()).tolist()
            offset = 0
            for scene_id, uids, scene_labels in zip(batch.scene_ids, batch.building_uids, batch.labels, strict=True):
                for uid, label in zip(uids, scene_labels.tolist(), strict=True):
                    values = probabilities[offset].tolist()
                    rows.append({
                        "scene_id": scene_id, "building_uid": uid, "reference_label": DAMAGE_LABELS[label],
                        "predicted_label": DAMAGE_LABELS[max(range(4), key=values.__getitem__)],
                        **{column: repr(value) for column, value in zip(PROBABILITY_COLUMNS, values, strict=True)},
                        **{column: repr(value) for column, value in zip(SEVERITY_COLUMNS, scores[offset], strict=True)},
                    })
                    offset += 1
        if len(counts) % 100 == 0:
            print(f"{'train' if training else 'validation'}: {len(counts)}/{len(loader)} scene batches; "
                  f"{len(targets):,} buildings", flush=True)
    if not targets:
        raise ValueError("No building targets were processed.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        **classification_metrics(targets, predictions),
        "loss": total_loss / len(targets), "elapsed_seconds": elapsed, "buildings_per_second": len(targets) / elapsed,
        "optimizer_steps": len(counts) if training else 0, "precision": precision,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "buildings_per_batch": {"min": min(counts), "max": max(counts), "mean": len(targets) / len(counts)},
    }, rows


def train_live(
    encoder: str, inputs: str, data_dir: Path, split_dir: Path, output_dir: Path, *,
    trainable_blocks: int = 0, encoder_dir: Path | None = None, init_head: Path | None = None,
    epochs: int = 5, batch_size: int = 1, validation_batch_size: int = 1, backbone_lr: float = 1e-5,
    head_lr: float = 1e-4, weight_decay: float = 1e-4, image_size: int = 1024, roi_size: int = 3,
    num_workers: int = 0, seed: int = 42, device: torch.device = torch.device("cpu"), precision: str = "float32",
    smoke_test: bool = False, backbone: torch.nn.Module | None = None,
) -> list[dict]:
    """Create a fresh run (never resumes or overwrites); best checkpoint by validation macro-F1.

    `backbone` substitutes a prebuilt encoder (tests); otherwise it is built from `encoder`/`encoder_dir`.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if encoder not in ENCODERS or inputs not in INPUTS:
        raise ValueError(f"Choose --encoder from {ENCODERS} and --inputs from {INPUTS}.")
    if min(epochs, batch_size, validation_batch_size) < 1 or num_workers < 0:
        raise ValueError("Epochs/batch sizes must be positive; num_workers must be nonnegative.")
    if (not all(math.isfinite(value) and value > 0 for value in (backbone_lr, head_lr))
            or not math.isfinite(weight_decay) or weight_decay < 0 or not 0 <= seed < 2**32):
        raise ValueError("Invalid learning rates, weight decay, or seed.")
    if init_head is not None and inputs != "paired":
        raise ValueError("--init-head applies to the paired fusion head only.")
    _validate_precision(precision, device)
    train, validation = (_dataset(split_dir / f"{split}.csv", data_dir, image_size, encoder) for split in ("train", "validation"))
    _validate_splits(train, validation)
    backbone = backbone if backbone is not None else build_encoder(encoder, weights_dir=encoder_dir)
    head_kwargs = {"projected_dim": 128, "hidden_dim": 128, "dropout": .1} if inputs == "paired" else {}
    initialization = {"head": "random"}
    head_state = None
    if init_head is not None:
        source, initialization = load_cached_head(backbone, init_head, {"train": train, "validation": validation}, roi_size)
        head_kwargs = {k: v for k, v in source["config"]["model"].items() if k not in ("building_dim", "context_dim", "num_classes")}
        head_state = source["model_state_dict"]
    head = build_head(inputs, backbone.spatial_channels * roi_size**2, backbone.context_dim, **head_kwargs)
    if head_state is not None:
        head.load_state_dict(head_state, strict=True)
    model = DamageClassifier(backbone, head, inputs=inputs, trainable_blocks=trainable_blocks, roi_size=roi_size)
    if smoke_test:
        if epochs != 1 or batch_size != 1 or validation_batch_size != 1:
            raise ValueError("Smoke runs use one epoch and one scene per batch.")
        _smoke_subset(train, 4, seed)
        _smoke_subset(validation, 2, seed + 1)
        print("SMOKE ONLY: at most four training / two validation scenes; not a performance estimate.", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.to(device)
    initial = {"frozen": _parameter_digest(backbone, trainable=False), "suffix": _parameter_digest(backbone, trainable=True),
               "head": _parameter_digest(model.head)} if smoke_test else None
    optimizer = torch.optim.AdamW(model.parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr), weight_decay=weight_decay)
    head_config = {"building_dim": head.building_dim, "context_dim": head.context_dim,
                   "num_classes": len(DAMAGE_LABELS), **head_kwargs}
    config = {
        "architecture": ARCHITECTURE, "encoder": encoder, "inputs": inputs, "trainable_blocks": trainable_blocks,
        "roi_size": roi_size, "head": {"type": type(head).__name__, **head_config},
        "backbone_constructor": backbone.constructor() if hasattr(backbone, "constructor") else None,
        "initialization": initialization, "objective": "cross-entropy", "class_to_index": CLASS_TO_INDEX,
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "validation_batch_size": validation_batch_size,
        "batch_unit": "scenes", "smoke_test": smoke_test, "precision": precision,
        "data_specs": {"train": _data_spec(train), "validation": _data_spec(validation)},
        "train_scene_ids": [scene.scene_id for scene in train.scenes],
        "validation_scene_ids": [scene.scene_id for scene in validation.scenes],
        "optimizer": {"name": "AdamW", "backbone_lr": backbone_lr, "head_lr": head_lr,
                      **{key: value for key, value in optimizer.defaults.items() if key != "lr"}},
        "selection_metric": "validation.macro_f1", "selection_tie_break": "earliest epoch",
        "device": str(device), "num_workers": num_workers, "torch_version": str(torch.__version__),
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
    }
    if encoder == "dinov3-vitl":
        import timm
        config["timm_version"] = timm.__version__
    loaders = {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
                            num_workers=num_workers, collate_fn=collate_pairs),
        "validation": DataLoader(validation, batch_size=validation_batch_size, num_workers=num_workers, collate_fn=collate_pairs),
    }
    output_dir.mkdir(parents=True)
    _atomic_write(output_dir / "config.json", config)
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        trained, _ = run_live_epoch(model, loaders["train"], device, optimizer, precision=precision)
        evaluated, _ = run_live_epoch(model, loaders["validation"], device, precision=precision)
        history.append({"epoch": epoch, "train": trained, "validation": evaluated})
        checkpoint = {"epoch": epoch, "config": config, "model_state_dict": model.state_dict(),
                      "backbone_buffers": {name: tensor.detach().cpu() for name, tensor in backbone.named_buffers()},
                      "optimizer_state_dict": optimizer.state_dict(), "validation": evaluated}
        _atomic_write(output_dir / "last.pt", checkpoint, tensors=True)
        if evaluated["macro_f1"] > best:
            best = evaluated["macro_f1"]
            _atomic_write(output_dir / "best.pt", checkpoint, tensors=True)
        _atomic_write(output_dir / "history.json", {"epochs": history})
        print(f"Epoch {epoch}: train loss={trained['loss']:.4f}; val loss={evaluated['loss']:.4f}; "
              f"val macro-F1={evaluated['macro_f1']:.4f}", flush=True)
    if initial is not None:
        checks = {
            "frozen_parameters_unchanged": _parameter_digest(backbone, trainable=False) == initial["frozen"],
            "trainable_encoder_updated": trainable_blocks == 0 or _parameter_digest(backbone, trainable=True) != initial["suffix"],
            "head_updated": _parameter_digest(model.head) != initial["head"],
            "frozen_parameters_have_no_gradients": all(p.grad is None for p in backbone.parameters() if not p.requires_grad),
            "parameters_remain_float32": all(p.dtype == torch.float32 for p in model.parameters()),
            "finite_losses": all(math.isfinite(item[phase]["loss"]) for item in history for phase in ("train", "validation")),
        }
        _atomic_write(output_dir / "smoke_checks.json", checks)
        if not all(checks.values()):
            raise RuntimeError(f"Smoke checks failed: {checks}")
        print("Smoke checks passed.", flush=True)
    return history


def _restore_buffers(backbone, saved: dict) -> None:
    buffers = dict(backbone.named_buffers())
    if set(buffers) != set(saved):
        raise ValueError("Checkpoint backbone buffers do not match the reconstructed model.")
    with torch.no_grad():
        for name, value in saved.items():
            if value.shape != buffers[name].shape or value.dtype != buffers[name].dtype:
                raise ValueError(f"Incompatible saved backbone buffer: {name}")
            buffers[name].copy_(value)


def load_live_model(checkpoint: dict, device: torch.device) -> tuple[DamageClassifier, dict]:
    """Rebuild a saved classifier without pretrained files. Returns (model, recipe) where recipe has the encoder,
    precision and image preprocessing needed to reproduce its inputs."""
    config = checkpoint["config"]
    if config["class_to_index"] != CLASS_TO_INDEX:
        raise ValueError("Checkpoint label mapping differs from the four original classes.")
    architecture = config["architecture"]
    if architecture == LEGACY_POST_ONLY:          # reported post-only baseline (earlier standalone format)
        if not config["encoder_frozen"]:
            raise ValueError("Expected the frozen-encoder post-only baseline.")
        backbone = build_encoder("resnet18", pretrained=False)
        head = build_head("post", backbone.spatial_channels * config["roi_size"]**2, backbone.context_dim)
        model = DamageClassifier(backbone, head, inputs="post", trainable_blocks=0, roi_size=config["roi_size"])
        state = {("extractor.backbone." + key if key.startswith("encoder.") else key.replace("classifier.", "head.linear.")): value
                 for key, value in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
        recipe = {"encoder": "resnet18", "precision": "float32", "image_size": config["image_size"], "data_specs": None}
        return model.to(device).eval(), recipe
    if architecture == LEGACY_FINETUNED:          # reported fine-tuned ViT (earlier format)
        import timm
        if config["timm_version"] != timm.__version__:
            raise ValueError("Restore using the recorded timm version.")
        from disaster_lens.damage.dinov3 import DINOv3Backbone
        backbone = DINOv3Backbone(timm.create_model(config["backbone_name"], pretrained=False, **config["backbone_constructor"]))
        head = TemporalFusionHead(**config["model"])
        encoder, inputs, blocks = "dinov3-vitl", "paired", config["trainable_blocks"]
    elif architecture == ARCHITECTURE:
        encoder, inputs, blocks = config["encoder"], config["inputs"], config["trainable_blocks"]
        if encoder == "dinov3-vitl":
            import timm
            if config["timm_version"] != timm.__version__:
                raise ValueError("Restore using the recorded timm version.")
            from disaster_lens.damage.dinov3 import DINOv3Backbone
            backbone = DINOv3Backbone.untrained(**config["backbone_constructor"])
        else:
            backbone = build_encoder(encoder, pretrained=False)
        settings = {k: v for k, v in config["head"].items() if k not in ("type", "building_dim", "context_dim", "num_classes")}
        head = build_head(inputs, config["head"]["building_dim"], config["head"]["context_dim"], **settings)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    _validate_precision(config["precision"], device)
    model = DamageClassifier(backbone, head, inputs=inputs, trainable_blocks=blocks, roi_size=config["roi_size"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _restore_buffers(backbone, checkpoint["backbone_buffers"])
    recipe = {"encoder": encoder, "precision": config["precision"],
              "image_size": config["data_specs"]["validation"]["preprocessing"]["image_size"], "data_specs": config["data_specs"]}
    return model.to(device).eval(), recipe


def export_live(run_dir: Path, data_dir: Path, manifest: Path, *, batch_size: int = 1, num_workers: int = 0,
                device: torch.device = torch.device("cpu"), checkpoint_name: str = "best") -> dict:
    """Predict every building of `manifest` with a saved checkpoint and publish checked rows atomically.

    Exports `validation_predictions/` for the recorded validation manifest; any other manifest (e.g. a test split)
    goes to `<manifest stem>_predictions/`.
    """
    if checkpoint_name not in ("best", "last") or batch_size < 1 or num_workers < 0:
        raise ValueError("Use best/last checkpoint, positive batch_size, and nonnegative num_workers.")
    path = run_dir / f"{checkpoint_name}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model, recipe = load_live_model(checkpoint, device)
    data = _dataset(manifest, data_dir, recipe["image_size"], recipe["encoder"])
    recorded = (recipe["data_specs"] or {}).get("validation")
    is_validation = recorded is not None and data.manifest_sha256 == recorded["manifest_sha256"]
    if recorded is not None and data.preprocessing_config() != recorded["preprocessing"]:
        raise ValueError("Image preprocessing differs from the training run.")
    stem = "validation" if is_validation or manifest.stem == "validation" else manifest.stem
    output = run_dir / (f"{stem}_predictions" if checkpoint_name == "best" else f"last_{stem}_predictions")
    with FileLock(str(run_dir / ".live_export.lock"), timeout=0):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        if checkpoint["config"].get("smoke_test", False) and is_validation:
            indexed = {scene.scene_id: scene for scene in data.scenes}
            data.scenes = [indexed[scene] for scene in checkpoint["config"]["validation_scene_ids"]]
        loader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_pairs)
        metrics, rows = run_live_epoch(model, loader, device, export_rows=True, precision=recipe["precision"])
        expected = [(scene.scene_id, b.uid, DAMAGE_LABELS[b.label]) for scene in data.scenes for b in scene.buildings]
        validate_prediction_rows(rows, expected)
        metrics.update(checkpoint=path.name, checkpoint_sha256=_file_digest(path), epoch=checkpoint["epoch"],
                       architecture=checkpoint["config"]["architecture"], encoder=recipe["encoder"], split=stem,
                       manifest_sha256=data.manifest_sha256, preprocessing=data.preprocessing_config(), device=str(device),
                       class_to_index=CLASS_TO_INDEX,
                       severity_index_definition="expected category rank / 3; not physical percent damage")
        with tempfile.TemporaryDirectory(prefix=".live-export-", dir=run_dir) as temporary:
            staging = Path(temporary) / "export"
            staging.mkdir()
            with (staging / "predictions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ORDINAL_PREDICTION_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with (staging / "predictions.csv").open(newline="") as stream:
                validate_prediction_rows(list(csv.DictReader(stream)), expected)
            _atomic_write(staging / "metrics.json", metrics)
            staging.rename(output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a classifier on live images.")
    train.add_argument("--encoder", choices=ENCODERS, required=True)
    train.add_argument("--inputs", choices=INPUTS, required=True)
    train.add_argument("--trainable-blocks", type=int, default=0, help="0 = frozen encoder; N = train the last N blocks.")
    train.add_argument("--encoder-dir", type=Path, help="Local DINOv3 timm checkpoint directory (dinov3-vitl).")
    train.add_argument("--init-head", type=Path, help="Cached-feature fusion head to start from (paired only).")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--split-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--validation-batch-size", type=int, default=1)
    train.add_argument("--backbone-lr", type=float, default=1e-5)
    train.add_argument("--head-lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--image-size", type=int, default=1024)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    train.add_argument("--smoke-test", action="store_true", help="One epoch on at most 4 train / 2 validation scenes.")
    export = commands.add_parser("export", help="Predict a manifest with a saved checkpoint.")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--batch-size", type=int, default=1)
    export.add_argument("--checkpoint-name", choices=("best", "last"), default="best")
    for subparser in (train, export):
        subparser.add_argument("--num-workers", type=int, default=0)
        subparser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["device"] = torch.device(args["device"])
    if command == "train":
        train_live(**args)
    else:
        metrics = export_live(**args)
        print(f"Exported {metrics['buildings']:,} buildings ({metrics['split']}); macro-F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
