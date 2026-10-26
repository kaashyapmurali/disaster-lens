"""Train only the temporal fusion head from completed frozen-feature caches."""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.nn import functional as F
from torch.utils.data import DataLoader

from disaster_lens.damage.features import PairedSceneFeatures
from disaster_lens.damage.heads import TemporalFusionHead
from disaster_lens.damage.feature_cache import (
    CachedFeatureDataset, _atomic_write, collate_cached_scenes,
)
from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.data.scenes import CLASS_TO_INDEX


def validate_caches(train: CachedFeatureDataset, validation: CachedFeatureDataset) -> torch.Tensor:
    """Check extraction compatibility and identities before constructing a head."""
    split_specific = {"manifest_sha256", "building_index_sha256", "image_provenance_sha256"}
    recipes = [
        {key: value for key, value in data.metadata["spec"].items() if key not in split_specific}
        for data in (train, validation)
    ]
    if recipes[0] != recipes[1]:
        raise ValueError("Train/validation cache extraction recipes differ.")
    scene_ids = [{entry["scene_id"] for entry in data.metadata["scenes"]}
                 for data in (train, validation)]
    if scene_ids[0] & scene_ids[1]:
        raise ValueError("Training and validation share scene IDs.")
    identities: list[set[str]] = []
    counts = torch.zeros(len(DAMAGE_LABELS), dtype=torch.long)
    for split_index, data in enumerate((train, validation)):
        uids: set[str] = set()
        for scene in data:
            if not scene.building_uids:
                raise ValueError(f"Empty cached scene: {scene.scene_id}")
            if any(not isinstance(uid, str) or not uid for uid in scene.building_uids):
                raise ValueError(f"Invalid building UID in {scene.scene_id}")
            if uids.intersection(scene.building_uids):
                raise ValueError("Duplicate building UIDs within a cache.")
            uids.update(scene.building_uids)
            if any(not tensor.isfinite().all() for tensor in scene.features.as_dict().values()):
                raise ValueError(f"Non-finite cached features in {scene.scene_id}")
            if split_index == 0:
                counts += torch.bincount(scene.labels, minlength=len(DAMAGE_LABELS))
        identities.append(uids)
    if identities[0] & identities[1]:
        raise ValueError("Training and validation share building UIDs.")
    if (counts == 0).any():
        raise ValueError("Training cache must contain all four classes.")
    return counts


def classification_metrics(targets: list[int], predictions: list[int]) -> dict:
    """Report all four original classes, including zero-support classes."""
    if not targets or len(targets) != len(predictions):
        raise ValueError("Expected nonempty, aligned targets and predictions.")
    indices = list(range(len(DAMAGE_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=indices, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1.mean()),
        "buildings": len(targets),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(DAMAGE_LABELS)
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=indices).tolist(),
    }


def run_epoch(
    model: TemporalFusionHead,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """Optimize projections/fusion only; cross-entropy loss and metrics are per building."""
    training = optimizer is not None
    model.train(training)
    targets: list[int] = []
    predictions: list[int] = []
    batch_counts: list[int] = []
    total_loss = 0.0
    started = time.perf_counter()
    for batch in loader:
        features = PairedSceneFeatures(**{
            name: tensor.to(device) for name, tensor in batch.features.as_dict().items()
        })
        labels = batch.labels.to(device)
        batch_counts.append(len(labels))
        with torch.set_grad_enabled(training):
            logits = model(features)
            losses = F.cross_entropy(logits, labels, weight=class_weights if training else None, reduction="none")
            if not torch.isfinite(logits).all() or not torch.isfinite(losses).all():
                raise ValueError(f"Non-finite logits/loss in scenes {batch.scene_ids}")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses.mean().backward()
                optimizer.step()
        total_loss += losses.detach().sum().item()
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(targets, predictions)
    elapsed = time.perf_counter() - started
    return {**metrics, "loss": total_loss / len(targets),
            "elapsed_seconds": elapsed, "buildings_per_second": len(targets) / elapsed,
            "optimizer_steps": len(batch_counts) if training else 0,
            "buildings_per_batch": {
                "min": min(batch_counts), "max": max(batch_counts),
                "mean": len(targets) / len(batch_counts),
            }}


def train_cached_fusion(
    train_cache: Path,
    validation_cache: Path,
    output_dir: Path,
    *,
    epochs: int = 5,
    batch_size: int = 8,
    validation_batch_size: int | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weight: str = "none",
    num_workers: int = 0,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    projected_dim: int = 128,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> list[dict]:
    """Create a new run; never resume training or overwrite an existing run."""
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("Use positive epochs/batch_size and nonnegative num_workers.")
    validation_batch_size = batch_size if validation_batch_size is None else validation_batch_size
    if validation_batch_size < 1:
        raise ValueError("validation_batch_size must be positive.")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Use finite positive lr and finite nonnegative weight_decay.")
    if class_weight not in ("none", "balanced"):
        raise ValueError("class_weight must be 'none' or 'balanced'.")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32).")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    train_data = CachedFeatureDataset(train_cache)
    val_data = CachedFeatureDataset(validation_cache)
    counts = validate_caches(train_data, val_data)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_config = {
        "building_dim": train_data.building_dim, "context_dim": train_data.context_dim,
        "projected_dim": projected_dim, "hidden_dim": hidden_dim,
        "num_classes": len(DAMAGE_LABELS), "dropout": dropout,
    }
    model = TemporalFusionHead(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = None
    if class_weight == "balanced":
        weights = (counts.sum() / (len(DAMAGE_LABELS) * counts.float())).to(device)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=num_workers,
        collate_fn=collate_cached_scenes,
    )
    val_loader = DataLoader(
        val_data, batch_size=validation_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_cached_scenes,
    )
    config = {
        "architecture": "cached-paired-temporal-fusion", "model": model_config,
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "num_workers": num_workers,
        "batch_unit": "scenes", "validation_batch_size": validation_batch_size,
        "training_loss_reduction": "mean over buildings in each scene batch",
        "device": str(device), "class_to_index": CLASS_TO_INDEX,
        "optimizer": {"name": "AdamW", **optimizer.defaults},
        "class_weight": class_weight,
        "class_weights": weights.cpu().tolist() if weights is not None else None,
        "train_class_counts": dict(zip(DAMAGE_LABELS, counts.tolist())),
        "objective": "cross-entropy",
        "selection_metric": "validation.macro_f1",
        "selection_direction": "max",
        "selection_tie_break": "earliest epoch",
        "torch_version": str(torch.__version__), "python_version": sys.version.split()[0],
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "caches": {
            split: {"path": str(data.cache_dir.resolve()), "cache_id": data.metadata["cache_id"],
                    "spec": data.metadata["spec"], "scenes": data.metadata["scenes"]}
            for split, data in (("train", train_data), ("validation", val_data))
        },
    }
    output_dir.mkdir(parents=True)
    _atomic_write(output_dir / "config.json", config)
    history: list[dict] = []
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, weights)
        val_metrics = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        checkpoint = {
            "epoch": epoch, "config": config, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "validation": val_metrics,
        }
        _atomic_write(output_dir / "last.pt", checkpoint, tensors=True)
        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            _atomic_write(output_dir / "best.pt", checkpoint, tensors=True)
        _atomic_write(output_dir / "history.json", {"epochs": history})
        print(f"Epoch {epoch}: train loss={train_metrics['loss']:.4f}; "
              f"val loss={val_metrics['loss']:.4f}; val macro-F1={val_metrics['macro_f1']:.4f}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8, help="Training scenes per batch.")
    parser.add_argument("--validation-batch-size", type=int, default=None,
                        help="Validation scenes per batch; defaults to the training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = vars(parser.parse_args())
    args["device"] = torch.device(args["device"])
    train_cached_fusion(**args)


if __name__ == "__main__":
    main()
