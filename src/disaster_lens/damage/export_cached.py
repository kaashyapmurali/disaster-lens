"""Export validation predictions from a cached-fusion run's best checkpoint."""

import argparse
import csv
import hashlib
import math
import tempfile
from pathlib import Path

import torch
from filelock import FileLock
from torch.nn import functional as F
from torch.utils.data import DataLoader

from disaster_lens.damage.features import PairedSceneFeatures
from disaster_lens.damage.heads import TemporalFusionHead
from disaster_lens.damage.severity import SEVERITY_COLUMNS, severity_outputs
from disaster_lens.damage.feature_cache import (
    CachedFeatureDataset, _atomic_write, collate_cached_scenes,
)
from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.damage.train_cached import classification_metrics
from disaster_lens.data.scenes import CLASS_TO_INDEX

PROBABILITY_COLUMNS = [f"probability_{label}" for label in DAMAGE_LABELS]
PREDICTION_COLUMNS = [
    "scene_id", "building_uid", "reference_label", "predicted_label", *PROBABILITY_COLUMNS,
]
ORDINAL_PREDICTION_COLUMNS = [*PREDICTION_COLUMNS, *SEVERITY_COLUMNS]


def validate_prediction_rows(
    rows: list[dict[str, str]],
    expected: list[tuple[str, str, str]],
) -> None:
    """Validate serialized rows against independently indexed cache identities."""
    if not expected or len(rows) != len(expected):
        raise ValueError("Prediction row count differs from the validation cache.")
    seen: set[tuple[str, str]] = set()
    for row, identity in zip(rows, expected, strict=True):
        if set(row) not in (set(PREDICTION_COLUMNS), set(ORDINAL_PREDICTION_COLUMNS)):
            raise ValueError("Unexpected prediction columns.")
        key = (row["scene_id"], row["building_uid"])
        if key in seen:
            raise ValueError(f"Duplicate prediction identity: {key}")
        seen.add(key)
        if (*key, row["reference_label"]) != identity:
            raise ValueError(f"Prediction identity/label alignment mismatch: {key}")
        probabilities = [float(row[column]) for column in PROBABILITY_COLUMNS]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError(f"Invalid class probabilities: {key}")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Class probabilities do not sum to one: {key}")
        predicted = DAMAGE_LABELS[max(range(len(probabilities)), key=probabilities.__getitem__)]
        if row["predicted_label"] != predicted:
            raise ValueError(f"Predicted label differs from probability argmax: {key}")
        if "severity_index" in row:
            derived = severity_outputs(torch.tensor([probabilities], dtype=torch.float64))[0].tolist()
            for column, value in zip(SEVERITY_COLUMNS, derived, strict=True):
                supplied = float(row[column])
                if not math.isfinite(supplied) or not math.isclose(supplied, value, rel_tol=0, abs_tol=1e-6):
                    raise ValueError(f"Exported {column} differs from class probabilities: {key}")


def export_validation_predictions(
    run_dir: Path,
    validation_cache: Path,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
    checkpoint_name: str = "best",
) -> dict:
    """Reload best/last checkpoint, then publish a separate, checked export."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Use positive batch_size and nonnegative num_workers.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if checkpoint_name not in ("best", "last"):
        raise ValueError("checkpoint_name must be 'best' or 'last'.")
    checkpoint_path = run_dir / f"{checkpoint_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best fusion checkpoint not found: {checkpoint_path}")
    output_dir = run_dir / (
        "validation_predictions" if checkpoint_name == "best" else "last_validation_predictions"
    )
    with FileLock(str(run_dir / ".prediction_export.lock"), timeout=0):
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir}")
        with checkpoint_path.open("rb") as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        # Old cross-entropy checkpoints predate explicit objective metadata.
        if config.get("objective", "cross-entropy") != "cross-entropy":
            raise ValueError("Only cross-entropy heads are supported.")
        if config["architecture"] != "cached-paired-temporal-fusion":
            raise ValueError("Expected a cached temporal fusion checkpoint, not a post-only model.")
        if config["class_to_index"] != CLASS_TO_INDEX:
            raise ValueError("Checkpoint class mapping differs from the original four classes.")
        dataset = CachedFeatureDataset(validation_cache)
        recorded = config["caches"]["validation"]
        if any(recorded[key] != dataset.metadata[key] for key in ("cache_id", "spec", "scenes")):
            raise ValueError("Validation cache differs from the checkpoint's validation cache.")
        expected: list[tuple[str, str, str]] = []
        seen_uids: set[str] = set()
        for scene in dataset:
            if not scene.building_uids:
                raise ValueError(f"Empty validation scene: {scene.scene_id}")
            for uid, label in zip(scene.building_uids, scene.labels.tolist(), strict=True):
                if not isinstance(uid, str) or not uid or uid in seen_uids:
                    raise ValueError(f"Invalid or duplicate validation building UID: {uid}")
                seen_uids.add(uid)
                expected.append((scene.scene_id, uid, DAMAGE_LABELS[label]))
        if checkpoint["validation"]["buildings"] != len(expected):
            raise ValueError("Validation building count differs from the checkpoint.")
        model_config = config["model"]
        if (model_config["building_dim"] != dataset.building_dim
                or model_config["context_dim"] != dataset.context_dim
                or model_config["num_classes"] != len(DAMAGE_LABELS)):
            raise ValueError("Checkpoint dimensions differ from the validation cache/classes.")
        model = TemporalFusionHead(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_cached_scenes,
        )
        rows: list[dict[str, str]] = []
        total_loss = 0.0
        with torch.inference_mode():
            for batch in loader:
                features = PairedSceneFeatures(**{
                    name: tensor.to(device) for name, tensor in batch.features.as_dict().items()
                })
                if any(not tensor.isfinite().all() for tensor in features.as_dict().values()):
                    raise ValueError(f"Non-finite validation features: {batch.scene_ids}")
                logits = model(features)
                if logits.shape != (len(batch.labels), len(DAMAGE_LABELS)) or not logits.isfinite().all():
                    raise ValueError(f"Invalid validation logits: {batch.scene_ids}")
                labels = batch.labels.to(device)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                if not loss.isfinite():
                    raise ValueError(f"Non-finite validation loss: {batch.scene_ids}")
                total_loss += loss.item()
                probabilities = logits.softmax(dim=1).cpu().tolist()
                severity = severity_outputs(torch.tensor(probabilities, dtype=torch.float64)).tolist()
                scene_indices = batch.features.building_to_scene.tolist()
                for index, (uid, label, values) in enumerate(zip(
                    batch.building_uids, batch.labels.tolist(), probabilities, strict=True,
                )):
                    predicted = max(range(len(values)), key=values.__getitem__)
                    rows.append({
                        "scene_id": batch.scene_ids[scene_indices[index]], "building_uid": uid,
                        "reference_label": DAMAGE_LABELS[label],
                        "predicted_label": DAMAGE_LABELS[predicted],
                        **{column: repr(value) for column, value in zip(
                            PROBABILITY_COLUMNS, values, strict=True,
                        )},
                        **{
                            column: repr(value) for column, value in zip(
                                SEVERITY_COLUMNS,
                                severity[index],
                                strict=True,
                            )
                        },
                    })
        validate_prediction_rows(rows, expected)
        metrics = {
            **classification_metrics(
                [CLASS_TO_INDEX[row["reference_label"]] for row in rows],
                [CLASS_TO_INDEX[row["predicted_label"]] for row in rows],
            ),
            "loss": total_loss / len(rows),
            "objective": "cross-entropy",
            "severity_index_definition": "expected category rank / 3; equal-boundary ordinal index, not percent damage",
            "selection_metric": config.get("selection_metric", "validation.macro_f1"),
            "split": "validation", "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256, "epoch": checkpoint["epoch"],
            "cache_id": dataset.metadata["cache_id"], "class_to_index": CLASS_TO_INDEX,
            "confusion_matrix_convention": "rows=reference, columns=predicted; class_to_index order",
            "model": model_config, "device": str(device), "torch_version": str(torch.__version__),
        }
        # Publish no final directory until the serialized CSV also passes validation.
        with tempfile.TemporaryDirectory(dir=run_dir, prefix=".validation_predictions-") as temporary:
            staging = Path(temporary) / "export"
            staging.mkdir()
            csv_path = staging / "predictions.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=ORDINAL_PREDICTION_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                validate_prediction_rows(list(csv.DictReader(stream)), expected)
            _atomic_write(staging / "metrics.json", metrics)
            staging.rename(output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Scenes per batch.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-name", choices=("best", "last"), default="best",
                        help="Use last for a fixed-budget comparison; exports into a separate directory.")
    args = vars(parser.parse_args())
    args["device"] = torch.device(args["device"])
    metrics = export_validation_predictions(**args)
    print(f"Exported {metrics['buildings']:,} validation buildings from epoch {metrics['epoch']}; "
          f"macro-F1={metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
