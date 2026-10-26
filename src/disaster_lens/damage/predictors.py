"""Load any trained damage classifier for inference on arbitrary building boxes.

Two checkpoint families:
* live checkpoints from `damage.train_live` (any encoder × inputs × trainable blocks), including the earlier formats
  of the reported fine-tuned ViT and post-only baseline;
* cached-feature fusion heads from `damage.train_cached`, rebuilt on the frozen encoder after checking that the live
  encoder weights, preprocessing and ROI recipe equal those of the cached features the head was trained on.

`reference_check` describes what the annotated-footprint predictions must reproduce: the run's saved per-building
predictions when supplied, otherwise the checkpoint's recorded validation confusion matrix.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from disaster_lens.damage.classifier import DamageClassifier, build_encoder
from disaster_lens.damage.features import PairedFeatureExtractor
from disaster_lens.damage.heads import TemporalFusionHead
from disaster_lens.damage.train_live import SPEC_KEYS, _dataset, _file_digest, load_live_model
from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.data.scenes import CLASS_TO_INDEX, PairedSceneDataset

CACHED_HEAD = "cached-paired-temporal-fusion"


class Predictor:
    """Encode each date once per scene, then classify any number of box sets ("arms")."""

    def __init__(self, model: DamageClassifier, precision: str) -> None:
        self.model, self.precision = model.eval(), precision

    @torch.inference_mode()
    def predict(self, pre_image: Tensor, post_image: Tensor, arms: dict[str, tuple[Tensor, Tensor]],
                device: torch.device) -> dict[str, Tensor]:
        extractor, paired = self.model.extractor, self.model.inputs == "paired"
        results = {}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            post_maps = extractor.encode_maps(post_image)
            pre_maps = extractor.encode_maps(pre_image) if paired else None
            for name, (pre_boxes, post_boxes) in arms.items():
                if len(pre_boxes) != len(post_boxes):
                    raise ValueError(f"{name}: PRE/POST box counts differ.")
                if len(post_boxes) == 0:
                    results[name] = torch.empty(0, len(DAMAGE_LABELS))
                    continue
                PairedFeatureExtractor._validate_inputs(post_image, [post_boxes], "post")
                if paired:
                    PairedFeatureExtractor._validate_inputs(pre_image, [pre_boxes], "pre")
                logits = self.model.classify(
                    extractor.pool(pre_maps.spatial_map, [pre_boxes]) if paired else None,
                    extractor.pool(post_maps.spatial_map, [post_boxes]),
                    pre_maps.context if paired else None, post_maps.context,
                    torch.zeros(len(post_boxes), dtype=torch.long, device=post_image.device))
                if not logits.isfinite().all():
                    raise ValueError(f"{name}: non-finite logits.")
                results[name] = logits.float().softmax(dim=1).cpu()
        return results


@dataclass
class LoadedDamageModel:
    predictor: Predictor
    dataset: PairedSceneDataset
    reference_check: dict          # {"saved_predictions": Path} or {"confusion_matrix": [[...]], "macro_f1": float}
    provenance: dict


def _strict_float32() -> None:
    """Cached features were written in float32 with 'highest' matmul precision and no TF32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _cached_head_model(config: dict, state: dict, manifest: Path, data_dir: Path, device: torch.device,
                       encoder_dir: Path | None) -> tuple[DamageClassifier, PairedSceneDataset]:
    from disaster_lens.damage.feature_cache import cache_spec

    if config.get("objective", "cross-entropy") != "cross-entropy":
        raise ValueError("Only cross-entropy heads are supported.")
    recorded = config["caches"]["validation"]["spec"]
    encoder = {"disaster_lens.damage.backbones.ResNet18Backbone": "resnet18",
               "disaster_lens.models.backbones.ResNet18Backbone": "resnet18",
               "disaster_lens.damage.dinov3.DINOv3Backbone": "dinov3-vitl",
               "disaster_lens.models.dinov3.DINOv3Backbone": "dinov3-vitl"}.get(recorded["backbone"])
    if encoder is None:
        raise ValueError(f"Unknown cached-feature encoder {recorded['backbone']}.")
    _strict_float32()
    preprocessing = recorded["preprocessing"]
    dataset = PairedSceneDataset(manifest, data_dir, preprocessing["image_size"],
                                 mean=tuple(preprocessing["mean"]), std=tuple(preprocessing["std"]))
    backbone = build_encoder(encoder, weights_dir=encoder_dir)
    frozen = PairedFeatureExtractor(backbone, roi_size=recorded["roi"]["size"], freeze_backbone=True)
    live = cache_spec(dataset, frozen)
    keys = [key for key in SPEC_KEYS if key not in ("backbone", "manifest_sha256", "building_index_sha256")]
    if dataset.manifest_sha256 == recorded["manifest_sha256"]:
        keys += ["manifest_sha256", "building_index_sha256"]     # same split: the index must match too
    differing = [key for key in keys if live.get(key) != recorded.get(key)]
    if differing:
        raise ValueError(f"Live feature recipe differs from the recorded validation cache: {differing}")
    head = TemporalFusionHead(**config["model"])
    head.load_state_dict(state, strict=True)
    model = DamageClassifier(backbone, head, inputs="paired", trainable_blocks=0, roi_size=recorded["roi"]["size"])
    return model.to(device).eval(), dataset


def load_damage_model(checkpoint_path: Path, manifest: Path, data_dir: Path, device: torch.device, *,
                      encoder_dir: Path | None = None, saved_predictions: Path | None = None) -> LoadedDamageModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("smoke_test", False) or config["class_to_index"] != CLASS_TO_INDEX:
        raise ValueError("Use a full (non-smoke) checkpoint with the original four-class mapping.")
    if config["architecture"] == CACHED_HEAD:
        model, dataset = _cached_head_model(config, checkpoint["model_state_dict"], manifest, data_dir, device, encoder_dir)
        precision, encoder = "float32", ("dinov3-vitl" if model.backbone.spatial_channels == 1024 else "resnet18")
    else:
        if encoder_dir is not None:
            raise ValueError("--encoder-dir is only used with cached-feature heads.")
        model, recipe = load_live_model(checkpoint, device)
        if recipe["precision"] == "float32":
            _strict_float32()
        dataset = _dataset(manifest, data_dir, recipe["image_size"], recipe["encoder"])
        recorded = (recipe["data_specs"] or {}).get("validation")
        if recorded is not None and dataset.preprocessing_config() != recorded["preprocessing"]:
            raise ValueError("Image preprocessing differs from the training run.")
        precision, encoder = recipe["precision"], recipe["encoder"]
    validation = checkpoint["validation"]
    check = ({"saved_predictions": saved_predictions} if saved_predictions is not None
             else {"confusion_matrix": validation["confusion_matrix"], "macro_f1": validation["macro_f1"]})
    provenance = {"damage_checkpoint_sha256": _file_digest(checkpoint_path), "damage_checkpoint_epoch": checkpoint["epoch"],
                  "damage_architecture": config["architecture"], "encoder": encoder, "inputs": model.inputs,
                  "trainable_blocks": model.trainable_blocks, "precision": precision}
    return LoadedDamageModel(Predictor(model, precision), dataset, check, provenance)
