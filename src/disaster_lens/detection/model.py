"""Single-class Mask R-CNN (ResNet-50-FPN) for pre-event building footprints; offline construction only.

This predicts building instances, not damage. COCO_V1 weights are loaded from a local file whose
SHA-256 is verified; without weights the model is randomly initialized (synthetic tests only).
"""

import math
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
import torchvision
from torch import Tensor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import FrozenBatchNorm2d

from disaster_lens.data.building_index import file_sha256
from disaster_lens.detection.compact_rcnn import install_compact_masks


@dataclass(frozen=True)
class DetectorConfig:
    """Single-class building Mask R-CNN recipe.

    1024-square inputs keep their resolution (tiles are upsampled to it by the model's resize).
    Anchors start at 16 px because buildings are small (median about 31 px). Proposal and
    detection limits are per image. Compact masks keep 28 × 28 ROI probabilities plus native
    boxes instead of N × H × W stacks; dense mode is a small-image reference path for tests.
    """

    min_size: int = 1024
    max_size: int = 1024
    anchor_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    trainable_backbone_layers: int = 3  # layer2..4; stem/layer1 frozen
    rpn_pre_nms_top_n: int = 4000
    rpn_post_nms_top_n: int = 3000
    detections_per_image: int = 2500
    score_threshold: float = 0.05
    nms_threshold: float = 0.5
    rpn_batch_size: int = 256
    box_batch_size: int = 512
    mask_representation: str = "compact"
    mask_threshold: float = .5

    def __post_init__(self) -> None:
        counts = (self.min_size, self.max_size, self.rpn_pre_nms_top_n,
                  self.rpn_post_nms_top_n, self.detections_per_image,
                  self.rpn_batch_size, self.box_batch_size)
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("Image sizes, proposal limits, and sample counts must be positive integers.")
        if self.min_size > self.max_size:
            raise ValueError("min_size must not exceed max_size.")
        if (len(self.anchor_sizes) != 5
                or any(type(size) is not int or size <= 0 for size in self.anchor_sizes)):
            raise ValueError("Specify five positive FPN anchor sizes.")
        if (not self.aspect_ratios
                or any(not math.isfinite(ratio) or ratio <= 0 for ratio in self.aspect_ratios)):
            raise ValueError("Anchor aspect ratios must be positive and finite.")
        if (type(self.trainable_backbone_layers) is not int
                or not 0 <= self.trainable_backbone_layers <= 5):
            raise ValueError("trainable_backbone_layers must be between 0 and 5.")
        if not 0 <= self.score_threshold <= 1 or not 0 < self.nms_threshold <= 1:
            raise ValueError("Invalid detection score/NMS threshold.")
        if self.detections_per_image > self.rpn_post_nms_top_n:
            raise ValueError("Detection limit cannot exceed the single-class proposal limit.")
        if self.mask_representation not in ("compact", "dense"):
            raise ValueError("Choose compact or dense mask representation.")
        if not math.isfinite(self.mask_threshold) or not 0 < self.mask_threshold < 1:
            raise ValueError("Mask threshold must be finite and strictly between 0 and 1.")

    def metadata(self) -> dict:
        return {
            "architecture": "Mask R-CNN ResNet-50-FPN (v1)",
            "torchvision_version": torchvision.__version__,
            "config": asdict(self), "classes": {"background": 0, "building": 1},
            "normalization": "detector-owned ImageNet RGB; input float32 [0,1]",
            "backbone_norm": "FrozenBatchNorm2d, eps=0 (COCO_V1 recipe)",
            "coordinates": "native input xyxy on returned predictions",
            "mask_roi_size": 14, "mask_logit_size": 28,
            "mask_output": ("mask_roi_probs [N,1,28,28] + native boxes + mask_image_size"
                            if self.mask_representation == "compact" else "masks [N,1,H,W] native-grid probabilities"),
            "mask_threshold_note": "conventional binary-mask threshold, not calibrated confidence",
        }


def build_detector(
    config: DetectorConfig | None = None, *,
    coco_weights: Path | None = None, expected_sha256: str | None = None,
) -> MaskRCNN:
    """No downloads. Optional local weights must be Mask R-CNN COCO_V1.

    Faster R-CNN, v2, and backbone-only checkpoints are not interchangeable.
    Load the original 91-class detector+mask state strictly, then replace both
    class-specific predictors. Random weights are for local synthetic tests.
    """
    config = config or DetectorConfig()
    if (coco_weights is None) != (expected_sha256 is None):
        raise ValueError("Supply local Mask R-CNN weights and their SHA-256 together.")
    if coco_weights is not None and file_sha256(coco_weights) != expected_sha256:
        raise ValueError("Local Mask R-CNN weights SHA-256 mismatch.")
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None,
        norm_layer=partial(FrozenBatchNorm2d, eps=0.),
        trainable_layers=config.trainable_backbone_layers,
    )
    anchors = AnchorGenerator(sizes=tuple((size,) for size in config.anchor_sizes),
                              aspect_ratios=(config.aspect_ratios,) * 5)
    model = MaskRCNN(
        backbone, num_classes=91 if coco_weights is not None else 2,
        min_size=config.min_size, max_size=config.max_size, rpn_anchor_generator=anchors,
        rpn_pre_nms_top_n_train=config.rpn_pre_nms_top_n,
        rpn_pre_nms_top_n_test=config.rpn_pre_nms_top_n,
        rpn_post_nms_top_n_train=config.rpn_post_nms_top_n,
        rpn_post_nms_top_n_test=config.rpn_post_nms_top_n,
        box_detections_per_img=config.detections_per_image,
        box_score_thresh=config.score_threshold, box_nms_thresh=config.nms_threshold,
        rpn_batch_size_per_image=config.rpn_batch_size,
        box_batch_size_per_image=config.box_batch_size,
    )
    if coco_weights is not None:
        state = torch.load(coco_weights, map_location="cpu", weights_only=True)
        if file_sha256(coco_weights) != expected_sha256:
            raise ValueError("Mask R-CNN weights changed during loading.")
        model.load_state_dict(state, strict=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        mask_predictor = model.roi_heads.mask_predictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            mask_predictor.conv5_mask.in_channels, mask_predictor.conv5_mask.out_channels, 2,
        )
    if config.mask_representation == "compact":
        install_compact_masks(model)
    return model


def identify_detections(prediction: dict[str, Tensor], *, image_key: str, run_key: str) -> list[dict]:
    """Assign IDs to final detector outputs without benchmark annotations.

    Keys must uniquely identify this image and prediction run. IDs are stable
    only within this saved prediction set, NOT tracking IDs across reruns/dates.
    Alignment/location transfer must carry these IDs rather than regenerate them.
    """
    if not image_key or not run_key:
        raise ValueError("Nonempty image_key and run_key are required.")
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    count = len(boxes)
    if boxes.shape != (count, 4) or scores.shape != (count,) or labels.shape != (count,):
        raise ValueError("Misaligned detection boxes/scores/labels.")
    if (not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()
            or (boxes[:, 2:] <= boxes[:, :2]).any() or (boxes < 0).any()
            or (scores < 0).any() or (scores > 1).any() or (labels != 1).any()):
        raise ValueError("Invalid building detections.")
    # Length-prefix namespaces so delimiters inside caller keys cannot collide.
    namespace = f"{len(run_key)}:{run_key}:{len(image_key)}:{image_key}"
    return [
        {"building_id": f"{namespace}:{index:06d}",
         "box": box.tolist(), "score": float(score)}
        for index, (box, score) in enumerate(zip(boxes, scores, strict=True))
    ]
