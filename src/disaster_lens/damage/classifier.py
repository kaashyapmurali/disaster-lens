"""Damage classifiers built from three choices: encoder, inputs, and how many encoder blocks are trained.

    encoder           resnet18 (ImageNet, stride-8 ROI features) | dinov3-vitl (satellite-pretrained, stride 16)
    inputs            post   (post-event image only, linear head)
                      paired (pre and post through the same encoder, temporal fusion head)
    trainable_blocks  0 = frozen encoder (only the head trains); N = also train the last N encoder blocks

The same encoder weights process both dates. Features are ROIAlign-pooled over each footprint's box.
"""

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from disaster_lens.damage.backbones import ResNet18Backbone, SceneBackbone
from disaster_lens.damage.features import PairedFeatureExtractor, PairedSceneFeatures
from disaster_lens.damage.heads import PostOnlyHead, TemporalFusionHead
from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.data.scenes import IMAGENET_MEAN, IMAGENET_STD

ENCODERS = ("resnet18", "dinov3-vitl")
INPUTS = ("post", "paired")


def normalization(encoder: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """RGB mean/std expected by each encoder's pretrained weights."""
    if encoder == "resnet18":
        return IMAGENET_MEAN, IMAGENET_STD
    if encoder == "dinov3-vitl":
        from disaster_lens.damage.dinov3 import SATELLITE_MEAN, SATELLITE_STD
        return SATELLITE_MEAN, SATELLITE_STD
    raise ValueError(f"Unknown encoder {encoder!r}; choose from {ENCODERS}.")


def build_encoder(encoder: str, *, weights_dir: Path | None = None, pretrained: bool = True) -> SceneBackbone:
    """ResNet-18 uses torchvision's cached ImageNet weights; DINOv3 loads a local timm checkpoint."""
    if encoder == "resnet18":
        from torchvision.models import ResNet18_Weights
        if weights_dir is not None:
            raise ValueError("resnet18 weights come from torchvision; --encoder-dir is for dinov3-vitl.")
        return ResNet18Backbone(ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if encoder == "dinov3-vitl":
        from disaster_lens.damage.dinov3 import DINOv3Backbone
        if pretrained:
            if weights_dir is None:
                raise ValueError("dinov3-vitl needs --encoder-dir with the local timm checkpoint.")
            return DINOv3Backbone.from_local_checkpoint(weights_dir)
        return DINOv3Backbone.untrained()
    raise ValueError(f"Unknown encoder {encoder!r}; choose from {ENCODERS}.")


def build_head(inputs: str, building_dim: int, context_dim: int, **fusion: int | float) -> nn.Module:
    if inputs == "post":
        return PostOnlyHead(building_dim, context_dim, num_classes=len(DAMAGE_LABELS))
    if inputs == "paired":
        return TemporalFusionHead(building_dim, context_dim, num_classes=len(DAMAGE_LABELS), **fusion)
    raise ValueError(f"Unknown inputs {inputs!r}; choose from {INPUTS}.")


class DamageClassifier(nn.Module):
    """Encoder + ROI pooling + head. Frozen encoder parts always run in eval mode.

    Attribute names (`extractor`, `head`) are part of saved checkpoints; keep them unchanged.
    """

    def __init__(self, backbone: SceneBackbone, head: nn.Module, *, inputs: str, trainable_blocks: int = 0,
                 roi_size: int = 3) -> None:
        super().__init__()
        if inputs not in INPUTS:
            raise ValueError(f"Unknown inputs {inputs!r}; choose from {INPUTS}.")
        units = backbone.trainable_units()
        if type(trainable_blocks) is not int or not 0 <= trainable_blocks <= len(units):
            raise ValueError(f"trainable_blocks must be between 0 and {len(units)} for this encoder.")
        if head.building_dim != backbone.spatial_channels * roi_size**2 or head.context_dim != backbone.context_dim:
            raise ValueError("Head dimensions differ from the encoder/ROI configuration.")
        self.extractor = PairedFeatureExtractor(backbone, roi_size=roi_size, freeze_backbone=trainable_blocks == 0)
        self.head = head
        self.inputs, self.trainable_blocks = inputs, trainable_blocks
        backbone.requires_grad_(False)
        if trainable_blocks:
            for module in [*units[-trainable_blocks:], *backbone.suffix_extras()]:
                module.requires_grad_(True)
        self.head.requires_grad_(True)
        self.train()

    @property
    def backbone(self) -> SceneBackbone:
        return self.extractor.backbone

    def _trainable_modules(self) -> list[nn.Module]:
        if not self.trainable_blocks:
            return []
        return [*self.backbone.trainable_units()[-self.trainable_blocks:], *self.backbone.suffix_extras()]

    def train(self, mode: bool = True) -> "DamageClassifier":
        super().train(mode)
        self.backbone.eval()
        for module in self._trainable_modules():
            module.train(mode)
            if self.backbone.batchnorm_frozen():
                for child in module.modules():
                    if isinstance(child, nn.modules.batchnorm._BatchNorm):
                        child.eval()
        return self

    def parameter_groups(self, *, backbone_lr: float, head_lr: float) -> list[dict]:
        """Disjoint optimizer groups; the encoder group is omitted when the encoder is frozen."""
        if not all(math.isfinite(value) and value > 0 for value in (backbone_lr, head_lr)):
            raise ValueError("Backbone and head learning rates must be finite and positive.")
        groups = [{"name": "head", "params": list(self.head.parameters()), "lr": head_lr}]
        encoder = [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        if encoder:
            groups.insert(0, {"name": "encoder_suffix", "params": encoder, "lr": backbone_lr})
        return groups

    def classify(self, pre_building: Tensor | None, post_building: Tensor, pre_context: Tensor | None,
                 post_context: Tensor, building_to_scene: Tensor) -> Tensor:
        """Logits from pooled features (pre-event features are ignored by the post-only head)."""
        if self.inputs == "paired" and (pre_building is None or pre_context is None):
            raise ValueError("The paired model needs pre-event features.")
        features = PairedSceneFeatures(
            post_building if pre_building is None else pre_building, post_building,
            post_context if pre_context is None else pre_context, post_context, building_to_scene)
        return self.head(features)

    def forward(self, pre_images: Tensor, post_images: Tensor, boxes: list[Tensor]) -> Tensor:
        """Footprint boxes are shared by both dates (pre-event outlines reused on the post image)."""
        if self.inputs == "paired":
            return self.head(self.extractor(pre_images, post_images, boxes))
        PairedFeatureExtractor._validate_inputs(post_images, boxes, "post")
        maps = self.extractor.encode_maps(post_images)
        counts = torch.tensor([len(b) for b in boxes], device=post_images.device)
        index = torch.arange(len(boxes), device=post_images.device).repeat_interleave(counts)
        return self.classify(None, self.extractor.pool(maps.spatial_map, boxes), None, maps.context, index)
