"""Backbone-independent pre/post region extraction and raw feature containers."""

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align

from disaster_lens.damage.backbones import SceneBackbone, SceneFeatures


@dataclass(frozen=True)
class PairedSceneFeatures:
    """Raw pre/post features, before trainable projections and subtraction.

    N = buildings, B = scene pairs. Context is stored once per scene/date.
    Keep corresponding scene/building UIDs, labels, and cache configuration
    alongside these tensors when persisting frozen features.
    """

    pre_building: Tensor       # [N, spatial_channels * roi_size**2]
    post_building: Tensor      # [N, spatial_channels * roi_size**2]
    pre_context: Tensor        # [B, context_dim]
    post_context: Tensor       # [B, context_dim]
    building_to_scene: Tensor  # [N], int64 indices into the B context vectors

    def as_dict(self) -> dict[str, Tensor]:
        """Tensor-only payload, suitable for torch.save / weights_only loading."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


class PairedFeatureExtractor(nn.Module):
    """Use one shared SceneBackbone to extract both dates' features.

    Frozen mode fixes weights and evaluation behavior and disables gradients.
    Fine-tunable mode preserves autograd and follows normal train/eval behavior.
    Only frozen, fixed-preprocessing features may be cached across training steps.

    Inputs are checkpoint-normalized images and boxes in each input image's
    pixel coordinates. ROI lists must match building UIDs AND order between
    dates; counts alone cannot verify identity. ROIAlign uses rectangles, not
    exact polygon masks. Reusing pre_boxes assumes aligned image grids.
    """

    def __init__(
        self,
        backbone: SceneBackbone,
        roi_size: int = 3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, SceneBackbone):
            raise TypeError("backbone must implement SceneBackbone.")
        if roi_size < 1:
            raise ValueError("roi_size must be positive.")
        if backbone.spatial_channels < 1 or backbone.context_dim < 1 or backbone.spatial_scale <= 0:
            raise ValueError("Backbone dimensions and spatial_scale must be positive.")
        self.backbone = backbone
        self.roi_size = roi_size
        self.freeze_backbone = freeze_backbone
        self.backbone.requires_grad_(not freeze_backbone)
        self.train()

    @property
    def building_dim(self) -> int:
        return self.backbone.spatial_channels * self.roi_size**2

    @property
    def context_dim(self) -> int:
        return self.backbone.context_dim

    def train(self, mode: bool = True) -> "PairedFeatureExtractor":
        # A frozen extractor cannot silently change BatchNorm statistics.
        super().train(False if self.freeze_backbone else mode)
        return self

    @staticmethod
    def _validate_inputs(images: Tensor, boxes: list[Tensor], phase: str) -> None:
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
            raise ValueError(f"{phase}: expected nonempty [B, 3, H, W] images.")
        if not images.is_floating_point() or len(boxes) != images.shape[0]:
            raise ValueError(f"{phase}: use floating images and one box tensor per scene.")
        height, width = images.shape[-2:]
        for box in boxes:
            if box.ndim != 2 or box.shape[1] != 4:
                raise ValueError(f"{phase}: each box tensor must have shape [N_i, 4].")
            if not torch.isfinite(box).all():
                raise ValueError(f"{phase}: boxes must be finite.")
            if (
                (box[:, :2] < 0).any()
                or (box[:, 2] > width).any()
                or (box[:, 3] > height).any()
                or ((box[:, 2:] - box[:, :2]) <= 0).any()
            ):
                raise ValueError(f"{phase}: boxes must have positive area inside the image.")

    def encode_maps(self, images: Tensor) -> SceneFeatures:
        """Run the shared backbone once; pool any number of box sets from the result with `pool`."""
        output = self.backbone(images)
        if not isinstance(output, SceneFeatures):
            raise TypeError("SceneBackbone.forward must return SceneFeatures.")
        spatial, context = output.spatial_map, output.context
        if spatial.ndim != 4 or spatial.shape[:2] != (len(images), self.backbone.spatial_channels):
            raise ValueError("Backbone spatial_map must have shape [B, spatial_channels, H, W].")
        if context.shape != (len(images), self.context_dim):
            raise ValueError("Backbone context must have shape [B, context_dim].")
        if spatial.device != images.device or context.device != images.device:
            raise ValueError("Backbone outputs must stay on the input device.")
        return output

    def pool(self, spatial: Tensor, boxes: list[Tensor]) -> Tensor:
        """ROIAlign building features from an encoded spatial map (float32, autocast disabled)."""
        # Native ROIAlign does not support BF16 on all backends. Casting here
        # preserves gradients into the low-precision encoder feature map.
        roi_map = spatial.float() if spatial.dtype in (torch.bfloat16, torch.float16) else spatial
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            return roi_align(
                roi_map,
                [box.to(device=roi_map.device, dtype=roi_map.dtype) for box in boxes],
                output_size=self.roi_size,
                spatial_scale=self.backbone.spatial_scale,
                sampling_ratio=2,
                aligned=True,
            ).flatten(start_dim=1)

    def _encode_date(self, images: Tensor, boxes: list[Tensor]) -> tuple[Tensor, Tensor]:
        output = self.encode_maps(images)
        return self.pool(output.spatial_map, boxes), output.context

    def forward(
        self,
        pre_images: Tensor,
        post_images: Tensor,
        pre_boxes: list[Tensor],
        post_boxes: list[Tensor] | None = None,
    ) -> PairedSceneFeatures:
        post_boxes = pre_boxes if post_boxes is None else post_boxes
        self._validate_inputs(pre_images, pre_boxes, "pre")
        self._validate_inputs(post_images, post_boxes, "post")
        if pre_images.shape[0] != post_images.shape[0]:
            raise ValueError("Pre/post inputs must contain the same number of scenes.")
        if pre_images.device != post_images.device:
            raise ValueError("Pre/post images must be on the same device.")
        if [len(box) for box in pre_boxes] != [len(box) for box in post_boxes]:
            raise ValueError("Pre/post scenes must have matching building counts and UID order.")
        # Respect an outer no_grad/inference context; do not re-enable gradients.
        enabled = torch.is_grad_enabled() and not self.freeze_backbone
        with torch.set_grad_enabled(enabled):
            pre_building, pre_context = self._encode_date(pre_images, pre_boxes)
            post_building, post_context = self._encode_date(post_images, post_boxes)
        counts = torch.tensor([len(box) for box in pre_boxes], device=pre_images.device)
        scene_indices = torch.arange(pre_images.shape[0], device=pre_images.device).repeat_interleave(counts)
        return PairedSceneFeatures(
            pre_building, post_building, pre_context, post_context, scene_indices,
        )
