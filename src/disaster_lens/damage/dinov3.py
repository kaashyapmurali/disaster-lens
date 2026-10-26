"""Native timm DINOv3 patch-grid adapter; weights must already exist locally."""

from pathlib import Path

import timm
from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3
from timm.models import load_checkpoint
from timm.models.eva import Eva
from torch import Tensor, nn

from disaster_lens.damage.backbones import SceneBackbone, SceneFeatures

TIMM_MODEL_NAME = "vit_large_patch16_dinov3.sat493m"
SATELLITE_MEAN = (0.430, 0.411, 0.296)
SATELLITE_STD = (0.213, 0.156, 0.143)


class DINOv3Backbone(SceneBackbone):
    """Expose normalized final patch tokens and their global spatial average.

    CLS/register tokens participate in attention but are excluded from the ROI
    grid and context average. Inputs must already use the checkpoint's image
    normalization. No resizing, cropping, or normalization happens here.

    ViT-L/16 supplies 1024 channels at stride 16: a 3x3 ROI has 9216 features,
    while each scene context has 1024. Freezing belongs to PairedFeatureExtractor.
    """

    def __init__(self, encoder: Eva) -> None:
        super().__init__()
        if not isinstance(encoder, Eva) or not isinstance(encoder.rope, RotaryEmbeddingDinoV3):
            raise TypeError("encoder must be a timm DINOv3 Eva model with DINOv3 RoPE.")
        patch = encoder.patch_embed.patch_size
        if len(patch) != 2 or patch[0] != patch[1] or patch[0] < 1:
            raise ValueError("DINOv3Backbone requires positive square patches for isotropic ROIAlign.")
        if encoder.in_chans != 3:
            raise ValueError("DINOv3Backbone requires three-channel RGB inputs.")
        if not isinstance(encoder.norm, nn.LayerNorm):
            raise ValueError("DINOv3Backbone requires normalized final patch tokens before pooling.")
        self.encoder = encoder
        self.patch_size = patch[0]
        self.spatial_channels = encoder.num_features
        self.context_dim = encoder.num_features
        self.spatial_scale = 1 / self.patch_size
        self.prefix_tokens = encoder.num_prefix_tokens

    @classmethod
    def from_local_checkpoint(cls, checkpoint_dir: Path) -> "DINOv3Backbone":
        """Strictly load local timm model.safetensors; no Hub downloads or conversion."""
        checkpoint = checkpoint_dir / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local timm DINOv3 checkpoint not found: {checkpoint}")
        encoder = timm.create_model(
            TIMM_MODEL_NAME, pretrained=False, num_classes=0, dynamic_img_size=True,
        )
        load_checkpoint(encoder, str(checkpoint), strict=True)
        return cls(encoder)

    @classmethod
    def untrained(cls, **constructor) -> "DINOv3Backbone":
        """Architecture only (weights restored from a saved checkpoint); defaults match the pretrained model."""
        settings = {"num_classes": 0, "dynamic_img_size": True, **constructor}
        return cls(timm.create_model(TIMM_MODEL_NAME, pretrained=False, **settings))

    def constructor(self) -> dict:
        """Arguments that rebuild this architecture with `untrained` (stored in checkpoints)."""
        first = self.encoder.blocks[0]
        return {"embed_dim": self.spatial_channels, "depth": len(self.encoder.blocks), "num_heads": first.attn.num_heads,
                "mlp_ratio": first.mlp.fc1.out_features / first.mlp.fc1.in_features, "num_classes": 0,
                "dynamic_img_size": True}

    def cache_config(self) -> dict:
        return {
            "timm_version": timm.__version__,
            "model_name": TIMM_MODEL_NAME,
            "model_config": {
                "embed_dim": self.spatial_channels, "depth": len(self.encoder.blocks),
                "patch_size": self.patch_size, "prefix_tokens": self.prefix_tokens,
                "dynamic_img_size": self.encoder.dynamic_img_size,
                "norm_eps": self.encoder.norm.eps,
                "attention_heads": [block.attn.num_heads for block in self.encoder.blocks],
                "attention_bias_parameters": [
                    name for name, _ in self.encoder.named_parameters()
                    if ".attn." in name and name.endswith(("q_bias", "k_bias", "v_bias", "qkv.bias"))
                ],
            },
            "context_pooling": "mean of final normalized spatial patch tokens; excludes CLS/registers",
            "fused_attention": [block.attn.fused_attn for block in self.encoder.blocks],
            "rope": {
                "implementation": type(self.encoder.rope).__name__,
                "periods_dtype": str(self.encoder.rope.periods.dtype),
                "temperature": self.encoder.rope.temperature,
                "min_period": self.encoder.rope.min_period,
                "max_period": self.encoder.rope.max_period,
                "normalize_coords": self.encoder.rope.normalize_coords,
                "rotate_half": self.encoder.rope.rotate_half,
                "grid_offset": self.encoder.rope.grid_offset,
                "grid_indexing": self.encoder.rope.grid_indexing,
                "shift_coords": self.encoder.rope.shift_coords,
                "jitter_coords": self.encoder.rope.jitter_coords,
                "rescale_coords": self.encoder.rope.rescale_coords,
            },
        }

    def forward(self, images: Tensor) -> SceneFeatures:
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
            raise ValueError("Expected nonempty RGB images with shape [B, 3, H, W].")
        if not images.is_floating_point():
            raise ValueError("DINOv3 inputs must be normalized floating-point images.")
        batch_size, _, height, width = images.shape
        if min(height, width) < self.patch_size or height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be positive multiples of the patch size.")
        grid_height, grid_width = height // self.patch_size, width // self.patch_size
        tokens = self.encoder.forward_features(images)
        expected = (batch_size, self.prefix_tokens + grid_height * grid_width, self.spatial_channels)
        if tokens.shape != expected:
            raise ValueError(f"DINOv3 output token layout differs from expected {expected}.")
        patches = tokens[:, self.prefix_tokens:, :]
        spatial = patches.transpose(1, 2).reshape(
            batch_size, self.spatial_channels, grid_height, grid_width,
        ).contiguous()
        return SceneFeatures(spatial, spatial.mean(dim=(-2, -1)))

    def trainable_units(self) -> list[nn.Module]:
        return list(self.encoder.blocks)

    def suffix_extras(self) -> list[nn.Module]:
        return [self.encoder.norm]
