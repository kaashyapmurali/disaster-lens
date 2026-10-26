"""Scene-backbone contract and the ResNet-18 adapter (the DINOv3 adapter lives in dinov3.py)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18



@dataclass(frozen=True)
class SceneFeatures:
    """Spatial building features [B, C, H, W] and scene context [B, D]."""

    spatial_map: Tensor
    context: Tensor


class SceneBackbone(nn.Module, ABC):
    """Encode normalized images without imposing freezing or caching policy.

    Adapters declare their spatial channels, context dimension, and isotropic
    image-to-feature coordinate scale. A future ViT adapter must reshape spatial
    patch tokens into a grid, excluding CLS/register tokens, and provide its
    chosen scene-context representation. Input normalization/resizing belongs
    to the data pipeline and must match the chosen checkpoint.
    """

    spatial_channels: int
    context_dim: int
    spatial_scale: float

    @abstractmethod
    def forward(self, images: Tensor) -> SceneFeatures:
        """Return spatial features and context from one scene encoding."""
        raise NotImplementedError

    def cache_config(self) -> dict:
        """Additional feature-producing configuration not captured by tensors."""
        return {}

    @abstractmethod
    def trainable_units(self) -> list[nn.Module]:
        """Encoder blocks in input-to-output order; partial fine-tuning unfreezes a suffix of this list."""
        raise NotImplementedError

    def suffix_extras(self) -> list[nn.Module]:
        """Modules unfrozen together with any trainable suffix (e.g. a final norm)."""
        return []

    def batchnorm_frozen(self) -> bool:
        """Whether normalization statistics stay fixed even for trainable blocks (batch size 1 scenes)."""
        return False


class ResNetSceneEncoder(nn.Module):
    """One scene pass yields stride-8 local maps and a 512-D global vector.

    Local ROIs use layer2 rather than the final stride-32 map to retain more
    spatial detail for small buildings. This is a baseline, not an FPN detector.
    """

    local_channels = 128
    global_channels = 512
    spatial_scale = 1 / 8

    def __init__(self, weights: ResNet18_Weights | None) -> None:
        super().__init__()
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1,
        )
        self.local_layers = backbone.layer2
        self.global_layers = nn.Sequential(backbone.layer3, backbone.layer4)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        local_map = self.local_layers(self.stem(images))
        global_map = self.global_layers(local_map)
        context = global_map.mean(dim=(-2, -1))
        return local_map, context


class ResNet18Backbone(SceneBackbone):
    """Layer2 stride-8 grid and globally averaged layer4 context.

    Module names (encoder.stem / local_layers / global_layers) are part of saved checkpoints and feature-cache
    identities; keep them unchanged. Trainable units are layer2, layer3 and layer4; BatchNorm statistics stay
    frozen because training uses one scene per batch.
    """

    spatial_channels = ResNetSceneEncoder.local_channels
    context_dim = ResNetSceneEncoder.global_channels
    spatial_scale = ResNetSceneEncoder.spatial_scale

    def __init__(self, weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT) -> None:
        super().__init__()
        self.encoder = ResNetSceneEncoder(weights)

    def forward(self, images: Tensor) -> SceneFeatures:
        spatial, context = self.encoder(images)
        return SceneFeatures(spatial, context)

    def trainable_units(self) -> list[nn.Module]:
        return [self.encoder.local_layers, *self.encoder.global_layers]

    def batchnorm_frozen(self) -> bool:
        return True
