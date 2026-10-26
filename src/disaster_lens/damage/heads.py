"""Classification heads: temporal fusion for paired dates and a linear head for the post-only baseline.

Use raw cached features with a frozen backbone, or differentiable live features
when fine-tuning. Projector/fusion outputs must not be cached during training.
"""

import torch
from torch import Tensor, nn

from disaster_lens.damage.features import PairedSceneFeatures


class TemporalFusionHead(nn.Module):
    """Project raw features to a common width, then fuse appearance and changes.

    Default shared building projector: 1152 -> 128 -> LayerNorm -> GELU.
    Default shared context projector:   512 -> 128 -> LayerNorm -> GELU.
    Fusion input: [post building, post context, building delta, context delta].
    Fusion MLP: 512 -> 128 -> GELU -> Dropout(0.1) -> 4 logits.

    Supply building_dim/context_dim from PairedFeatureExtractor for other
    backbones or ROI sizes. Deltas are signed post-minus-pre differences AFTER
    projection. Context projection runs once per scene/date, not per building.
    """

    def __init__(
        self,
        building_dim: int = 1152,
        context_dim: int = 512,
        projected_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(building_dim, context_dim, projected_dim, hidden_dim) < 1 or num_classes < 2:
            raise ValueError("Feature dimensions must be positive and num_classes must be >= 2.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.building_dim = building_dim
        self.context_dim = context_dim
        self.projected_dim = projected_dim
        self.building_projector = nn.Sequential(
            nn.Linear(building_dim, projected_dim), nn.LayerNorm(projected_dim), nn.GELU(),
        )
        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, projected_dim), nn.LayerNorm(projected_dim), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(4 * projected_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _validate_features(self, features: PairedSceneFeatures) -> None:
        tensors = features.as_dict()
        pre_b, post_b = features.pre_building, features.post_building
        pre_c, post_c = features.pre_context, features.post_context
        if pre_b.ndim != 2 or pre_b.shape != post_b.shape or pre_b.shape[1] != self.building_dim:
            raise ValueError("Pre/post building features must have matching [N, building_dim] shapes.")
        if (
            pre_c.ndim != 2 or pre_c.shape != post_c.shape
            or pre_c.shape[1] != self.context_dim or pre_c.shape[0] < 1
        ):
            raise ValueError("Pre/post context features must have matching [B, context_dim] shapes.")
        indices = features.building_to_scene
        if indices.ndim != 1 or len(indices) != len(pre_b) or indices.dtype != torch.long:
            raise ValueError("building_to_scene must be an int64 vector with one index per building.")
        if len({tensor.device for tensor in tensors.values()}) != 1:
            raise ValueError("All feature tensors must be on the same device.")
        if indices.numel() and ((indices < 0).any() or (indices >= len(pre_c)).any()):
            raise ValueError("building_to_scene contains an out-of-range scene index.")

    def fusion_features(self, features: PairedSceneFeatures) -> Tensor:
        """Return the explicit [post B, post C, delta B, delta C] representation."""
        self._validate_features(features)
        pre_b = self.building_projector(features.pre_building)
        post_b = self.building_projector(features.post_building)
        pre_c = self.context_projector(features.pre_context)
        post_c = self.context_projector(features.post_context)
        indices = features.building_to_scene
        return torch.cat(
            (post_b, post_c[indices], post_b - pre_b, (post_c - pre_c)[indices]), dim=1,
        )

    def forward(self, features: PairedSceneFeatures) -> Tensor:
        return self.fusion(self.fusion_features(features))


class PostOnlyHead(nn.Module):
    """Multinomial logistic regression on [post building ROI features, post scene context]."""

    def __init__(self, building_dim: int = 1152, context_dim: int = 512, num_classes: int = 4) -> None:
        super().__init__()
        if min(building_dim, context_dim) < 1 or num_classes < 2:
            raise ValueError("Feature dimensions must be positive and num_classes must be >= 2.")
        self.building_dim, self.context_dim = building_dim, context_dim
        self.linear = nn.Linear(building_dim + context_dim, num_classes)

    def forward(self, features: PairedSceneFeatures) -> Tensor:
        context = features.post_context[features.building_to_scene]
        return self.linear(torch.cat((features.post_building, context), dim=1))
