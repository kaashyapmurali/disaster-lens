"""Synthetic CPU checks only: no datasets, pretrained downloads, or remote jobs."""

import io
import unittest
from unittest.mock import patch

import torch
from torch import nn

from disaster_lens.damage.backbones import ResNet18Backbone, SceneBackbone, SceneFeatures
from disaster_lens.damage.features import PairedFeatureExtractor, PairedSceneFeatures
from disaster_lens.damage.heads import TemporalFusionHead


class TinyPatchGridBackbone(SceneBackbone):
    def trainable_units(self):
        return []

    """Differentiable patch-grid test double, not an actual ViT implementation."""

    spatial_channels = 6
    context_dim = 4
    spatial_scale = 1 / 4

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, self.spatial_channels, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(self.spatial_channels)
        self.context_projection = nn.Linear(self.spatial_channels, self.context_dim)

    def forward(self, images: torch.Tensor) -> SceneFeatures:
        spatial = self.norm(self.patch_embed(images))
        context = self.context_projection(spatial.mean(dim=(-2, -1)))
        return SceneFeatures(spatial, context)


def example_features() -> PairedSceneFeatures:
    return PairedSceneFeatures(
        pre_building=torch.randn(3, 1152), post_building=torch.randn(3, 1152),
        pre_context=torch.randn(2, 512), post_context=torch.randn(2, 512),
        building_to_scene=torch.tensor([0, 0, 1]),
    )


class TemporalFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_encoder_runs_once_per_date_and_context_is_not_duplicated(self) -> None:
        extractor = PairedFeatureExtractor(ResNet18Backbone(weights=None))
        extractor.train()
        pre, post = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
        boxes = [torch.tensor([[0., 0., 20., 20.], [30., 30., 60., 60.]]),
                 torch.tensor([[10., 10., 40., 40.]])]
        head = TemporalFusionHead().eval()
        bn_before = extractor.backbone.encoder.stem[1].running_mean.clone()
        with patch.object(extractor.backbone, "forward", wraps=extractor.backbone.forward) as encode:
            features = extractor(pre, post, boxes)
            head(features)
            head(features)  # Reuse raw features without another encoder call.
        self.assertEqual(encode.call_count, 2)
        self.assertEqual(tuple(features.pre_building.shape), (3, 1152))
        self.assertEqual(tuple(features.post_building.shape), (3, 1152))
        self.assertEqual(tuple(features.pre_context.shape), (2, 512))
        self.assertEqual(tuple(features.post_context.shape), (2, 512))
        self.assertEqual(features.building_to_scene.tolist(), [0, 0, 1])
        self.assertFalse(extractor.training)
        self.assertFalse(extractor.backbone.training)
        self.assertTrue(all(not value.requires_grad for value in features.as_dict().values()))
        self.assertTrue(all(not p.requires_grad for p in extractor.parameters()))
        torch.testing.assert_close(bn_before, extractor.backbone.encoder.stem[1].running_mean)

    def test_identical_dates_have_identical_raw_features(self) -> None:
        extractor = PairedFeatureExtractor(ResNet18Backbone(weights=None))
        images = torch.randn(1, 3, 64, 64)
        boxes = [torch.tensor([[4., 8., 32., 48.]])]
        features = extractor(images, images, boxes)
        torch.testing.assert_close(features.pre_building, features.post_building)
        torch.testing.assert_close(features.pre_context, features.post_context)

    def test_feature_layout_preserves_post_absolutes_and_signed_differences(self) -> None:
        head = TemporalFusionHead(building_dim=2, context_dim=2, projected_dim=2)
        # Identity projectors make temporal arithmetic directly inspectable.
        head.building_projector = nn.Identity()
        head.context_projector = nn.Identity()
        features = PairedSceneFeatures(
            pre_building=torch.tensor([[1., 4.], [2., 3.], [10., 20.]]),
            post_building=torch.tensor([[3., 2.], [1., 5.], [15., 18.]]),
            pre_context=torch.tensor([[1., 2.], [8., 4.]]),
            post_context=torch.tensor([[4., 1.], [6., 7.]]),
            building_to_scene=torch.tensor([0, 0, 1]),
        )
        expected = torch.tensor([
            [3., 2., 4., 1., 2., -2., 3., -1.],
            [1., 5., 4., 1., -1., 2., 3., -1.],
            [15., 18., 6., 7., 5., -2., -2., 3.],
        ])
        torch.testing.assert_close(head.fusion_features(features), expected)

    def test_shared_projectors_receive_scene_context_once_per_date(self) -> None:
        features = example_features()
        head = TemporalFusionHead()
        with patch.object(head.building_projector, "forward", wraps=head.building_projector.forward) as buildings, \
             patch.object(head.context_projector, "forward", wraps=head.context_projector.forward) as contexts:
            logits = head(features)
        self.assertEqual(tuple(logits.shape), (3, 4))
        self.assertEqual(buildings.call_count, 2)
        self.assertEqual(contexts.call_count, 2)
        self.assertTrue(all(call.args[0].shape == (2, 512) for call in contexts.call_args_list))

    def test_projectors_and_fusion_receive_gradients_from_cached_tensors(self) -> None:
        head = TemporalFusionHead()
        features = example_features()
        loss = nn.functional.cross_entropy(head(features), torch.tensor([0, 1, 3]))
        loss.backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(all(value.grad is None for value in features.as_dict().values()))

    def test_tensor_payload_roundtrip_and_head_checkpoint_reload(self) -> None:
        features = example_features()
        head = TemporalFusionHead().eval()
        expected = head(features).detach()
        buffer = io.BytesIO()
        torch.save({"features": features.as_dict(), "head": head.state_dict()}, buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True)
        restored = TemporalFusionHead().eval()
        restored.load_state_dict(payload["head"], strict=True)
        torch.testing.assert_close(restored(PairedSceneFeatures(**payload["features"])), expected)

    def test_empty_building_batch(self) -> None:
        features = PairedSceneFeatures(
            pre_building=torch.empty(0, 1152), post_building=torch.empty(0, 1152),
            pre_context=torch.randn(1, 512), post_context=torch.randn(1, 512),
            building_to_scene=torch.empty(0, dtype=torch.long),
        )
        self.assertEqual(tuple(TemporalFusionHead()(features).shape), (0, 4))

    def test_rejects_mismatched_rois_and_invalid_boxes(self) -> None:
        extractor = PairedFeatureExtractor(ResNet18Backbone(weights=None))
        images = torch.randn(1, 3, 64, 64)
        boxes = [torch.tensor([[0., 0., 20., 20.]])]
        with self.assertRaisesRegex(ValueError, "matching building counts"):
            extractor(images, images, boxes, [torch.empty(0, 4)])
        with self.assertRaisesRegex(ValueError, "positive area"):
            extractor(images, images, [torch.tensor([[0., 0., 100., 20.]])])

    def test_rejects_invalid_scene_index_and_feature_dimensions(self) -> None:
        features = example_features()
        payload = features.as_dict()
        payload["building_to_scene"] = torch.tensor([0, 0, 2])
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            TemporalFusionHead()(PairedSceneFeatures(**payload))
        payload = features.as_dict()
        payload["pre_building"] = torch.randn(3, 128)
        with self.assertRaisesRegex(ValueError, "building features"):
            TemporalFusionHead()(PairedSceneFeatures(**payload))

    def test_alternative_backbone_dimensions_keep_fusion_width_fixed(self) -> None:
        extractor = PairedFeatureExtractor(TinyPatchGridBackbone(), roi_size=2)
        head = TemporalFusionHead(
            building_dim=extractor.building_dim, context_dim=extractor.context_dim,
        )
        images = torch.randn(1, 3, 32, 32)
        boxes = [torch.tensor([[0., 0., 16., 16.], [12., 12., 28., 28.]])]
        features = extractor(images, images, boxes)
        self.assertEqual(tuple(features.pre_building.shape), (2, 24))
        self.assertEqual(tuple(features.pre_context.shape), (1, 4))
        self.assertEqual(tuple(head.fusion_features(features).shape), (2, 512))
        self.assertEqual(tuple(head(features).shape), (2, 4))

    def test_finetuning_backpropagates_through_shared_backbone_and_both_dates(self) -> None:
        backbone = TinyPatchGridBackbone()
        extractor = PairedFeatureExtractor(backbone, freeze_backbone=False)
        head = TemporalFusionHead(
            building_dim=extractor.building_dim, context_dim=extractor.context_dim, dropout=0,
        )
        pre = torch.randn(1, 3, 32, 32, requires_grad=True)
        post = torch.randn(1, 3, 32, 32, requires_grad=True)
        boxes = [torch.tensor([[4., 4., 20., 20.]])]
        before = backbone.norm.running_mean.clone()
        features = extractor(pre, post, boxes)
        loss = nn.functional.cross_entropy(head(features), torch.tensor([2]))
        loss.backward()
        self.assertTrue(extractor.training)
        self.assertTrue(backbone.training)
        self.assertFalse(torch.equal(before, backbone.norm.running_mean))
        self.assertGreater(pre.grad.abs().sum().item(), 0)
        self.assertGreater(post.grad.abs().sum().item(), 0)
        for parameter in backbone.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        extractor.eval()
        self.assertFalse(backbone.training)
        with torch.no_grad():
            inference_features = extractor(pre, post, boxes)
        self.assertTrue(all(not value.requires_grad for value in inference_features.as_dict().values()))

    def test_backbone_contract_rejects_unreshaped_tokens(self) -> None:
        class BadGridBackbone(TinyPatchGridBackbone):
            def forward(self, images: torch.Tensor) -> SceneFeatures:
                output = super().forward(images)
                return SceneFeatures(output.spatial_map.flatten(start_dim=2), output.context)
        extractor = PairedFeatureExtractor(BadGridBackbone())
        images = torch.randn(1, 3, 32, 32)
        boxes = [torch.tensor([[0., 0., 16., 16.]])]
        with self.assertRaisesRegex(ValueError, "spatial_map"):
            extractor(images, images, boxes)

    def test_scene_backbone_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            SceneBackbone()

    def test_rejects_invalid_head_configuration(self) -> None:
        for settings in ({"projected_dim": 0}, {"dropout": 1.0}, {"num_classes": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TemporalFusionHead(**settings)


if __name__ == "__main__":
    unittest.main()
