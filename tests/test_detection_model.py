"""Real Mask R-CNN synthetic CPU forward/backward; no pretrained downloads."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import FrozenBatchNorm2d

from disaster_lens.detection.model import DetectorConfig, build_detector
from disaster_lens.detection.model import identify_detections
from disaster_lens.detection.train import train_epoch


def tiny_config() -> DetectorConfig:
    return DetectorConfig(
        min_size=64, max_size=96, anchor_sizes=(8, 16, 32, 64, 128),
        rpn_pre_nms_top_n=32, rpn_post_nms_top_n=16, detections_per_image=8,
        rpn_batch_size=16, box_batch_size=16, mask_representation="dense",
    )


class InstanceSegmentationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_offline_construction_frozen_norm_and_two_class_predictors(self) -> None:
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("No downloads")):
            model = build_detector(tiny_config())
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)
        self.assertEqual(model.roi_heads.mask_predictor.mask_fcn_logits.out_channels, 2)
        self.assertEqual(model.roi_heads.mask_roi_pool.output_size, (14, 14))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.body.layer1.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.backbone.body.layer2.parameters()))
        self.assertTrue(any(isinstance(m, FrozenBatchNorm2d) for m in model.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))
        self.assertEqual(tiny_config().metadata()["architecture"], "Mask R-CNN ResNet-50-FPN (v1)")
        for value in (0., 1., float("nan")):
            with self.assertRaises(ValueError):
                DetectorConfig(mask_threshold=value)

    def test_mask_loss_updates_and_native_grid_inference_with_empty_scene(self) -> None:
        torch.manual_seed(42)
        config = tiny_config()
        model = build_detector(config)
        mask = torch.zeros(1, 40, 80, dtype=torch.uint8)
        mask[0, 5:20, 10:30] = 1
        data = [
            (torch.rand(3, 40, 80), {"boxes": torch.tensor([[10., 5., 30., 20.]]),
                                    "labels": torch.tensor([1]), "masks": mask}),
            (torch.rand(3, 64, 64), {"boxes": torch.empty(0, 4),
                                    "labels": torch.empty(0, dtype=torch.int64),
                                    "masks": torch.empty(0, 64, 64, dtype=torch.uint8)}),
        ]
        loader = DataLoader(data, batch_size=2, collate_fn=lambda samples: tuple(map(list, zip(*samples))))
        before_mask = model.roi_heads.mask_predictor.mask_fcn_logits.weight.detach().clone()
        before_box = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
        metrics = train_epoch(model, loader, torch.device("cpu"), optimizer)
        self.assertIn("loss_mask", metrics["loss_components"])
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertFalse(torch.equal(before_mask, model.roi_heads.mask_predictor.mask_fcn_logits.weight))
        self.assertFalse(torch.equal(before_box, model.roi_heads.box_predictor.cls_score.weight))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        mask_grads = [p.grad for p in model.roi_heads.mask_head.parameters() if p.grad is not None]
        self.assertTrue(mask_grads)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in mask_grads))
        model.eval()
        with torch.inference_mode():
            outputs = model([row[0] for row in data])
        for prediction, (image, _) in zip(outputs, data, strict=True):
            count = len(prediction["boxes"])
            self.assertEqual(prediction["masks"].shape, (count, 1, image.shape[1], image.shape[2]))
            self.assertTrue(torch.isfinite(prediction["masks"]).all())
            self.assertTrue(((prediction["masks"] >= 0) & (prediction["masks"] <= 1)).all())
            ids = identify_detections(prediction, image_key=f"image-{image.shape[2]}", run_key="synthetic")
            self.assertEqual(len(ids), count)

    def test_strict_local_mask_state_loading_and_faster_rcnn_rejection(self) -> None:
        source = build_detector(tiny_config())
        source.roi_heads.box_predictor = FastRCNNPredictor(1024, 91)
        source.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 91)
        state = source.state_dict()
        with patch("disaster_lens.detection.model.file_sha256", return_value="a" * 64), \
             patch("disaster_lens.detection.model.torch.load", return_value=state):
            loaded = build_detector(tiny_config(), coco_weights=Path("synthetic.pt"), expected_sha256="a" * 64)
        self.assertEqual(loaded.roi_heads.mask_predictor.mask_fcn_logits.out_channels, 2)
        self.assertEqual(loaded.roi_heads.box_predictor.cls_score.out_features, 2)
        torch.testing.assert_close(loaded.roi_heads.mask_head[0][0].weight, source.roi_heads.mask_head[0][0].weight)
        box_only = {key: value for key, value in state.items() if not key.startswith(("roi_heads.mask_head", "roi_heads.mask_predictor"))}
        with patch("disaster_lens.detection.model.file_sha256", return_value="a" * 64), \
             patch("disaster_lens.detection.model.torch.load", return_value=box_only):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                build_detector(tiny_config(), coco_weights=Path("synthetic-box-only.pt"), expected_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "together"):
            build_detector(coco_weights=Path("not-read.pt"))
        with patch("disaster_lens.detection.model.file_sha256", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_detector(coco_weights=Path("not-read.pt"), expected_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
