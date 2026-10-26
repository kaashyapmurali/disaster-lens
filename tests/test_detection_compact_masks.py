"""Compact/dense equivalence on small CPU references; no GPU or downloads."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.roi_heads import paste_masks_in_image, project_masks_on_boxes

from disaster_lens.detection.masks import CompactMasks, materialize_mask_window
from disaster_lens.detection.model import DetectorConfig, build_detector
from disaster_lens.data.footprints import MaskWindow, rasterize_footprint_window
from disaster_lens.detection.train import train_epoch


def compact_from_dense(dense: torch.Tensor) -> CompactMasks:
    windows = []
    for mask in dense:
        ys, xs = torch.where(mask > 0)
        if not len(xs):
            windows.append(MaskWindow((0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)))
        else:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            windows.append(MaskWindow((x1, y1, x2, y2), mask[y1:y2, x1:x2].numpy().copy()))
    return CompactMasks(tuple(dense.shape[-2:]), tuple(windows))


def tiny_config() -> DetectorConfig:
    return DetectorConfig(
        min_size=32, max_size=64, anchor_sizes=(8, 16, 32, 64, 128),
        rpn_pre_nms_top_n=16, rpn_post_nms_top_n=8, detections_per_image=4,
        rpn_batch_size=8, box_batch_size=8,
    )


class CompactMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def masks(self) -> torch.Tensor:
        masks = torch.zeros(3, 19, 23, dtype=torch.uint8)
        masks[0, :8, :7] = 1  # Image edges exercise clamping, not artificial zero padding.
        masks[1, 4:18, 6:22] = 1
        masks[1, 7:12, 10:16] = 0  # Hole.
        masks[2, 18, 22] = 1  # Small image-edge instance.
        return masks

    def test_compact_storage_resize_matches_dense_nearest_exactly(self) -> None:
        dense = self.masks()
        compact = compact_from_dense(dense)
        self.assertLess(compact.stored_bytes, dense.numel())
        torch.testing.assert_close(compact.dense_reference(), dense)
        for size in ((19, 23), (7, 9), (32, 41), (1, 1), (20, 17)):
            expected = F.interpolate(dense[:, None].float(), size=size, mode="nearest")[:, 0].byte()
            resized = compact.resized(size)
            self.assertEqual(len(resized), len(compact))
            torch.testing.assert_close(resized.dense_reference(), expected, rtol=0, atol=0)

    def test_roi_projection_matches_dense_sampling_at_edges_holes_and_fractional_boxes(self) -> None:
        dense = self.masks()
        compact = compact_from_dense(dense)
        boxes = torch.tensor([[0., 0., 23., 19.], [1.3, 2.7, 18.8, 17.2],
                              [6., 4., 22., 18.], [19., 14., 23., 19.],
                              [0., 0., 4., 4.], [21.5, 17.5, 23., 19.]])
        indices = torch.tensor([0, 1, 1, 0, 1, 2])
        for size in (7, 14, 28):
            expected = project_masks_on_boxes(dense, boxes, indices, size)
            actual = compact.project(boxes, indices, size)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        empty = compact.project(torch.empty(0, 4), torch.empty(0, dtype=torch.int64), 28)
        self.assertEqual(empty.shape, (0, 28, 28))

    def test_single_prediction_window_matches_dense_paste(self) -> None:
        torch.manual_seed(7)
        probability = torch.rand(1, 1, 28, 28)
        for box in ([0., 0., 7., 8.], [3.2, 2.4, 17.3, 16.9], [20., 16., 23., 19.], [0., 0., 23., 19.]):
            tensor_box = torch.tensor(box)
            expected = paste_masks_in_image(probability, tensor_box[None], (19, 23))[0, 0]
            window = materialize_mask_window(probability[0], tensor_box, (19, 23))
            x1, y1, x2, y2 = window.bounds
            actual = torch.zeros(19, 23)
            actual[y1:y2, x1:x2] = window.probabilities
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(window.binary().dtype, np.uint8)

    def test_dense_scene_windows_do_not_allocate_dense_reference(self) -> None:
        window = rasterize_footprint_window("POLYGON ((10 10,14 10,14 14,10 14,10 10))", 1024, 1024)
        compact = CompactMasks((1024, 1024), tuple(window for _ in range(2000)))
        self.assertEqual(compact.stored_bytes, 2000 * 16)
        with self.assertRaises(MemoryError):
            compact.dense_reference()
        self.assertEqual(compact.resized((768, 768)).image_size, (768, 768))

    def test_compact_and_dense_model_losses_gradients_and_predictions_agree(self) -> None:
        torch.manual_seed(42)
        compact_model = build_detector(tiny_config())
        dense_model = build_detector(replace(tiny_config(), mask_representation="dense"))
        dense_model.load_state_dict(compact_model.state_dict(), strict=True)
        image = torch.rand(3, 19, 23)
        masks = self.masks()[:2]
        boxes = torch.tensor([[0., 0., 7., 8.], [6., 4., 22., 18.]])
        dense_target = {"boxes": boxes, "labels": torch.ones(2, dtype=torch.int64), "masks": masks}
        compact_target = {"boxes": boxes, "labels": torch.ones(2, dtype=torch.int64),
                          "compact_masks": compact_from_dense(masks)}
        dense_model.train()
        compact_model.train()
        torch.manual_seed(99)
        dense_losses = dense_model([image], [dense_target])
        torch.manual_seed(99)
        with patch.object(CompactMasks, "dense_reference", side_effect=AssertionError("No dense target stacks")):
            compact_losses = compact_model([image], [compact_target])
        self.assertEqual(set(compact_losses), set(dense_losses))
        for name in dense_losses:
            torch.testing.assert_close(compact_losses[name], dense_losses[name], rtol=1e-5, atol=2e-6)
        sum(dense_losses.values()).backward()
        sum(compact_losses.values()).backward()
        torch.testing.assert_close(compact_model.roi_heads.mask_head[0][0].weight.grad,
                                   dense_model.roi_heads.mask_head[0][0].weight.grad, rtol=1e-4, atol=1e-5)
        dense_model.eval()
        compact_model.eval()
        with torch.inference_mode():
            dense = dense_model([image])[0]
            compact = compact_model([image])[0]
        self.assertNotIn("masks", compact)
        self.assertEqual(compact["mask_image_size"], (19, 23))
        self.assertEqual(compact["mask_roi_probs"].shape, (len(compact["boxes"]), 1, 28, 28))
        torch.testing.assert_close(compact["boxes"], dense["boxes"])
        expected_masks = paste_masks_in_image(compact["mask_roi_probs"], compact["boxes"], (19, 23))
        torch.testing.assert_close(expected_masks, dense["masks"], rtol=1e-5, atol=2e-6)

    def test_empty_detections_keep_compact_output_contract(self) -> None:
        model = build_detector(replace(tiny_config(), score_threshold=1.))
        model.eval()
        with torch.inference_mode():
            prediction = model([torch.rand(3, 19, 23)])[0]
        self.assertEqual(prediction["mask_roi_probs"].shape, (0, 1, 28, 28))
        self.assertEqual(prediction["mask_image_size"], (19, 23))
        self.assertNotIn("masks", prediction)

    def test_invalid_projection_indices_and_version_changes_are_rejected(self) -> None:
        masks = compact_from_dense(self.masks())
        box = torch.tensor([[0., 0., 7., 8.]])
        with self.assertRaises(ValueError):
            masks.project(box, torch.tensor([999]), 28)
        with self.assertRaises(ValueError):
            masks.project(box, torch.tensor([.5]), 28)
        with patch("disaster_lens.detection.compact_rcnn.torchvision.__version__", "999.0"):
            with self.assertRaisesRegex(RuntimeError, "review before changing"):
                build_detector(tiny_config())

    def test_compact_epoch_supports_empty_scene_and_updates_mask_head(self) -> None:
        model = build_detector(tiny_config())
        image = torch.rand(3, 19, 23)
        masks = self.masks()[:1]
        data = [(image, {"boxes": torch.tensor([[0., 0., 7., 8.]]), "labels": torch.tensor([1]),
                         "compact_masks": compact_from_dense(masks)}),
                (image, {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64),
                         "compact_masks": CompactMasks((19, 23), ())})]
        loader = DataLoader(data, batch_size=2, collate_fn=lambda samples: tuple(map(list, zip(*samples))))
        before = model.roi_heads.mask_predictor.mask_fcn_logits.weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
        with patch.object(CompactMasks, "dense_reference", side_effect=AssertionError("No dense stacks")):
            metrics = train_epoch(model, loader, torch.device("cpu"), optimizer)
        self.assertIn("loss_mask", metrics["loss_components"])
        self.assertFalse(torch.equal(before, model.roi_heads.mask_predictor.mask_fcn_logits.weight))


if __name__ == "__main__":
    unittest.main()
