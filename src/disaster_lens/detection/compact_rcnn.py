"""Eager-only Mask R-CNN adapters for compact target and prediction masks.

Box/mask forward flow is adapted from torchvision 0.29.1 roi_heads.py,
Copyright (c) Soumith Chintala 2016, under BSD-3-Clause. Retained license:
LICENSES/torchvision-BSD-3-Clause.txt. No global monkey-patching is used.
State-dict parameter names/shapes remain compatible with the v1 model.
"""

import torch
import torchvision
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_inference
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from disaster_lens.detection.masks import CompactMasks


class CompactMaskTransform(GeneralizedRCNNTransform):
    def resize(self, image: Tensor, target: dict | None = None) -> tuple[Tensor, dict | None]:
        if target is not None:
            if "masks" in target or not isinstance(target.get("compact_masks"), CompactMasks):
                raise ValueError("Compact Mask R-CNN requires compact_masks, not dense masks.")
            if target["compact_masks"].image_size != tuple(image.shape[-2:]):
                raise ValueError("Compact target masks and original image grid differ.")
        image, target = super().resize(image, target)
        if target is not None:
            target["compact_masks"] = target["compact_masks"].resized(tuple(image.shape[-2:]))
        return image, target

    def postprocess(self, result: list[dict], image_shapes: list[tuple], original_image_sizes: list[tuple]) -> list[dict]:
        if any("masks" in prediction for prediction in result):
            raise ValueError("Compact ROI head must not emit a dense/native masks field.")
        # The base class resizes boxes, but only pastes masks for the 'masks' key.
        result = super().postprocess(result, image_shapes, original_image_sizes)
        if not self.training:
            for prediction, size in zip(result, original_image_sizes, strict=True):
                prediction["mask_image_size"] = tuple(size)
        return result


class CompactMaskRoIHeads(RoIHeads):
    @classmethod
    def from_heads(cls, heads: RoIHeads) -> "CompactMaskRoIHeads":
        if heads.has_keypoint() or not heads.has_mask():
            raise ValueError("Adapter requires mask heads and no keypoint heads.")
        return cls(
            heads.box_roi_pool, heads.box_head, heads.box_predictor,
            heads.proposal_matcher.high_threshold, heads.proposal_matcher.low_threshold,
            heads.fg_bg_sampler.batch_size_per_image, heads.fg_bg_sampler.positive_fraction,
            heads.box_coder.weights, heads.score_thresh, heads.nms_thresh, heads.detections_per_img,
            heads.mask_roi_pool, heads.mask_head, heads.mask_predictor,
        )

    def check_targets(self, targets: list[dict] | None) -> None:
        if targets is None:
            raise ValueError("Training targets required.")
        for target in targets:
            masks = target.get("compact_masks")
            if (not isinstance(masks, CompactMasks) or "masks" in target
                    or "boxes" not in target or "labels" not in target
                    or len(masks) != len(target["boxes"]) or len(masks) != len(target["labels"])):
                raise ValueError("Aligned compact_masks, boxes, and labels required.")
            if target["boxes"].dtype not in (torch.float32, torch.float64) or target["labels"].dtype != torch.int64:
                raise TypeError("Floating boxes and int64 labels required.")

    def forward(self, features: dict[str, Tensor], proposals: list[Tensor], image_shapes: list[tuple],
                targets: list[dict] | None = None) -> tuple[list[dict], dict[str, Tensor]]:
        if self.training:
            proposals, matched, labels, regression = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        result = []
        losses = {}
        if self.training:
            classification, box_loss = fastrcnn_loss(class_logits, box_regression, labels, regression)
            losses.update(loss_classifier=classification, loss_box_reg=box_loss)
            mask_proposals, mask_matches = [], []
            for proposal, matched_ids, label in zip(proposals, matched, labels, strict=True):
                positive = torch.where(label > 0)[0]
                mask_proposals.append(proposal[positive])
                mask_matches.append(matched_ids[positive])
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            result = [{"boxes": box, "scores": score, "labels": label}
                      for box, score, label in zip(boxes, scores, labels, strict=True)]
            mask_proposals = boxes
        mask_features = self.mask_head(self.mask_roi_pool(features, mask_proposals, image_shapes))
        mask_logits = self.mask_predictor(mask_features)
        if self.training:
            size = mask_logits.shape[-1]
            projected = [target["compact_masks"].project(boxes, indices, size)
                         for target, boxes, indices in zip(targets, mask_proposals, mask_matches, strict=True)]
            mask_targets = torch.cat(projected, dim=0)
            mask_labels = torch.cat([target["labels"][indices] for target, indices in zip(targets, mask_matches, strict=True)])
            if mask_targets.numel() == 0:
                loss_mask = mask_logits.sum() * 0
            else:
                selected = mask_logits[torch.arange(len(mask_labels), device=mask_labels.device), mask_labels]
                loss_mask = F.binary_cross_entropy_with_logits(selected, mask_targets)
            losses["loss_mask"] = loss_mask
        else:
            probabilities = maskrcnn_inference(mask_logits, [prediction["labels"] for prediction in result])
            for prediction, probability in zip(result, probabilities, strict=True):
                prediction["mask_roi_probs"] = probability
        return result, losses


def install_compact_masks(model: torch.nn.Module) -> None:
    """Transplant modules without changing state keys; no TorchScript/ONNX claim."""
    if torchvision.__version__.split("+")[0] != "0.29.1":
        raise RuntimeError("Compact mask adapters are verified against torchvision 0.29.1; review before changing versions.")
    model.roi_heads = CompactMaskRoIHeads.from_heads(model.roi_heads)
    original = model.transform
    model.transform = CompactMaskTransform(
        original.min_size, original.max_size, original.image_mean, original.image_std,
        size_divisible=original.size_divisible, fixed_size=original.fixed_size,
    )
