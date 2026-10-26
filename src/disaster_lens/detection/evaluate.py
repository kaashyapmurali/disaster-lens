"""Streaming native-mask AP/recall, separate box metrics, and compact exports."""

import json
import time
from typing import IO

import torch
from torch.utils.data import DataLoader

from disaster_lens.detection.mask_metrics import encode_window, match_mask_scene
from disaster_lens.detection.box_metrics import match_scene, summarize_localization
from disaster_lens.detection.masks import CompactMasks, materialize_mask_window
from disaster_lens.detection.model import DetectorConfig
from disaster_lens.detection.model import identify_detections
from disaster_lens.data.footprints import MaskWindow
from disaster_lens.profiling import batch_timing_summary, measurements, profiled_batches, start_measurement, sync


def evaluate_instances(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, *, config: DetectorConfig,
    operating_threshold: float = .5, prediction_stream: IO[str] | None = None, run_key: str = "",
) -> dict:
    """Primary pooled pixel-mask AP, with separately named bounding-box results.

    Materializes only one native prediction window at a time on CPU. The output
    limit and score floor are explicit; this is not standard COCO AP100.
    """
    if config.mask_representation != "compact":
        raise ValueError("Instance evaluator requires compact outputs, not dense full-image mask stacks.")
    if prediction_stream is not None and not run_key:
        raise ValueError("Export run key required.")
    data = loader.dataset
    model.eval()
    seen = set()
    box_scenes, mask_scenes = [], []
    repaired = 0
    profile: dict = {}
    started = start_measurement(device)
    with torch.inference_mode():
        for batch_index, ((images, targets), timing) in enumerate(profiled_batches(loader, profile), start=1):
            transfer_started = time.perf_counter()
            device_images = [image.to(device) for image in images]
            sync(device)
            timing["transfer_seconds"] = time.perf_counter() - transfer_started
            inference_started = time.perf_counter()
            predictions = model(device_images)
            sync(device)
            timing["inference_seconds"] = time.perf_counter() - inference_started
            matching_started = time.perf_counter()
            if len(predictions) != len(targets):
                raise ValueError("Instance prediction/scene count mismatch.")
            for prediction, target in zip(predictions, targets, strict=True):
                index = int(target["image_id"].item())
                if index in seen or not 0 <= index < len(data):
                    raise ValueError("Repeated or invalid instance evaluation scene index.")
                seen.add(index)
                row = data.scenes[index]
                expected = torch.tensor([building["box"] for building in row["buildings"]], dtype=torch.float32).reshape(-1, 4)
                references = target.get("compact_masks")
                if (not isinstance(references, CompactMasks) or references.image_size != (row["height"], row["width"])
                        or len(references) != len(expected) or not torch.equal(target["boxes"], expected)
                        or not (target["labels"] == 1).all()):
                    raise ValueError("Instance reference targets differ from the indexed scene.")
                if "masks" in prediction or tuple(prediction["mask_image_size"]) != references.image_size:
                    raise ValueError("Expected compact ROI masks plus the native grid.")
                output = {key: prediction[key].detach().float().cpu() if key != "labels"
                          else prediction[key].detach().cpu() for key in ("boxes", "scores", "labels", "mask_roi_probs")}
                if output["labels"].dtype != torch.int64:
                    raise ValueError("Instance labels must be int64.")
                identify_detections(output, image_key=row["scene_id"], run_key="validation")
                boxes, scores, probabilities = output["boxes"], output["scores"], output["mask_roi_probs"]
                if (probabilities.shape != (len(boxes), 1, 28, 28) or not torch.isfinite(probabilities).all()
                        or (probabilities < 0).any() or (probabilities > 1).any()
                        or (boxes[:, (0, 2)] > row["width"]).any() or (boxes[:, (1, 3)] > row["height"]).any()
                        or len(boxes) > config.detections_per_image or (scores < config.score_threshold).any()):
                    raise ValueError("Invalid compact instance predictions or output limits.")
                box_matches = match_scene(row["scene_id"], row["event"], boxes.numpy(), scores.numpy(), expected.numpy(),
                                          max_detections=config.detections_per_image)
                encoded = {}
                def prediction_window(detection: int) -> MaskWindow:
                    native = materialize_mask_window(probabilities[detection], boxes[detection], references.image_size)
                    binary = MaskWindow(native.bounds, native.binary(config.mask_threshold))
                    if prediction_stream is not None:
                        encoded[detection] = encode_window(binary)
                    return binary
                mask_matches = match_mask_scene(row["scene_id"], row["event"], prediction_window, scores.numpy(),
                                                references.windows, max_detections=config.detections_per_image)
                box_scenes.append(box_matches)
                mask_scenes.append(mask_matches)
                repairs = [{"reference_uid": building["reference_uid"], **window.repair}
                           for building, window in zip(row["buildings"], references.windows, strict=True)
                           if window.repair is not None]
                repaired += len(repairs)
                if prediction_stream is not None:
                    ordered = {key: output[key][mask_matches.order] for key in ("boxes", "scores", "labels")}
                    detections = identify_detections(ordered, image_key=row["scene_id"], run_key=run_key)
                    for position, (detection, original) in enumerate(zip(detections, mask_matches.order, strict=True)):
                        detection["mask_window_rle"] = encoded[int(original)]
                        reference = mask_matches.matched_indices[0, position]
                        detection["matched_reference_uid_mask_iou50"] = (
                            row["buildings"][reference]["reference_uid"] if reference >= 0 else None)
                    record = {
                        "scene_id": row["scene_id"], "event": row["event"],
                        "width": row["width"], "height": row["height"], "pre_image": row["pre_image"],
                        "annotation_sha256": row["annotation_sha256"], "geometry_policy": data.geometry_policy,
                        "mask_threshold": config.mask_threshold, "geometry_repairs": repairs,
                        "detections": detections,
                        "reference_buildings": [{"reference_uid": building["reference_uid"], "box": building["box"],
                                                 "mask_window_rle": encode_window(window)}
                                                for building, window in zip(row["buildings"], references.windows, strict=True)],
                    }
                    prediction_stream.write(json.dumps(record, allow_nan=False) + "\n")
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f"validation: {len(seen)}/{len(data)} instance scenes", flush=True)
            timing["cpu_validation_matching_export_seconds"] = time.perf_counter() - matching_started
    if seen != set(range(len(data))):
        raise ValueError("Instance validation must cover each selected scene exactly once.")
    aggregation_started = time.perf_counter()
    kwargs = {"max_detections": config.detections_per_image, "score_floor": config.score_threshold,
              "operating_threshold": operating_threshold}
    metrics = summarize_localization(mask_scenes, metric_kind="mask", **kwargs)
    metrics["definition"].update(mask_threshold=config.mask_threshold, geometry_policy=data.geometry_policy,
                                  mask_representation="native-window binary pixel masks; one at a time on CPU",
                                  reference_rasterization="pixel-center intersection including boundary",
                                  export_encoding="window-local C-order alternating zero/one RLE; not COCO RLE")
    metrics["box_metrics"] = summarize_localization(box_scenes, **kwargs)
    metrics["repaired_reference_buildings"] = repaired
    aggregation_seconds = time.perf_counter() - aggregation_started
    metrics["runtime"] = {**measurements(device, started, len(seen)),
                          "batch_timing": batch_timing_summary(profile), "metric_aggregation_seconds": aggregation_seconds}
    return metrics
