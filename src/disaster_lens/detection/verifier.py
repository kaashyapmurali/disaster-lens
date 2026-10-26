"""Re-score detector candidates with a verifier on frozen DINOv3 features (pilot).

    # GPU: features for every candidate of a verified export (resumable, one shard per scene)
    python -m disaster_lens.detection.verifier extract --export-dir EXPORT --data-dir datasets \
        --encoder-dir DINOV3_DIR --output-dir FEATURES [--floor 0.1]
    # CPU: scene-grouped cross-validation on one split, then exact re-matching and scoring
    python -m disaster_lens.detection.verifier cross-validate --export-dir EXPORT --features FEATURES \
        --damage-run END_TO_END_RUN --output-dir OUT

Each candidate is described by DINOv3 ViT-L features of the pre-event image, computed on 512 px tiles enlarged 2x
(stride 8 native pixels): a 2x2 box grid, the box enlarged 2x (context) and a mean over the predicted mask, plus
detector-score and geometry features. The verifier predicts whether the candidate is the best outline of an
annotated footprint (highest mask IoU for that footprint, >= 0.5; near-duplicates are negatives). Its out-of-fold probabilities replace the detector scores; matching, AP and the end-to-end damage
score are then recomputed with the export's own code. Candidates from one scene never train the head that scores
them; folds are balanced per event.
"""

import argparse
import csv
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision.ops import roi_align

from disaster_lens.detection.candidates import (
    GEOMETRY_FEATURES,
    SceneCandidates,
    geometry_features,
    load_candidates,
    localization_metrics,
    matched_reference_uids,
)
from disaster_lens.detection.mask_metrics import decode_window

FEATURE_VERSION = 1
IOU_TARGET = .5


# --- features -----------------------------------------------------------------------------------------------------

def tiled_feature_map(backbone, image: Tensor, *, tile: int = 512, margin: int = 64, scale: int = 2) -> Tensor:
    """[C, H/stride, W/stride] patch features of one normalized [3, H, W] image, stride = patch / scale.

    Each tile is encoded with `margin` pixels of surrounding context, enlarged `scale`x, and only its own cells kept.
    """
    _, height, width = image.shape
    patch = backbone.patch_size
    stride = patch // scale
    if patch % scale or height % tile or width % tile or tile % stride or margin % stride:
        raise ValueError("Tile, margin and image size must align with the feature stride.")
    out = image.new_zeros(backbone.spatial_channels, height // stride, width // stride)
    cells = tile // stride
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1, x1 = max(0, y0 - margin), max(0, x0 - margin)
            y2, x2 = min(height, y0 + tile + margin), min(width, x0 + tile + margin)
            window = F.interpolate(image[None, :, y1:y2, x1:x2], scale_factor=scale, mode="bilinear", align_corners=False)
            spatial = backbone(window).spatial_map[0].to(out.dtype)
            oy, ox = (y0 - y1) // stride, (x0 - x1) // stride
            out[:, y0 // stride:y0 // stride + cells, x0 // stride:x0 // stride + cells] = spatial[:, oy:oy + cells, ox:ox + cells]
    return out


def mask_mean(feature_map: Tensor, rle: dict, image_size: tuple[int, int], stride: int) -> Tensor | None:
    """Feature mean over the cells a predicted mask covers, weighted by covered pixels (None for an empty mask)."""
    window = decode_window(rle, image_size)
    if window.area == 0:
        return None
    x1, y1, x2, y2 = window.bounds
    gx1, gy1, gx2, gy2 = x1 // stride, y1 // stride, -(-x2 // stride), -(-y2 // stride)
    padded = np.zeros(((gy2 - gy1) * stride, (gx2 - gx1) * stride), dtype=np.float32)
    padded[y1 - gy1 * stride:y2 - gy1 * stride, x1 - gx1 * stride:x2 - gx1 * stride] = window.pixels
    weights = padded.reshape(gy2 - gy1, stride, gx2 - gx1, stride).sum(axis=(1, 3))
    weights = torch.from_numpy(weights / weights.sum()).to(feature_map)
    return (feature_map[:, gy1:gy2, gx1:gx2] * weights).sum(dim=(1, 2))


def candidate_features(feature_map: Tensor, scene: SceneCandidates, stride: int) -> Tensor:
    """[N, 6C]: 2x2 box grid (4C), 2x enlarged box (C), predicted-mask mean (C)."""
    if not len(scene):
        return feature_map.new_zeros(0, 6 * feature_map.shape[0])
    boxes = torch.as_tensor(scene.boxes, dtype=torch.float32, device=feature_map.device)
    centre, half = (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]).clamp(min=stride)
    limit = boxes.new_tensor([scene.width, scene.height])
    context = torch.cat([(centre - half).clamp(min=0), torch.minimum(centre + half, limit)], dim=1)
    pooled = feature_map[None].float()
    grid = roi_align(pooled, [boxes], output_size=2, spatial_scale=1 / stride, sampling_ratio=2, aligned=True)
    around = roi_align(pooled, [context], output_size=1, spatial_scale=1 / stride, sampling_ratio=2, aligned=True)
    grid, around = grid.flatten(1), around.flatten(1)
    masks = []
    for index, rle in enumerate(scene.mask_rles):
        mean = mask_mean(pooled[0], rle, (scene.height, scene.width), stride)
        masks.append(grid[index].reshape(-1, 4).mean(dim=1) if mean is None else mean)
    return torch.cat([grid, around, torch.stack(masks)], dim=1)


def load_image(data_dir: Path, relative: str, mean: tuple, std: tuple) -> Tensor:
    path = (data_dir / relative).resolve()
    if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Pre-event image outside data_dir or missing: {relative}")
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


def _atomic_save(path: Path, value, *, tensors: bool) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        if tensors:
            torch.save(value, stream)
        else:
            stream.write((json.dumps(value, indent=2) + "\n").encode())
    temporary.replace(path)


def extract(export_dir: Path, data_dir: Path, encoder_dir: Path, output_dir: Path, *, floor: float, device: torch.device,
            tile: int = 512, margin: int = 64, scale: int = 2, progress=print) -> dict:
    from disaster_lens.damage.dinov3 import (
        SATELLITE_MEAN,
        SATELLITE_STD,
        DINOv3Backbone,
    )

    scenes, identity = load_candidates(export_dir, floor)
    with (encoder_dir / "model.safetensors").open("rb") as stream:
        encoder_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    spec = {"feature_version": FEATURE_VERSION, "export": identity["predictions_sha256"], "floor": floor,
            "encoder_sha256": encoder_sha, "tile": tile, "margin": margin, "scale": scale,
            "layout": "2x2 box grid | 2x enlarged box | predicted-mask mean; float16",
            "image": "pre-event", "normalization": [SATELLITE_MEAN, SATELLITE_STD],
            "precision": "bfloat16 autocast" if device.type == "cuda" else "float32"}
    spec = json.loads(json.dumps(spec))                      # tuples -> lists, as stored
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenes").mkdir(exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text())["spec"] != spec:
        raise ValueError("Existing features were built with a different configuration.")
    _atomic_save(metadata_path, {"spec": spec, "complete": False}, tensors=False)
    backbone = DINOv3Backbone.from_local_checkpoint(encoder_dir).to(device).eval().requires_grad_(False)
    stride = backbone.patch_size // scale
    started, written = time.perf_counter(), 0
    for index, scene in enumerate(scenes, start=1):
        shard = output_dir / "scenes" / f"{scene.scene_id}.pt"
        if not shard.exists():
            image = load_image(data_dir, scene.pre_image, SATELLITE_MEAN, SATELLITE_STD).to(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                feature_map = tiled_feature_map(backbone, image, tile=tile, margin=margin, scale=scale).float()
            with torch.inference_mode():
                features = candidate_features(feature_map, scene, stride)
            if not torch.isfinite(features).all():
                raise ValueError(f"Non-finite features in {scene.scene_id}")
            _atomic_save(shard, {"scene_id": scene.scene_id, "building_ids": scene.building_ids,
                                 "features": features.to("cpu", torch.float16)}, tensors=True)
            written += 1
        if index % 25 == 0 or index == len(scenes):
            progress(f"verifier features: {index}/{len(scenes)} scenes; {time.perf_counter() - started:.0f}s")
    summary = {"scenes": len(scenes), "written": written, "candidates": sum(map(len, scenes)),
               "elapsed_seconds": time.perf_counter() - started}
    _atomic_save(metadata_path, {"spec": spec, "complete": True, "summary": summary}, tensors=False)
    return summary


# --- verifier head and cross-validation ---------------------------------------------------------------------------

class VerifierHead(nn.Module):
    def __init__(self, inputs: int, hidden: int, dropout: float = .2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def scene_folds(scenes: list[SceneCandidates], folds: int, seed: int) -> np.ndarray:
    """Fold per scene; each event's scenes are shuffled and dealt round-robin, so every fold covers every event."""
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(scenes), dtype=np.int64)
    events = sorted({scene.event for scene in scenes})
    offset = 0
    for event in events:
        members = rng.permutation([i for i, scene in enumerate(scenes) if scene.event == event])
        assignment[members] = (np.arange(len(members)) + offset) % folds
        offset += len(members)
    return assignment


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, hidden: int, epochs: int,
                seed: int, batch_size: int = 1024, lr: float = 1e-3, weight_decay: float = 1e-2) -> np.ndarray:
    """Standardize on the training rows, fit a small MLP with BCE, return probabilities for test rows."""
    torch.manual_seed(seed)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    x = torch.from_numpy((train_x - mean) / std)
    y = torch.from_numpy(train_y.astype(np.float32))
    model = VerifierHead(x.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * math.ceil(len(x) / batch_size))
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for batch in torch.randperm(len(x), generator=generator).split(batch_size):
            loss = F.binary_cross_entropy_with_logits(model(x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy((test_x - mean) / std))).numpy().astype(np.float64)


def load_features(scenes: list[SceneCandidates], feature_dir: Path, export_sha: str, floor: float) -> np.ndarray:
    metadata = json.loads((feature_dir / "metadata.json").read_text())
    if not metadata["complete"] or metadata["spec"]["export"] != export_sha or metadata["spec"]["floor"] != floor:
        raise ValueError("Features are incomplete or were built from a different export/floor.")
    rows = []
    for scene in scenes:
        payload = torch.load(feature_dir / "scenes" / f"{scene.scene_id}.pt", map_location="cpu", weights_only=True)
        if payload["scene_id"] != scene.scene_id or payload["building_ids"] != scene.building_ids:
            raise ValueError(f"Feature shard does not match the candidates of {scene.scene_id}")
        rows.append(payload["features"].float().numpy())
    return np.concatenate(rows)


def cross_validated_scores(inputs: np.ndarray, targets: np.ndarray, scene_of_row: np.ndarray, fold_of_scene: np.ndarray,
                           *, hidden: int, epochs: int, seed: int) -> np.ndarray:
    fold_of_row = fold_of_scene[scene_of_row]
    scores = np.full(len(targets), np.nan)
    for fold in range(fold_of_scene.max() + 1):
        held = fold_of_row == fold
        scores[held] = fit_predict(inputs[~held], targets[~held], inputs[held], hidden=hidden, epochs=epochs, seed=seed + fold)
    if np.isnan(scores).any():
        raise ValueError("Every candidate needs an out-of-fold score.")
    return scores


def operating_curve(scenes: list[SceneCandidates], scores: list[np.ndarray]) -> dict:
    """Recall at the precision levels of interest, from IoU-0.5 matches pooled over scenes."""
    from disaster_lens.detection.candidates import match

    matches = match(scenes, scores)
    pooled = np.concatenate([m.scores for m in matches])
    hits = np.concatenate([m.matched_indices[0] >= 0 for m in matches])
    references = sum(len(scene.reference_uids) for scene in scenes)
    order = np.argsort(-pooled, kind="stable")
    tp = np.cumsum(hits[order])
    precision, recall = tp / np.arange(1, len(tp) + 1), tp / references
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(f1.argmax())
    return {"recall_at_precision": {f"{p:.2f}": float(recall[precision >= p].max()) if (precision >= p).any() else 0.
                                    for p in (.6, .65, .7, .75, .8)},
            "best_f1": float(f1[best]), "best_f1_threshold": float(pooled[order][best])}


def end_to_end_scores(scenes: list[SceneCandidates], scores: list[np.ndarray], damage_run: Path,
                      thresholds=(.5,)) -> dict:
    """Damage macro-F1 end to end, after re-matching the candidates under the new scores."""
    from disaster_lens.data.manifest import DAMAGE_LABELS
    from disaster_lens.end_to_end.score import _rows, score_detected

    truth = {r["building_uid"]: DAMAGE_LABELS.index(r["reference_label"]) for r in _rows(damage_run / "reference_predictions.csv")}
    damage = {r["building_id"]: r for r in _rows(damage_run / "detection_predictions.csv")}
    matched = matched_reference_uids(scenes, scores)
    rows = []
    for scene, values in zip(scenes, scores, strict=True):
        for building_id, score in zip(scene.building_ids, values, strict=True):
            if building_id not in damage:
                raise ValueError("The damage run has no prediction for a candidate; use the same export and floor.")
            rows.append({**damage[building_id], "score": score, "matched_reference_uid": matched[building_id]})
    return {f"{t:g}": score_detected(rows, truth, t) for t in thresholds}


def cross_validate(export_dir: Path, feature_dir: Path, damage_run: Path, output_dir: Path, *, floor: float = .1,
                   folds: int = 5, seed: int = 0, epochs: int = 8) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    scenes, identity = load_candidates(export_dir, floor)
    vit = load_features(scenes, feature_dir, identity["predictions_sha256"], floor)
    geometry = np.concatenate([geometry_features(scene) for scene in scenes])
    targets = np.concatenate([scene.best_outlines(IOU_TARGET) for scene in scenes]).astype(np.int64)
    scene_of_row = np.repeat(np.arange(len(scenes)), [len(scene) for scene in scenes])
    fold_of_scene = scene_folds(scenes, folds, seed)
    split = np.cumsum([len(scene) for scene in scenes])[:-1]
    variants = {"detector": np.concatenate([scene.scores for scene in scenes])}
    variants["geometry_head"] = cross_validated_scores(geometry, targets, scene_of_row, fold_of_scene,
                                                       hidden=64, epochs=epochs, seed=seed)
    variants["vit_head"] = cross_validated_scores(np.concatenate([vit, geometry], axis=1), targets, scene_of_row,
                                                  fold_of_scene, hidden=256, epochs=epochs, seed=seed)
    report = {"candidates": len(targets), "positives": int(targets.sum()), "scenes": len(scenes), "folds": folds,
              "floor": floor, "export_sha256": identity["predictions_sha256"], "geometry_features": GEOMETRY_FEATURES,
              "vit_feature_dim": vit.shape[1], "epochs": epochs, "seed": seed, "variants": {}}
    for name, flat in variants.items():
        per_scene = np.split(flat, split)
        detection = localization_metrics(scenes, per_scene, score_floor=0.)["pooled"]
        report["variants"][name] = {
            "mask_ap": detection["ap_50_95"], "mask_ap50": detection["ap50"],
            "at_0.5": detection["at_score_threshold_iou50"], **operating_curve(scenes, per_scene),
            "end_to_end": end_to_end_scores(scenes, per_scene, damage_run)}
    output_dir.mkdir(parents=True)
    with (output_dir / "scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_id", "building_id", "target", "fold", *variants])
        ids = [(scene.scene_id, b) for scene in scenes for b in scene.building_ids]
        for row, (scene_id, building_id) in enumerate(ids):
            writer.writerow([scene_id, building_id, targets[row], fold_of_scene[scene_of_row[row]],
                             *(repr(float(v[row])) for v in variants.values())])
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    e = commands.add_parser("extract")
    e.add_argument("--export-dir", type=Path, required=True)
    e.add_argument("--data-dir", type=Path, required=True)
    e.add_argument("--encoder-dir", type=Path, required=True)
    e.add_argument("--output-dir", type=Path, required=True)
    e.add_argument("--floor", type=float, default=.1)
    e.add_argument("--device", default="cuda")
    c = commands.add_parser("cross-validate")
    c.add_argument("--export-dir", type=Path, required=True)
    c.add_argument("--features", type=Path, required=True)
    c.add_argument("--damage-run", type=Path, required=True, help="End-to-end run on the same export (floor <= --floor).")
    c.add_argument("--output-dir", type=Path, required=True)
    c.add_argument("--floor", type=float, default=.1)
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--epochs", type=int, default=8)
    c.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "extract":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        print(json.dumps(extract(args.export_dir, args.data_dir, args.encoder_dir, args.output_dir,
                                 floor=args.floor, device=device), indent=2), flush=True)
    else:
        report = cross_validate(args.export_dir, args.features, args.damage_run, args.output_dir,
                                floor=args.floor, folds=args.folds, seed=args.seed, epochs=args.epochs)
        print(json.dumps(report["variants"], indent=1))


if __name__ == "__main__":
    main()
