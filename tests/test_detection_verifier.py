import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from disaster_lens.damage.backbones import SceneFeatures
from disaster_lens.data.footprints import MaskWindow
from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.detection import verifier
from disaster_lens.detection.candidates import (
    GEOMETRY_FEATURES,
    geometry_features,
    load_candidates,
    localization_metrics,
    matched_reference_uids,
)
from disaster_lens.detection.mask_metrics import encode_window

SIZE = 128


class PoolingBackbone(nn.Module):
    """Stand-in for DINOv3: each 16 px patch's feature is its mean colour."""
    patch_size, spatial_channels = 16, 3

    def forward(self, images):
        spatial = F.avg_pool2d(images, 16)
        return SceneFeatures(spatial, spatial.mean(dim=(-2, -1)))


def square(x1, y1, x2, y2):
    pixels = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)
    return MaskWindow((x1, y1, x2, y2), pixels)


def detection(scene, index, window, score, matched=None):
    return {"building_id": f"{scene}:{index:06d}", "box": list(map(float, window.bounds)), "score": score,
            "mask_window_rle": encode_window(window), "matched_reference_uid_mask_iou50": matched}


def write_export(root: Path, data_dir: Path) -> Path:
    """Two scenes: duplicates and a false positive in `a`; a single good candidate and a low-score one in `b`."""
    scenes = {
        "a": ([("a-r0", square(10, 10, 40, 40)), ("a-r1", square(60, 60, 100, 100))],
              [(square(10, 10, 40, 40), .9, "a-r0"), (square(12, 12, 40, 40), .6, None),
               (square(60, 60, 98, 100), .3, "a-r1"), (square(100, 0, 120, 20), .2, None), (square(0, 100, 8, 108), .05, None)]),
        "b": ([("b-r0", square(30, 30, 70, 70))],
              [(square(30, 30, 70, 70), .8, "b-r0"), (square(80, 80, 90, 90), .15, None)]),
    }
    root.mkdir(parents=True)
    lines = []
    for scene, (references, detections) in scenes.items():
        image = data_dir / f"raw/train/images/{scene}_pre_disaster.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(len(scene))
        Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(image)
        lines.append(json.dumps({
            "scene_id": scene, "event": f"event-{scene}", "width": SIZE, "height": SIZE,
            "pre_image": f"raw/train/images/{scene}_pre_disaster.png",
            "detections": [detection(scene, i, w, s, m) for i, (w, s, m) in enumerate(detections)],
            "reference_buildings": [{"reference_uid": uid, "box": list(w.bounds), "mask_window_rle": encode_window(w)}
                                    for uid, w in references]}))
    payload = ("\n".join(lines) + "\n").encode()
    (root / "predictions.jsonl").write_bytes(payload)
    (root / "metrics.json").write_text(json.dumps({
        "export_verification": "matched_checkpoint_metrics", "definition": {"score_floor": .001},
        "provenance": {"predictions_sha256": hashlib.sha256(payload).hexdigest()}}))
    return root


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.data = root / "data"
        self.export = write_export(root / "export", self.data)

    def tearDown(self):
        self.temporary.cleanup()

    def test_candidates_keep_floor_and_precompute_ious(self):
        scenes, identity = load_candidates(self.export, .1)
        self.assertEqual([len(s) for s in scenes], [4, 2])
        self.assertEqual(identity["floor"], .1)
        np.testing.assert_allclose(scenes[0].best_iou(), [1., 28 * 28 / 900, 38 * 40 / 1600, 0.])
        self.assertEqual(scenes[0].best_outlines().tolist(), [True, False, True, False])   # the duplicate is negative

    def test_rescoring_rematches_and_reproduces_original_matching(self):
        scenes, _ = load_candidates(self.export, .001)
        original = matched_reference_uids(scenes, [s.scores for s in scenes])
        self.assertEqual(original["a:000000"], "a-r0")
        self.assertEqual(original["a:000001"], "")                 # duplicate of a-r0 under the original order
        swapped = [np.array([.6, .9, .3, .2, .05]), scenes[1].scores]
        rematched = matched_reference_uids(scenes, swapped)
        self.assertEqual((rematched["a:000000"], rematched["a:000001"]), ("", "a-r0"))   # IoU .87 >= .5: now matches
        metrics = localization_metrics(scenes, [s.scores for s in scenes], score_floor=.001)["pooled"]
        self.assertEqual(metrics["at_score_threshold_iou50"]["true_positives"], 2)     # a-r0 and b-r0 at score >= .5

    def test_geometry_features_flag_overlap_with_higher_scored_candidates(self):
        scenes, _ = load_candidates(self.export, .1)
        features = geometry_features(scenes[0])
        self.assertEqual(features.shape, (4, len(GEOMETRY_FEATURES)))
        column = GEOMETRY_FEATURES.index("max_iou_higher")
        self.assertEqual(features[0, column], 0.)
        self.assertAlmostEqual(float(features[1, column]), 28 * 28 / 900, places=5)
        self.assertEqual(features[3, GEOMETRY_FEATURES.index("max_inside_higher")], 0.)


class FeatureTests(unittest.TestCase):
    def test_tiled_map_equals_whole_image_encoding_away_from_nothing(self):
        image = torch.rand(3, SIZE, SIZE)
        tiled = verifier.tiled_feature_map(PoolingBackbone(), image, tile=64, margin=16, scale=2)
        whole = F.avg_pool2d(F.interpolate(image[None], scale_factor=2, mode="bilinear", align_corners=False), 16)[0]
        self.assertEqual(tiled.shape, (3, SIZE // 8, SIZE // 8))
        torch.testing.assert_close(tiled, whole)
        with self.assertRaises(ValueError):
            verifier.tiled_feature_map(PoolingBackbone(), image, tile=60, margin=16, scale=2)

    def test_mask_mean_weights_cells_by_covered_pixels(self):
        feature_map = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
        one_cell = encode_window(square(8, 0, 16, 8))
        torch.testing.assert_close(verifier.mask_mean(feature_map, one_cell, (32, 32), 8), feature_map[:, 0, 1])
        straddling = encode_window(square(4, 0, 12, 8))           # half of cell (0,0), half of cell (0,1)
        torch.testing.assert_close(verifier.mask_mean(feature_map, straddling, (32, 32), 8),
                                   (feature_map[:, 0, 0] + feature_map[:, 0, 1]) / 2)

    def test_folds_split_every_event(self):
        scenes = [mock.Mock(event=e) for e in ["x"] * 6 + ["y"] * 4]
        folds = verifier.scene_folds(scenes, 2, seed=1)
        for event in "xy":
            self.assertEqual(set(folds[[s.event == event for s in scenes]]), {0, 1})


class PipelineTests(unittest.TestCase):
    def test_extract_then_cross_validate_on_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            export = write_export(root / "export", data)
            encoder = root / "encoder"
            encoder.mkdir()
            (encoder / "model.safetensors").write_bytes(b"stand-in")
            with mock.patch("disaster_lens.damage.dinov3.DINOv3Backbone.from_local_checkpoint", return_value=PoolingBackbone()):
                summary = verifier.extract(export, data, encoder, root / "features", floor=.1, device=torch.device("cpu"),
                                           tile=64, margin=16, scale=2, progress=lambda _: None)
                again = verifier.extract(export, data, encoder, root / "features", floor=.1, device=torch.device("cpu"),
                                         tile=64, margin=16, scale=2, progress=lambda _: None)
            self.assertEqual((summary["candidates"], summary["written"], again["written"]), (6, 2, 0))
            shard = torch.load(root / "features/scenes/a.pt", weights_only=True)
            self.assertEqual(tuple(shard["features"].shape), (4, 18))
            damage = self.damage_run(root / "damage", export)
            report = verifier.cross_validate(export, root / "features", damage, root / "out", folds=2, epochs=2)
            self.assertEqual(set(report["variants"]), {"detector", "geometry_head", "vit_head"})
            self.assertEqual((report["candidates"], report["positives"]), (6, 3))
            self.assertEqual(report["variants"]["detector"]["end_to_end"]["0.5"]["matched"], 2)
            with (root / "out/scores.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            with self.assertRaises(FileExistsError):
                verifier.cross_validate(export, root / "features", damage, root / "out", folds=2, epochs=2)

    @staticmethod
    def damage_run(path: Path, export: Path) -> Path:
        path.mkdir()
        columns = [f"probability_{label}" for label in DAMAGE_LABELS]
        records = [json.loads(line) for line in (export / "predictions.jsonl").read_text().splitlines()]
        with (path / "reference_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["scene_id", "building_uid", "reference_label", *columns])
            for record in records:
                for ref in record["reference_buildings"]:
                    writer.writerow([record["scene_id"], ref["reference_uid"], "no-damage", 1, 0, 0, 0])
        with (path / "detection_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["scene_id", "building_id", "score", "matched_reference_uid", *columns])
            for record in records:
                for d in record["detections"]:
                    writer.writerow([record["scene_id"], d["building_id"], d["score"],
                                     d["matched_reference_uid_mask_iou50"] or "", 1, 0, 0, 0])
        return path


if __name__ == "__main__":
    unittest.main()
