"""Damage predictors for arbitrary box sets, the end-to-end runner, and the scorer; synthetic CPU data only."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from disaster_lens.data.manifest import DAMAGE_LABELS
from disaster_lens.end_to_end.run import PROBABILITY_COLUMNS, run_end_to_end, scene_arms, verify_confusion
from disaster_lens.end_to_end.score import macro_f1, score_detected

HAS_TIMM = importlib.util.find_spec("timm") is not None
if HAS_TIMM:
    import test_damage_live as fixtures
    from disaster_lens.damage.classifier import DamageClassifier, build_head
    from disaster_lens.damage.predictors import Predictor, load_damage_model
    from disaster_lens.damage.train_live import train_live


def detections_for(scenes) -> dict:
    return {scene: {"width": 64, "height": 64, "event": "e", "ids": [f"{scene}:d0", f"{scene}:d1", f"{scene}:d2"],
                    "scores": [.9, .6, .2], "boxes": [[8, 8, 40, 40], [24, 8, 56, 40], [40, 36, 58, 60]],
                    "matched": [f"{scene}-uid", None, None]} for scene in scenes}


@unittest.skipUnless(HAS_TIMM, "Install the vit dependency group.")
class PredictorTests(unittest.TestCase):
    setUpClass, tearDownClass = fixtures.DamageClassifierTests.setUpClass, fixtures.DamageClassifierTests.tearDownClass

    def setUp(self):
        torch.manual_seed(0)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = fixtures.write_split_files(self.root)

    def model(self, inputs):
        kwargs = {"projected_dim": 8, "hidden_dim": 8, "dropout": 0} if inputs == "paired" else {}
        return DamageClassifier(fixtures.tiny_vit(), build_head(inputs, 288, 32, **kwargs), inputs=inputs).eval()

    def test_predictor_equals_the_model_forward_for_both_input_types(self):
        sample = self.data["validation"][0]
        pre, post, boxes = sample.pre_image[None], sample.post_image[None], sample.pre_boxes
        for inputs in ("post", "paired"):
            with self.subTest(inputs=inputs):
                model = self.model(inputs)
                with torch.inference_mode():
                    expected = model(pre, post, [boxes]).float().softmax(1)
                got = Predictor(model, "float32").predict(pre, post, {"a": (boxes, boxes), "e": (boxes[:0], boxes[:0])},
                                                          torch.device("cpu"))
                torch.testing.assert_close(got["a"], expected, rtol=0, atol=1e-6)
                self.assertEqual(got["e"].shape, (0, 4))

    def test_loader_handles_live_checkpoints_and_cached_heads(self):
        output = self.root / "live"
        backbone = fixtures.tiny_vit()
        train_live("dinov3-vitl", "post", self.root, self.root, output, image_size=64, epochs=1, backbone=backbone)
        loaded = load_damage_model(output / "best.pt", self.root / "validation.csv", self.root, torch.device("cpu"))
        self.assertEqual((loaded.provenance["inputs"], loaded.provenance["encoder"]), ("post", "dinov3-vitl"))
        self.assertIn("confusion_matrix", loaded.reference_check)
        head = fixtures.cached_head_checkpoint(self.root, self.data, backbone)
        with patch("disaster_lens.damage.dinov3.DINOv3Backbone.from_local_checkpoint", return_value=backbone):
            cached = load_damage_model(head, self.root / "validation.csv", self.root, torch.device("cpu"),
                                       encoder_dir=self.root, saved_predictions=self.root / "saved.csv")
        self.assertEqual((cached.provenance["inputs"], cached.provenance["trainable_blocks"]), ("paired", 0))
        self.assertEqual(cached.reference_check, {"saved_predictions": self.root / "saved.csv"})
        with torch.no_grad():
            backbone.encoder.patch_embed.proj.weight.add_(1)          # different encoder weights than the cache
        with patch("disaster_lens.damage.dinov3.DINOv3Backbone.from_local_checkpoint", return_value=backbone), \
             self.assertRaisesRegex(ValueError, "encoder_state_sha256"):
            load_damage_model(head, self.root / "validation.csv", self.root, torch.device("cpu"), encoder_dir=self.root)
        with self.assertRaisesRegex(ValueError, "encoder-dir"):
            load_damage_model(output / "best.pt", self.root / "validation.csv", self.root, torch.device("cpu"), encoder_dir=self.root)

    def saved_predictions(self, model, flip=False) -> Path:
        path = self.root / "saved.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["scene_id", "building_uid", *PROBABILITY_COLUMNS])
            writer.writeheader()
            for index, scene in enumerate(self.data["validation"].scenes):
                sample = self.data["validation"][index]
                with torch.inference_mode():
                    probabilities = model(sample.pre_image[None], sample.post_image[None], [sample.pre_boxes]).softmax(1)
                for building, row in zip(scene.buildings, probabilities.tolist(), strict=True):
                    writer.writerow({"scene_id": scene.scene_id, "building_uid": building.uid,
                                     **dict(zip(PROBABILITY_COLUMNS, map(repr, row[::-1] if flip else row), strict=True))})
        return path

    def test_runner_publishes_both_arms_and_verifies_the_annotated_footprints(self):
        model = self.model("paired")
        dataset = self.data["validation"]
        detections = detections_for([s.scene_id for s in dataset.scenes])
        out = self.root / "out"
        metrics = run_end_to_end(predictor=Predictor(model, "float32"), dataset=dataset, detections=detections, output_dir=out,
                                 device=torch.device("cpu"), reference_check={"saved_predictions": self.saved_predictions(model)},
                                 progress=lambda *_: None)
        self.assertEqual(metrics["reference_check"]["argmax_agreement"], 1.)
        self.assertEqual((metrics["reference_rows"], metrics["detection_rows"]), (4, 3 * len(dataset.scenes)))
        with self.assertRaises(FileExistsError):
            run_end_to_end(predictor=Predictor(model, "float32"), dataset=dataset, detections=detections, output_dir=out,
                           device=torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            run_end_to_end(predictor=Predictor(model, "float32"), dataset=dataset, detections=detections,
                           output_dir=self.root / "bad", device=torch.device("cpu"),
                           reference_check={"saved_predictions": self.saved_predictions(model, flip=True)}, progress=lambda *_: None)
        self.assertTrue((self.root / "bad/metrics.json").exists())          # kept for inspection
        with self.assertRaisesRegex(ValueError, "every scene"):
            run_end_to_end(predictor=Predictor(model, "float32"), dataset=dataset, detections={}, output_dir=self.root / "x",
                           device=torch.device("cpu"))


class ArmAndCheckTests(unittest.TestCase):
    def test_scene_arms_reuse_boxes_on_both_dates_and_require_the_native_grid(self):
        arms = scene_arms(torch.tensor([[1., 2., 9., 9.]]), detections_for(["s"])["s"], 64, torch.device("cpu"))
        self.assertEqual(set(arms), {"reference", "detected"})
        for pre, post in arms.values():
            self.assertTrue(torch.equal(pre, post))
        with self.assertRaisesRegex(ValueError, "native image grid"):
            scene_arms(torch.zeros(0, 4), {**detections_for(["s"])["s"], "width": 128}, 64, torch.device("cpu"))

    def rows(self, pairs):
        out = []
        for truth, predicted in pairs:
            probabilities = [0.] * 4
            probabilities[predicted] = 1.
            out.append({"reference_label": DAMAGE_LABELS[truth], **dict(zip(PROBABILITY_COLUMNS, map(str, probabilities), strict=True))})
        return out

    def test_confusion_check_passes_on_reproduction_and_fails_on_changes(self):
        pairs = [(c, c) for c in range(4) for _ in range(500)]
        confusion = [[500 if i == j else 0 for j in range(4)] for i in range(4)]
        self.assertTrue(verify_confusion(self.rows(pairs), confusion, 1.)["passed"])
        flipped = [(0, 1)] * 3 + pairs[3:]
        result = verify_confusion(self.rows(flipped), confusion, 1.)
        self.assertEqual(result["changed_predictions_upper_bound"], 3)
        self.assertFalse(result["passed"])
        self.assertFalse(verify_confusion(self.rows(pairs[:-1]), confusion, 1.)["passed"])


def score_row(uid, score, cls):
    probabilities = [0.] * 4
    probabilities[cls] = 1.
    return {"matched_reference_uid": uid, "score": str(score), **dict(zip(PROBABILITY_COLUMNS, map(str, probabilities), strict=True))}


class ScoreTests(unittest.TestCase):
    def test_macro_f1(self):
        truth = np.array([0, 1, 2, 3])
        self.assertEqual(macro_f1(truth, truth), 1.)
        self.assertEqual(macro_f1(truth, (truth + 1) % 4), 0.)

    def test_detected_scoring_separates_detection_from_damage(self):
        truth = {"a": 0, "b": 1, "c": 2}
        rows = [score_row("a", .9, 0), score_row("b", .8, 3), score_row("a", .6, 0), score_row("", .7, 1),
                score_row("zzz", .9, 0), score_row("c", .2, 2)]
        out = score_detected(rows, truth, .5)
        self.assertEqual(out["matched"], 2)                         # c is below threshold; zzz has no damage label
        self.assertAlmostEqual(out["recall_of_labeled"], 2 / 3)
        self.assertEqual(out["unmatched_false_positives"], 2)       # the duplicate of a and the unmatched detection
        self.assertEqual(out["damage_given_match_accuracy"], .5)
        self.assertLess(out["end_to_end_macro_f1"], out["damage_given_match_macro_f1"])

    def test_score_run_reads_new_and_earlier_run_layouts(self):
        from disaster_lens.end_to_end.score import score_run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm_column in (False, True):
                reference = [{"scene_id": "s", "building_uid": "a", "reference_label": "no-damage", **score_row("", 1, 0)}]
                detections = [{"scene_id": "s", "building_id": "d", **score_row("a", .9, 0)}]
                for name, rows in (("reference_predictions.csv", reference), ("detection_predictions.csv", detections)):
                    if arm_column:   # earlier runs stored several arms; only the unshifted one is scored
                        prefix = "reference" if name.startswith("reference") else "detected"
                        rows = [{"arm": f"{prefix}_identity", **r} for r in rows] + [{"arm": f"{prefix}_tile", **rows[0]}]
                    with (root / name).open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                (root / "metrics.json").write_text(json.dumps({"reference_check": {"passed": True}}))
                report = score_run(root)
                self.assertEqual(report["annotated_footprints"]["buildings"], 1)
                self.assertEqual(report["detected"]["0.5"]["matched"], 1)


if __name__ == "__main__":
    unittest.main()
