"""Synthetic raw-cache tests; no images, pretrained weights, or remote calls."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from disaster_lens.damage.heads import TemporalFusionHead
from disaster_lens.damage.severity import SEVERITY_COLUMNS, severity_outputs
from disaster_lens.damage.feature_cache import CACHE_VERSION, CachedFeatureDataset, collate_cached_scenes
from disaster_lens.damage.train_cached import run_epoch, train_cached_fusion, validate_caches
from disaster_lens.data.scenes import CLASS_TO_INDEX
from disaster_lens.damage.export_cached import (
    PROBABILITY_COLUMNS, export_validation_predictions, validate_prediction_rows,
)


class CachedFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = self.make_cache("train")
        self.validation = self.make_cache("validation")
        self.output = self.root / "run"

    def make_cache(
        self, split: str, *, scene_prefix: str | None = None,
        uid_prefix: str | None = None, encoder: str = "frozen-weights",
    ) -> Path:
        path = self.root / split
        (path / "scenes").mkdir(parents=True)
        spec = {
            "cache_version": CACHE_VERSION, "class_to_index": CLASS_TO_INDEX,
            "building_dim": 6, "context_dim": 4, "encoder_state_sha256": encoder,
            "preprocessing": {"image_size": 64}, "roi": {"size": 3},
            "manifest_sha256": split, "building_index_sha256": split,
            "image_provenance_sha256": split,
        }
        cache_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        entries = []
        generator = torch.Generator().manual_seed(17)
        for index, labels in enumerate(([0, 1, 2], [3])):
            scene_id = f"{scene_prefix or split}-{index}"
            entry = {"scene_id": scene_id, "file": f"scenes/{scene_id}.pt",
                     "buildings": len(labels)}
            entries.append(entry)
            payload = {
                "cache_id": cache_id, "scene_id": scene_id,
                "building_uids": [f"{uid_prefix or split}-{label}" for label in labels],
                "labels": torch.tensor(labels),
                "features": {
                    "pre_building": torch.randn(len(labels), 6, generator=generator),
                    "post_building": torch.randn(len(labels), 6, generator=generator),
                    "pre_context": torch.randn(1, 4, generator=generator),
                    "post_context": torch.randn(1, 4, generator=generator),
                    "building_to_scene": torch.zeros(len(labels), dtype=torch.long),
                },
            }
            torch.save(payload, path / entry["file"])
        (path / "metadata.json").write_text(json.dumps({
            "spec": spec, "cache_id": cache_id, "complete": True, "scenes": entries,
        }))
        return path

    def test_training_records_config_and_selects_best_checkpoint(self) -> None:
        initial = TemporalFusionHead(6, 4, projected_dim=8, hidden_dim=8)
        initial_state = {key: value.clone() for key, value in initial.state_dict().items()}
        epoch_runner = run_epoch
        validation_scores = iter((0.8, 0.8, 0.3))

        def controlled_scores(model, loader, device, optimizer=None, class_weights=None):
            result = epoch_runner(model, loader, device, optimizer, class_weights)
            if optimizer is None:
                result["macro_f1"] = next(validation_scores)
            return result

        with patch("PIL.Image.open", side_effect=AssertionError("no images")), \
             patch("disaster_lens.damage.classifier.ResNet18Backbone",
                   side_effect=AssertionError("no encoder")), \
             patch("disaster_lens.damage.train_cached.TemporalFusionHead", return_value=initial), \
             patch("disaster_lens.damage.train_cached.run_epoch", side_effect=controlled_scores):
            history = train_cached_fusion(
                self.train, self.validation, self.output, epochs=3, batch_size=2,
                projected_dim=8, hidden_dim=8, class_weight="balanced", validation_batch_size=1,
            )
        best = torch.load(self.output / "best.pt", weights_only=True)
        last = torch.load(self.output / "last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(last["epoch"], 3)
        self.assertTrue(any(not torch.equal(value, last["model_state_dict"][key])
                            for key, value in initial_state.items()))
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(config["model"]["building_dim"], 6)
        self.assertEqual(config["model"]["context_dim"], 4)
        self.assertEqual(config["class_to_index"], CLASS_TO_INDEX)
        self.assertEqual(config["optimizer"]["name"], "AdamW")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["batch_size"], 2)
        self.assertEqual(config["validation_batch_size"], 1)
        self.assertEqual(history[0]["train"]["optimizer_steps"], 1)
        self.assertEqual(history[0]["train"]["buildings_per_batch"], {"min": 4, "max": 4, "mean": 4.0})
        self.assertEqual(history[0]["validation"]["optimizer_steps"], 0)
        self.assertEqual(history[0]["validation"]["buildings_per_batch"], {"min": 1, "max": 3, "mean": 2.0})
        self.assertEqual(config["class_weights"], [1., 1., 1., 1.])
        self.assertEqual(config["caches"]["train"]["cache_id"],
                         CachedFeatureDataset(self.train).metadata["cache_id"])
        self.assertEqual(json.loads((self.output / "history.json").read_text())["epochs"], history)
        self.assertEqual(sum(map(sum, history[-1]["validation"]["confusion_matrix"])), 4)
        self.assertEqual(list(history[-1]["validation"]["per_class"]), list(CLASS_TO_INDEX))
        before = (self.output / "best.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            train_cached_fusion(self.train, self.validation, self.output)
        self.assertEqual(before, (self.output / "best.pt").read_bytes())

    def test_validation_does_not_update_head_and_counts_buildings(self) -> None:
        data = CachedFeatureDataset(self.validation)
        loader = DataLoader(data, batch_size=2, collate_fn=collate_cached_scenes)
        head = TemporalFusionHead(6, 4, projected_dim=8, hidden_dim=8)
        before = {key: value.clone() for key, value in head.state_dict().items()}
        metrics = run_epoch(head, loader, torch.device("cpu"))
        self.assertEqual(metrics["buildings"], 4)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
        batch = collate_cached_scenes([data[0], data[1]])
        with torch.no_grad():
            expected = torch.nn.functional.cross_entropy(head(batch.features), batch.labels)
        self.assertAlmostEqual(metrics["loss"], expected.item())

    def test_rejects_incompatible_recipes_and_overlapping_identities_before_training(self) -> None:
        incompatible = self.make_cache("different", encoder="other-weights")
        scenes = self.make_cache("scene-overlap", scene_prefix="train")
        buildings = self.make_cache("uid-overlap", uid_prefix="train")
        for cache, message in ((incompatible, "recipes differ"), (scenes, "share scene IDs"),
                               (buildings, "share building UIDs")):
            with self.subTest(cache=cache), \
                 patch("disaster_lens.damage.train_cached.TemporalFusionHead") as head:
                with self.assertRaisesRegex(ValueError, message):
                    train_cached_fusion(self.train, cache, self.output)
                head.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_rejects_nonfinite_features_and_missing_training_class(self) -> None:
        shard_path = self.train / "scenes/train-1.pt"
        payload = torch.load(shard_path, weights_only=True)
        payload["features"]["pre_building"][0, 0] = float("nan")
        torch.save(payload, shard_path)
        with self.assertRaisesRegex(ValueError, "Non-finite cached features"):
            validate_caches(CachedFeatureDataset(self.train), CachedFeatureDataset(self.validation))
        payload["features"]["pre_building"][0, 0] = 0
        payload["labels"][0] = 0
        torch.save(payload, shard_path)
        with self.assertRaisesRegex(ValueError, "all four classes"):
            validate_caches(CachedFeatureDataset(self.train), CachedFeatureDataset(self.validation))

    def test_export_reloads_best_checkpoint_and_preserves_validation_identities(self) -> None:
        train_cached_fusion(
            self.train, self.validation, self.output, epochs=1, batch_size=2,
            projected_dim=8, hidden_dim=8,
        )
        best = torch.load(self.output / "best.pt", weights_only=True)
        model = TemporalFusionHead(**best["config"]["model"])
        model.load_state_dict(best["model_state_dict"])
        model.eval()
        data = CachedFeatureDataset(self.validation)
        batch = collate_cached_scenes([data[0], data[1]])
        with torch.inference_mode():
            expected_probabilities = model(batch.features).softmax(dim=1)
        # A broken last checkpoint must not affect best-checkpoint export.
        (self.output / "last.pt").write_bytes(b"not a checkpoint")
        with patch("PIL.Image.open", side_effect=AssertionError("no images")), \
             patch("disaster_lens.damage.classifier.ResNet18Backbone",
                   side_effect=AssertionError("no encoder")):
            metrics = export_validation_predictions(
                self.output, self.validation, batch_size=2,
            )
        directory = self.output / "validation_predictions"
        with (directory / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(row["scene_id"], row["building_uid"], row["reference_label"]) for row in rows],
                         [("validation-0", "validation-0", "no-damage"),
                          ("validation-0", "validation-1", "minor-damage"),
                          ("validation-0", "validation-2", "major-damage"),
                          ("validation-1", "validation-3", "destroyed")])
        actual_probabilities = torch.tensor([
            [float(row[column]) for column in PROBABILITY_COLUMNS] for row in rows
        ])
        torch.testing.assert_close(actual_probabilities, expected_probabilities)
        self.assertEqual(metrics["macro_f1"], best["validation"]["macro_f1"])
        self.assertEqual(metrics["confusion_matrix"], best["validation"]["confusion_matrix"])
        self.assertAlmostEqual(metrics["loss"], best["validation"]["loss"], places=6)
        self.assertEqual(metrics["epoch"], best["epoch"])
        self.assertEqual(json.loads((directory / "metrics.json").read_text()), metrics)
        before = (directory / "predictions.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            export_validation_predictions(self.output, self.validation)
        self.assertEqual(before, (directory / "predictions.csv").read_bytes())

    def test_export_rejects_wrong_cache_and_nonfinite_model_without_publishing(self) -> None:
        train_cached_fusion(
            self.train, self.validation, self.output, epochs=1,
            projected_dim=8, hidden_dim=8,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint's validation cache"):
            export_validation_predictions(self.output, self.train)
        best_path = self.output / "best.pt"
        best = torch.load(best_path, weights_only=True)
        best["model_state_dict"]["fusion.3.bias"][0] = float("nan")
        torch.save(best, best_path)
        with self.assertRaisesRegex(ValueError, "Invalid validation logits"):
            export_validation_predictions(self.output, self.validation)
        self.assertFalse((self.output / "validation_predictions").exists())
        self.assertEqual(list(self.output.glob(".validation_predictions-*")), [])

    def test_prediction_row_validation_rejects_corruption(self) -> None:
        expected = [("scene", "uid", "no-damage")]
        row = {
            "scene_id": "scene", "building_uid": "uid",
            "reference_label": "no-damage", "predicted_label": "no-damage",
            **dict(zip(PROBABILITY_COLUMNS, ("0.7", "0.1", "0.1", "0.1"))),
        }
        validate_prediction_rows([row], expected)
        for changes, message in (
            ({"building_uid": "other"}, "alignment mismatch"),
            ({"reference_label": "minor-damage"}, "alignment mismatch"),
            ({PROBABILITY_COLUMNS[0]: "nan"}, "Invalid class probabilities"),
            ({PROBABILITY_COLUMNS[0]: "inf"}, "Invalid class probabilities"),
            ({PROBABILITY_COLUMNS[0]: "-0.1"}, "Invalid class probabilities"),
            ({PROBABILITY_COLUMNS[0]: "0.2"}, "sum to one"),
            ({"predicted_label": "destroyed"}, "argmax"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                validate_prediction_rows([{**row, **changes}], expected)
        with self.assertRaisesRegex(ValueError, "row count"):
            validate_prediction_rows([], expected)
        with self.assertRaisesRegex(ValueError, "Duplicate prediction identity"):
            validate_prediction_rows([row, row], expected * 2)

    def test_export_rejects_batch_identity_misalignment(self) -> None:
        train_cached_fusion(
            self.train, self.validation, self.output, epochs=1,
            projected_dim=8, hidden_dim=8,
        )

        def reorder_uids(samples):
            batch = collate_cached_scenes(samples)
            batch.building_uids.reverse()
            return batch

        with patch("disaster_lens.damage.export_cached.collate_cached_scenes", side_effect=reorder_uids):
            with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                export_validation_predictions(self.output, self.validation, batch_size=2)
        self.assertFalse((self.output / "validation_predictions").exists())

    def test_export_supports_legacy_ce_checkpoint_without_objective_metadata(self) -> None:
        train_cached_fusion(
            self.train, self.validation, self.output, epochs=1, projected_dim=8, hidden_dim=8,
        )
        path = self.output / "best.pt"
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["config"].pop("objective")
        torch.save(checkpoint, path)
        metrics = export_validation_predictions(self.output, self.validation)
        self.assertEqual(metrics["objective"], "cross-entropy")
        self.assertAlmostEqual(metrics["loss"], checkpoint["validation"]["loss"], places=6)

    def test_prediction_row_validation_rejects_inconsistent_severity_outputs(self) -> None:
        expected = [("scene", "uid", "destroyed")]
        values = [.1, .1, .1, .7]
        row = {
            "scene_id": "scene", "building_uid": "uid",
            "reference_label": "destroyed", "predicted_label": "destroyed",
            **{column: str(value) for column, value in zip(PROBABILITY_COLUMNS, values)},
            **{column: str(value) for column, value in zip(
                SEVERITY_COLUMNS, severity_outputs(torch.tensor([values], dtype=torch.float64))[0].tolist(),
            )},
        }
        validate_prediction_rows([row], expected)
        for value in ("nan", "1.2", "0"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "severity_index"):
                validate_prediction_rows([{**row, "severity_index": value}], expected)


if __name__ == "__main__":
    unittest.main()
