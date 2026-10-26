"""Continuation tests: exact equivalence, lineage, safety refusals, legacy RNG fallback."""

import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import test_detection_training as base
from disaster_lens.detection.train import _restore_rng, _rng_snapshot, export, make_parser, train


class ContinuationTests(unittest.TestCase):
    setUp = base.LocalizationTrainingTests.setUp
    setUpClass = base.LocalizationTrainingTests.setUpClass
    tearDownClass = base.LocalizationTrainingTests.tearDownClass
    export_args = base.LocalizationTrainingTests.export_args

    def common(self, run: Path) -> list[str]:
        return ["--data-dir", str(self.root), "--index-dir", str(self.index), "--output-dir", str(run),
                "--device", "cpu", "--image-size", "64", "--max-detections", "12",
                "--post-nms-proposals", "16", "--pre-nms-proposals", "32"]

    def fresh(self, run: Path, epochs: int) -> dict:
        args = make_parser().parse_args(["train", "--fresh-training", "--coco-weights", "synthetic.pt",
                                         "--weights-sha256", "a" * 64, "--epochs", str(epochs), *self.common(run)])
        return train(args)

    def resume(self, parent: Path, run: Path, extra: list[str] | None = None, additional: int = 1, name: str = "last"):
        args = make_parser().parse_args(["train", "--resume-from", str(parent), "--resume-checkpoint", name,
                                         "--additional-epochs", str(additional), *self.common(run), *(extra or [])])
        return train(args)

    @staticmethod
    def state(run: Path) -> dict:
        return torch.load(run / "last.pt", weights_only=True)

    def test_continuation_equals_uninterrupted_training(self):
        continuous = self.root / "continuous"
        self.fresh(continuous, 2)
        parent = self.root / "parent"
        self.fresh(parent, 1)
        before = {name: (parent / name).read_bytes() for name in ("last.pt", "best.pt", "config.json", "history.json", "status.json")}
        child = self.root / "child"
        result = self.resume(parent, child)
        for name, data in before.items():
            self.assertEqual((parent / name).read_bytes(), data, f"parent {name} was modified")
        reference, resumed = self.state(continuous), self.state(child)
        self.assertEqual(resumed["epoch"], 2)
        for key, value in reference["model_state_dict"].items():
            torch.testing.assert_close(resumed["model_state_dict"][key], value, rtol=0, atol=0)
        for index, saved in reference["optimizer_state_dict"]["state"].items():
            torch.testing.assert_close(resumed["optimizer_state_dict"]["state"][index]["momentum_buffer"],
                                       saved["momentum_buffer"], rtol=0, atol=0)
        config = result["config"]
        self.assertEqual(config["initialization_mode"], "continuation")
        self.assertEqual((config["start_epoch"], config["epochs"], config["additional_epochs"]), (2, 2, 1))
        self.assertEqual([entry["epoch"] for entry in result["history"]], [2])
        self.assertTrue(config["rng_restored"])
        self.assertEqual(config["lineage"][0]["epoch"], 1)
        self.assertEqual(len(config["lineage"][0]["checkpoint_sha256"]), 64)
        self.assertEqual(config["initialization"]["weights_sha256"], "a" * 64)   # original pretrained identity preserved
        status = json.loads((child / "status.json").read_text())
        self.assertEqual(status["completed_epoch"], 2)
        self.assertIn("improves_on_parent", status)
        # A continuation run exports like any other run.
        metrics = export(make_parser().parse_args([
            "export", "--data-dir", str(self.root), "--index-dir", str(self.index), "--run-dir", str(child), "--device", "cpu"]))
        self.assertEqual(metrics["provenance"]["epoch"], 2)

    def test_chain_numbers_epochs_and_extends_lineage(self):
        parent = self.root / "p"
        self.fresh(parent, 1)
        self.resume(parent, self.root / "c1")
        result = self.resume(self.root / "c1", self.root / "c2", additional=2)
        self.assertEqual([entry["epoch"] for entry in result["history"]], [3, 4])
        self.assertEqual([item["epoch"] for item in result["config"]["lineage"]], [1, 2])
        self.assertEqual(result["config"]["epochs"], 4)

    def test_legacy_parent_without_rng_state_reseeds_and_says_so(self):
        parent = self.root / "legacy"
        self.fresh(parent, 1)
        for name in ("last.pt", "best.pt"):
            checkpoint = torch.load(parent / name, weights_only=True)
            checkpoint.pop("rng_state")
            torch.save(checkpoint, parent / name)
        config = self.resume(parent, self.root / "legacy-child")["config"]
        self.assertFalse(config["rng_restored"])
        self.assertIn("not an exact replay", config["rng_note"])

    def test_learning_rate_change_needs_explicit_flag_and_is_recorded(self):
        parent = self.root / "p"
        self.fresh(parent, 1)
        with self.assertRaisesRegex(ValueError, "allow-lr-change"):
            self.resume(parent, self.root / "no", ["--head-lr", "0.001"])
        self.assertFalse((self.root / "no").exists())
        result = self.resume(parent, self.root / "yes", ["--head-lr", "0.001", "--allow-lr-change"])
        self.assertEqual(result["config"]["recipe_changes"], {"head_lr": {"parent": .005, "new": .001}})
        saved = self.state(self.root / "yes")["optimizer_state_dict"]["param_groups"]
        self.assertEqual([group["lr"] for group in saved], [.001, .001])

    def test_recipe_and_mode_mismatches_are_refused_without_creating_output(self):
        parent = self.root / "p"
        self.fresh(parent, 1)
        cases = {
            "anchors": (["--anchor-sizes", "8", "16", "32", "64", "128"], "detector recipe"),
            "momentum": (["--momentum", "0.5"], "momentum"),
            "batch": (["--validation-batch-size", "2"], "validation_batch_size"),
        }
        for name, (extra, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                self.resume(parent, self.root / name, extra)
            self.assertFalse((self.root / name).exists())
        parser = make_parser()
        with self.assertRaisesRegex(ValueError, "Choose exactly one"):
            train(parser.parse_args(["train", *self.common(self.root / "neither")]))
        with self.assertRaisesRegex(ValueError, "Choose exactly one"):
            train(parser.parse_args(["train", "--fresh-training", "--resume-from", str(parent),
                                     "--additional-epochs", "1", *self.common(self.root / "both")]))
        with self.assertRaisesRegex(ValueError, "needs --additional-epochs"):
            train(parser.parse_args(["train", "--resume-from", str(parent), "--epochs", "3", *self.common(self.root / "e")]))
        with self.assertRaisesRegex(ValueError, "needs --additional-epochs"):
            train(parser.parse_args(["train", "--resume-from", str(parent), "--coco-weights", "x.pt",
                                     "--additional-epochs", "1", *self.common(self.root / "w")]))
        with self.assertRaisesRegex(ValueError, "apply only to --resume-from"):
            train(parser.parse_args(["train", "--fresh-training", "--coco-weights", "x.pt", "--weights-sha256", "a" * 64,
                                     "--additional-epochs", "1", *self.common(self.root / "mix")]))

    def test_unfinished_tampered_or_smoke_parents_are_refused(self):
        parent = self.root / "p"
        self.fresh(parent, 1)
        original = (parent / "status.json").read_text()
        (parent / "status.json").write_text(json.dumps({"state": "running"}))
        with self.assertRaisesRegex(ValueError, "completed non-smoke"):
            self.resume(parent, self.root / "a")
        (parent / "status.json").write_text(original)
        config = json.loads((parent / "config.json").read_text())
        config["seed"] += 1
        (parent / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.resume(parent, self.root / "b")
        self.assertFalse((self.root / "a").exists() or (self.root / "b").exists())

    def test_rng_snapshot_roundtrip_restores_all_streams(self):
        class Rows(torch.utils.data.Dataset):
            def __len__(self): return 8
            def __getitem__(self, index): return index
        loader = DataLoader(Rows(), batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(5))
        random.seed(1); np.random.seed(2); torch.manual_seed(3)
        list(loader)                                   # advance the shuffle stream
        snapshot = _rng_snapshot(loader, torch.device("cpu"))
        saved = torch.load  # round-trip through weights-only serialization as checkpoints do
        import io
        buffer = io.BytesIO()
        torch.save(snapshot, buffer)
        buffer.seek(0)
        snapshot = saved(buffer, weights_only=True)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)), [int(v) for v in list(loader)])
        random.seed(99); np.random.seed(99); torch.manual_seed(99)
        other = DataLoader(Rows(), batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(77))
        self.assertTrue(_restore_rng(snapshot, other, torch.device("cpu"), 0))
        actual = (random.random(), float(np.random.rand()), float(torch.rand(1)), [int(v) for v in list(other)])
        self.assertEqual(actual, expected)
        self.assertFalse(_restore_rng(None, other, torch.device("cpu"), 7))

if __name__ == "__main__":
    unittest.main()
