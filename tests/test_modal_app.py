"""Modal launcher: command builders (parsed by the real CLIs) and the remote stage runner (no Modal calls)."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_MODAL = importlib.util.find_spec("modal") is not None
if HAS_MODAL:
    import modal_app as app

SHA = "a" * 64


@unittest.skipUnless(HAS_MODAL, "Cloud dependency group required.")
class CommandTests(unittest.TestCase):
    def test_detector_commands_parse_with_the_trainer_cli(self):
        from disaster_lens.detection.train import make_parser
        fresh = make_parser().parse_args(app.detector_train_stages("det", SHA)[0][4:])
        self.assertTrue(fresh.fresh_training)
        self.assertEqual((fresh.epochs, fresh.weights_sha256, fresh.geometry_policy), (5, SHA, "make-valid-polygonal"))
        crops = make_parser().parse_args(app.detector_crops_stages("det-2x", "det")[0][4:])
        self.assertEqual((crops.resume_from, crops.crop_size, crops.validation_tile_size, crops.additional_epochs),
                         (app.RUNS / "det", 512, 512, 1))
        self.assertTrue(crops.allow_protocol_change)
        tiled = make_parser().parse_args(app.detector_export_stages("det-2x", tile_size=512)[0][4:])
        self.assertEqual((tiled.run_dir, tiled.tile_size), (app.RUNS / "det-2x", 512))
        self.assertIsNone(make_parser().parse_args(app.detector_export_stages("det")[0][4:]).tile_size)

    def test_damage_commands_cover_cached_and_live_training(self):
        features = app.damage_features_stages("vit-features", "dinov3-vitl")[0]
        self.assertEqual(features[3], "disaster_lens.damage.feature_cache")
        self.assertIn(str(app.DINOV3), features)
        self.assertNotIn("--encoder-dir", app.damage_features_stages("cnn-features", "resnet18")[0])
        train, export = app.damage_head_stages("cnn-head", "cnn-features")
        self.assertIn(str(app.CACHES / "cnn-features" / "train"), train)
        self.assertEqual(export[3], "disaster_lens.damage.export_cached")
        live_train, live_export = app.damage_live_stages("vit-ft", "dinov3-vitl", "paired", 4, init_head="vit-head", precision="bf16")
        self.assertEqual(live_train[3:5], ["disaster_lens.damage.train_live", "train"])
        self.assertEqual(live_train[live_train.index("--init-head") + 1], str(app.RUNS / "vit-head" / "best.pt"))
        self.assertEqual(live_train[live_train.index("--trainable-blocks") + 1], "4")
        self.assertEqual(live_export[3:5], ["disaster_lens.damage.train_live", "export"])

    def test_verifier_features_command(self):
        command = app.verifier_features_stages("vf-01", "det-2x")[0]
        self.assertEqual(command[3:5], ["disaster_lens.detection.verifier", "extract"])
        self.assertEqual(command[command.index("--export-dir") + 1], str(app.RUNS / "det-2x" / "validation_predictions_tile512"))
        self.assertEqual(command[command.index("--encoder-dir") + 1], str(app.DINOV3))
        with self.assertRaises(ValueError):
            app.verifier_features_stages("vf-01", "det-2x", split="train")

    def test_no_change_diagnostic_skips_the_reference_check(self):
        command = app.end_to_end_stages("nc", "vit-ft", "det-2x", split="test", no_change=True)[0]
        self.assertIn("--no-change", command)
        self.assertEqual(command.count("--skip-reference-check"), 1)

    def test_test_split_commands(self):
        from disaster_lens.detection.train import make_parser
        exported = make_parser().parse_args(app.detector_export_stages("det-2x", tile_size=512, split="test")[0][4:])
        self.assertEqual((exported.split, exported.index_dir), ("test", app.TEST_INDEX))
        command = app.end_to_end_stages("e2e-test", "vit-ft", "det-2x", split="test")[0]
        self.assertEqual(command[command.index("--manifest") + 1], str(app.TEST_MANIFEST))
        self.assertEqual(command[command.index("--export-dir") + 1], str(app.RUNS / "det-2x" / "test_predictions_tile512"))
        self.assertIn("--skip-reference-check", command)
        self.assertNotIn("--saved-predictions", command)
        with self.assertRaises(ValueError):
            app.end_to_end_stages("x", "a", "b", split="holdout")

    def test_end_to_end_command_and_name_rules(self):
        command = app.end_to_end_stages("e2e", "vit-ft", "det-2x", cached_dinov3_head=True)[0]
        self.assertEqual(command[3], "disaster_lens.end_to_end.run")
        self.assertEqual(command[command.index("--export-dir") + 1], str(app.RUNS / "det-2x" / "validation_predictions_tile512"))
        self.assertIn("--encoder-dir", command)
        self.assertNotIn("--saved-predictions", app.end_to_end_stages("e2e", "post-only", "det-2x", saved_predictions="")[0])
        for bad in ("", "../x", "models", "feature-caches", "a b"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                app.detector_export_stages(bad)

    def test_submit_without_confirmation_is_a_dry_run(self):
        with patch.object(app.run, "spawn") as spawn:
            app.submit("dry", [["echo", "hi"]], 60, confirm_run=False)
            spawn.assert_not_called()


@unittest.skipUnless(HAS_MODAL, "Cloud dependency group required.")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.runs = Path(temporary.name)
        self.commits = []

    def run_stages(self, name, stages, seconds=60):
        return app.run_stages(name, stages, seconds, runs=self.runs, commit=lambda: self.commits.append(1))

    def status(self, name):
        return json.loads((self.runs / f"{name}.status.json").read_text())

    def test_stages_run_in_order_with_logs_and_status(self):
        stages = [[sys.executable, "-c", "print('first')"], [sys.executable, "-c", "print('second')"]]
        result = self.run_stages("ok", stages)
        self.assertEqual((result["state"], result["completed_stages"]), ("complete", 2))
        self.assertEqual((self.runs / "ok.log").read_text().split(), ["first", "second"])
        self.assertTrue(self.commits)
        with self.assertRaises(FileExistsError):              # never overwrite or rerun
            self.run_stages("ok", stages)

    def test_failed_stage_stops_the_run_without_retry(self):
        stages = [[sys.executable, "-c", "import sys; sys.exit(3)"], [sys.executable, "-c", "print('not reached')"]]
        with self.assertRaisesRegex(RuntimeError, "status 3"):
            self.run_stages("fails", stages)
        self.assertEqual((self.status("fails")["state"], self.status("fails")["completed_stages"]), ("failed", 0))
        self.assertNotIn("not reached", (self.runs / "fails.log").read_text())

    def test_work_budget_kills_a_hung_stage(self):
        with self.assertRaisesRegex(TimeoutError, "work budget"):
            self.run_stages("slow", [[sys.executable, "-c", "import time; time.sleep(30)"]], seconds=1)
        self.assertEqual(self.status("slow")["state"], "failed")

    def test_existing_run_directory_is_refused(self):
        (self.runs / "taken").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_stages("taken", [[sys.executable, "-c", "pass"]])
        with self.assertRaises(ValueError):
            self.run_stages("bad-budget", [[sys.executable, "-c", "pass"]], seconds=0)


if __name__ == "__main__":
    unittest.main()
