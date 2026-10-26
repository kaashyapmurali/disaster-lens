"""Deterministic stage-timer checks; no GPU or remote operations."""

import unittest
from unittest.mock import MagicMock, patch

from disaster_lens.profiling import StageTimer


class StageTimerTests(unittest.TestCase):
    def test_start_end_and_callback_visibility(self) -> None:
        states = []
        with patch("disaster_lens.profiling.time.perf_counter", side_effect=[10., 11., 14.]):
            timer = StageTimer(on_change=lambda: states.append(timer.records[-1]["state"]))
            with timer.measure("load"):
                self.assertEqual(timer.records[0]["state"], "running")
        self.assertEqual(states, ["running", "complete"])
        self.assertEqual(timer.records[0]["start_offset_seconds"], 1.)
        self.assertEqual(timer.records[0]["elapsed_seconds"], 3.)

    def test_failed_stage_preserved_and_exception_propagated(self) -> None:
        timer = StageTimer()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with timer.measure("failed"):
                raise RuntimeError("synthetic")
        self.assertEqual(timer.records[0]["state"], "failed")
        self.assertGreaterEqual(timer.records[0]["elapsed_seconds"], 0.)

    def test_empty_duplicate_names_rejected_without_callback(self) -> None:
        callback = MagicMock()
        timer = StageTimer(callback)
        with self.assertRaises(ValueError):
            with timer.measure(""):
                pass
        callback.assert_not_called()
        with timer.measure("one"):
            pass
        with self.assertRaises(ValueError):
            with timer.measure("one"):
                pass
        self.assertEqual(callback.call_count, 2)


if __name__ == "__main__":
    unittest.main()
