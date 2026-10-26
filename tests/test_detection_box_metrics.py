"""Analytically check single-class AP, matching, empty scenes, and event pools."""

import unittest

import numpy as np

from disaster_lens.detection.box_metrics import match_scene, metrics_agree, summarize_localization


BOX = [0., 0., 10., 10.]


def scene(key: str, boxes: list, scores: list, references: list, event: str = "event", limit: int = 2500):
    return match_scene(key, event, np.array(boxes).reshape(-1, 4), np.array(scores),
                       np.array(references).reshape(-1, 4), max_detections=limit)


def report(scenes, limit: int = 2500, floor: float = 0.):
    return summarize_localization(scenes, max_detections=limit, score_floor=floor)


class LocalizationMetricTests(unittest.TestCase):
    def test_export_comparison_allows_only_tiny_float_roundoff(self) -> None:
        expected = {"count": 32337, "ap": .3427, "nested": [None, .5]}
        self.assertTrue(metrics_agree({**expected, "ap": .3427 + 1e-14}, expected))
        self.assertFalse(metrics_agree({**expected, "ap": .3428}, expected))
        self.assertFalse(metrics_agree({**expected, "count": 32338}, expected))
        self.assertFalse(metrics_agree({**expected, "count": 32337.0}, expected))
        self.assertFalse(metrics_agree({**expected, "nested": [.5]}, expected))
        self.assertFalse(metrics_agree(float("nan"), float("nan")))
        self.assertFalse(metrics_agree(float("inf"), float("inf")))

    def test_perfect_predictions_and_late_duplicate(self) -> None:
        matches = scene("a", [BOX, BOX], [.9, .8], [BOX])
        np.testing.assert_array_equal(matches.matched_indices[:, 0], np.zeros(10))
        np.testing.assert_array_equal(matches.matched_indices[:, 1], -np.ones(10))
        pooled = report([matches])["pooled"]
        self.assertEqual(pooled["ap_50_95"], 1.)
        self.assertEqual(pooled["at_score_threshold_iou50"]["true_positives"], 1)
        self.assertEqual(pooled["at_score_threshold_iou50"]["false_positives"], 1)
        self.assertEqual(pooled["at_score_threshold_iou50"]["precision"], .5)

    def test_high_score_false_positive_reduces_ap_to_one_half(self) -> None:
        matches = scene("a", [[20., 20., 30., 30.], BOX], [.9, .8], [BOX])
        pooled = report([matches])["pooled"]
        self.assertEqual(pooled["ap50"], .5)
        self.assertEqual(pooled["mean_recall_50_95"], 1.)

    def test_iou_thresholds_and_101_point_interpolation(self) -> None:
        # IoU=.6: TP at .50/.55/.60, FP at higher thresholds.
        matches = scene("a", [[0., 0., 6., 10.]], [.9], [BOX])
        pooled = report([matches])["pooled"]
        self.assertEqual(pooled["ap50"], 1.)
        self.assertEqual(pooled["ap75"], 0.)
        self.assertAlmostEqual(pooled["ap_50_95"], .3)
        pooled = report([scene("a", [BOX], [.9], [BOX, [20., 20., 30., 30.]])])["pooled"]
        self.assertAlmostEqual(pooled["ap50"], 51 / 101)
        self.assertEqual(pooled["recall_by_iou"]["0.50"], .5)

    def test_matching_uses_best_available_not_already_claimed_gt(self) -> None:
        matches = scene("a", [BOX, [0., 0., 11., 10.]], [.9, .8], [BOX, [2., 0., 12., 10.]])
        self.assertEqual(matches.matched_indices[0].tolist(), [0, 1])
        self.assertEqual(matches.matched_indices[-1].tolist(), [0, -1])

    def test_no_predictions_and_no_reference_scenes(self) -> None:
        missed = report([scene("a", [], [], [BOX])])["pooled"]
        self.assertEqual(missed["ap50"], 0.)
        self.assertEqual(missed["at_score_threshold_iou50"]["false_negatives"], 1)
        negative = report([scene("a", [BOX], [.9], [])])["pooled"]
        self.assertIsNone(negative["ap50"])
        self.assertIsNone(negative["mean_recall_50_95"])
        self.assertEqual(negative["at_score_threshold_iou50"]["false_positives"], 1)
        empty = report([scene("a", [], [], [])])["pooled"]
        self.assertIsNone(empty["ap50"])
        self.assertEqual(empty["detections"], 0)

    def test_negative_scene_false_positive_counts_in_pooled_ap(self) -> None:
        pooled = report([scene("negative", [BOX], [.99], []), scene("positive", [BOX], [.9], [BOX])])["pooled"]
        self.assertEqual(pooled["ap50"], .5)

    def test_caps_sort_scores_and_ties_are_deterministic(self) -> None:
        matches = scene("a", [BOX, [20., 20., 30., 30.]], [.8, .9], [BOX], limit=1)
        self.assertEqual(matches.order.tolist(), [1])
        self.assertEqual(report([matches], limit=1)["pooled"]["ap50"], 0.)
        tied = scene("a", [BOX, [20., 20., 30., 30.]], [.9, .9], [BOX], limit=1)
        self.assertEqual(tied.order.tolist(), [0])
        self.assertEqual(report([tied], limit=1)["pooled"]["ap50"], 1.)

    def test_size_recall_support_boundaries_and_operating_threshold(self) -> None:
        refs = [BOX, [100., 100., 132., 132.], [200., 200., 296., 296.]]
        pooled = report([scene("a", refs[:2], [.9, .1], refs)])["pooled"]
        sizes = pooled["size_recall_at_output_limit"]
        self.assertEqual([sizes[name]["support"] for name in ("small", "medium", "large")], [1, 1, 1])
        self.assertEqual(sizes["medium"]["recall_iou50"], 1.)
        self.assertEqual(sizes["large"]["recall_iou50"], 0.)
        self.assertEqual(pooled["at_score_threshold_iou50"]["recall"], 1 / 3)
        self.assertEqual(pooled["recall_by_iou"]["0.50"], 2 / 3)

    def test_event_metrics_are_independent_pools_not_the_primary_average(self) -> None:
        scenes = [scene("a", [BOX], [.9], [BOX], event="success"),
                  scene("b", [], [], [BOX], event="failure")]
        metrics = report(scenes)
        self.assertEqual(metrics["per_event"]["success"]["ap50"], 1.)
        self.assertEqual(metrics["per_event"]["failure"]["ap50"], 0.)
        self.assertAlmostEqual(metrics["pooled"]["ap50"], 51 / 101)
        self.assertEqual(metrics, report(list(reversed(scenes))))

    def test_invalid_boxes_scores_and_duplicate_scenes_rejected(self) -> None:
        for boxes, scores in (([BOX], [float("nan")]), ([BOX], [1.1]), ([[1., 1., 1., 2.]], [.9]),
                              ([[-1., 0., 1., 1.]], [.9])):
            with self.subTest(boxes=boxes, scores=scores), self.assertRaises(ValueError):
                scene("a", boxes, scores, [BOX])
        matches = scene("a", [BOX], [.2], [BOX])
        with self.assertRaises(ValueError):
            report([matches, matches])
        with self.assertRaises(ValueError):
            report([matches], floor=.3)
        with self.assertRaises(ValueError):
            report([], limit=1)


if __name__ == "__main__":
    unittest.main()
