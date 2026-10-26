"""Analytic pixel-mask AP and window-RLE checks, including streaming masks."""

import unittest

import numpy as np

from disaster_lens.detection.mask_metrics import decode_window, encode_window, mask_iou_row, match_mask_scene
from disaster_lens.detection.box_metrics import summarize_localization
from disaster_lens.data.footprints import MaskWindow


def window(pixels, x=0, y=0):
    array = np.asarray(pixels, dtype=np.uint8)
    return MaskWindow((x, y, x + array.shape[1], y + array.shape[0]), array)


class InstanceMetricTests(unittest.TestCase):
    def test_window_rle_roundtrip_empty_full_holes_and_offsets(self) -> None:
        for original in (window([[0, 0], [0, 0]], 3, 2), window([[1, 1], [1, 1]]),
                         window([[1, 0, 1], [0, 1, 0]], 5, 7),
                         MaskWindow((0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8))):
            record = encode_window(original)
            decoded = decode_window(record, (20, 20))
            self.assertEqual(decoded.bounds, original.bounds)
            np.testing.assert_array_equal(decoded.pixels, original.pixels)
        self.assertEqual(encode_window(window([[1, 1]]))["counts"], [0, 2])

    def test_invalid_rle_is_rejected(self) -> None:
        record = encode_window(window([[1, 0], [0, 1]]))
        for updates in ({"counts": [999]}, {"order": "F"}, {"bounds": [-1, 0, 1, 2]},
                        {"counts": [-1, 5]}, {"size": [1, 4]}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                decode_window({**record, **updates}, (20, 20))
        with self.assertRaises(ValueError):
            decode_window(record, (20, 20), max_pixels=1)

    def test_identical_boxes_do_not_imply_identical_masks(self) -> None:
        reference = window([[1, 0], [0, 1]])
        opposite = window([[0, 1], [1, 0]])
        row = mask_iou_row(opposite, (reference,), np.array([2.]))
        self.assertEqual(row.tolist(), [0.])
        matches = match_mask_scene("scene", "event", [opposite], np.array([.9]), (reference,), max_detections=2500)
        metrics = summarize_localization([matches], max_detections=2500, score_floor=0., metric_kind="mask")
        self.assertEqual(metrics["pooled"]["ap_50_95"], 0.)
        self.assertEqual(metrics["definition"]["metric_kind"], "mask")

    def test_streamed_masks_are_decoded_once_in_score_order(self) -> None:
        reference = window([[1, 0], [0, 1]])
        predictions = [reference, window([[0, 1], [1, 0]])]
        decoded = []
        def stream(index):
            decoded.append(index)
            return predictions[index]
        matches = match_mask_scene("scene", "event", stream, np.array([.8, .9]), (reference,), max_detections=2500)
        self.assertEqual(decoded, [1, 0])
        metrics = summarize_localization([matches], max_detections=2500, score_floor=0., metric_kind="mask")
        self.assertEqual(metrics["pooled"]["ap_50_95"], .5)

    def test_fractional_iou_thresholds_and_window_offsets(self) -> None:
        reference = window([[1, 0], [0, 1]], 10, 20)
        prediction = window([[1, 1], [1, 1]], 10, 20)
        matches = match_mask_scene("scene", "event", [prediction], np.array([.9]), (reference,), max_detections=2500)
        metrics = summarize_localization([matches], max_detections=2500, score_floor=0., metric_kind="mask")
        self.assertAlmostEqual(metrics["pooled"]["ap_50_95"], .1)
        self.assertEqual(metrics["pooled"]["ap50"], 1.)
        disjoint = mask_iou_row(window([[1]], 100, 100), (reference,), np.array([2.]))
        self.assertEqual(disjoint.tolist(), [0.])

    def test_empty_predictions_and_absent_reference_support(self) -> None:
        reference = window([[1]])
        no_predictions = match_mask_scene("scene", "event", [], np.array([]), (reference,), max_detections=2500)
        self.assertEqual(summarize_localization([no_predictions], max_detections=2500, score_floor=0.,
                                               metric_kind="mask")["pooled"]["ap50"], 0.)
        negative = match_mask_scene("scene", "event", [reference], np.array([.9]), (), max_detections=2500)
        self.assertIsNone(summarize_localization([negative], max_detections=2500, score_floor=0.,
                                                metric_kind="mask")["pooled"]["ap50"])


if __name__ == "__main__":
    unittest.main()


class VerifiedExportTests(unittest.TestCase):
    def write(self, root, marker, tamper=False):
        import hashlib, json
        (root / "predictions.jsonl").write_text('{"scene_id": "s"}\n')
        digest = hashlib.sha256((root / "predictions.jsonl").read_bytes()).hexdigest()
        (root / "metrics.json").write_text(json.dumps({"export_verification": marker,
                                                       "provenance": {"predictions_sha256": digest}}))
        if tamper:
            (root / "predictions.jsonl").write_text('{"scene_id": "x"}\n')

    def test_accepts_untiled_and_tiled_verified_exports_and_rejects_others(self):
        import tempfile
        from pathlib import Path
        from disaster_lens.detection.mask_metrics import verified_export
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for marker in ("matched_checkpoint_metrics", "tiled_inference_not_comparable_to_checkpoint_metrics"):
                self.write(root, marker)
                metrics, path, digest = verified_export(root)
                self.assertEqual((metrics["export_verification"], path.name, len(digest)), (marker, "predictions.jsonl", 64))
            for marker, tamper in (("failed_checkpoint_metrics", False), ("matched_checkpoint_metrics", True)):
                self.write(root, marker, tamper)
                with self.subTest(marker=marker, tamper=tamper), self.assertRaises(ValueError):
                    verified_export(root)
