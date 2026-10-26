"""Tiling/merging against a deterministic fake detector, plus a real tiny Mask R-CNN on CPU."""

import unittest

import torch

from disaster_lens.detection.tiling import TiledInstancePredictor, TilingConfig, tile_starts


class SquareDetector(torch.nn.Module):
    """Finds bright axis-aligned squares by thresholding; boxes are clipped to the tile it is given."""

    def forward(self, images):
        results = []
        for image in images:
            mask = (image[0] > .5).numpy()
            boxes = []
            from scipy import ndimage
            labelled, count = ndimage.label(mask)
            for slices in ndimage.find_objects(labelled):
                boxes.append([slices[1].start, slices[0].start, slices[1].stop, slices[0].stop])
            boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            results.append({"boxes": boxes, "scores": torch.full((len(boxes),), .9), "labels": torch.ones(len(boxes), dtype=torch.int64),
                            "mask_roi_probs": torch.full((len(boxes), 1, 28, 28), .75), "mask_image_size": tuple(image.shape[-2:])})
        return results


def scene(squares, size=(300, 400)):
    image = torch.zeros(3, *size)
    for x, y, w, h in squares:
        image[:, y:y + h, x:x + w] = 1.
    return image


class TileGridTests(unittest.TestCase):
    def test_tile_starts_cover_the_image_with_at_least_the_requested_overlap(self):
        self.assertEqual(tile_starts(1024, 512, 128), ([0, 256, 512], 512, 256))
        self.assertEqual(tile_starts(1024, 640, 128), ([0, 384], 640, 256))
        self.assertEqual(tile_starts(300, 512, 128), ([0], 300, 300))       # smaller than a tile: one full tile
        for length in range(200, 1100, 37):
            for tile, overlap in ((128, 32), (256, 64), (200, 100)):
                starts, used, actual = tile_starts(length, tile, overlap)
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1] + used, length)                  # last tile ends exactly at the border
                self.assertGreaterEqual(actual, overlap if length > tile else 0)
                self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))

    def test_config_validation(self):
        for kwargs in ({"tile_size": 16}, {"min_overlap": 600}, {"min_overlap": 3}, {"nms_threshold": 0},
                       {"tile_batch_size": 0}, {"edge_margin": -1.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TilingConfig(**kwargs)


class MergeTests(unittest.TestCase):
    def predictor(self, **kwargs):
        config = TilingConfig(tile_size=128, min_overlap=48, tile_batch_size=2, **kwargs)
        return TiledInstancePredictor(SquareDetector(), config).eval()

    def boxes_of(self, output):
        return sorted(map(tuple, output["boxes"].tolist()))

    def test_every_building_is_found_exactly_once_wherever_it_falls(self):
        # Squares up to overlap - 2*margin (44 px) must always be recovered whole and once, including at tile borders.
        generator = torch.Generator().manual_seed(0)
        for trial in range(30):
            squares, occupied = [], torch.zeros(300, 400, dtype=torch.bool)
            for _ in range(25):
                w, h = (int(v) for v in torch.randint(6, 40, (2,), generator=generator))
                x = int(torch.randint(1, 400 - w - 1, (1,), generator=generator))
                y = int(torch.randint(1, 300 - h - 1, (1,), generator=generator))
                if not occupied[max(y - 2, 0):y + h + 2, max(x - 2, 0):x + w + 2].any():       # keep them separated
                    occupied[y:y + h, x:x + w] = True
                    squares.append((x, y, w, h))
            output = self.predictor()([scene(squares)])[0]
            self.assertEqual(self.boxes_of(output), sorted((float(x), float(y), float(x + w), float(y + h)) for x, y, w, h in squares),
                             msg=f"trial {trial}")

    def test_output_format_matches_the_model_and_is_score_sorted_and_capped(self):
        output = self.predictor(detections_per_image=3)([scene([(10 + 30 * i, 20, 10, 10) for i in range(8)])])[0]
        self.assertEqual(len(output["boxes"]), 3)
        self.assertEqual(output["mask_roi_probs"].shape, (3, 1, 28, 28))
        self.assertEqual(output["mask_image_size"], (300, 400))
        self.assertEqual(output["labels"].dtype, torch.int64)
        self.assertTrue((output["scores"][:-1] >= output["scores"][1:]).all())
        self.assertNotIn("masks", output)

    def test_building_larger_than_the_overlap_is_kept_not_dropped(self):
        output = self.predictor()([scene([(60, 100, 90, 20)])])[0]    # 90 px wide > 48 px overlap
        self.assertGreaterEqual(len(output["boxes"]), 1)
        self.assertGreater(output["boxes"][:, 2].max() - output["boxes"][:, 0].min(), 60)

    def test_empty_scene_and_single_tile_equivalence(self):
        empty = self.predictor()([torch.zeros(3, 300, 400)])[0]
        self.assertEqual(empty["boxes"].shape, (0, 4))
        self.assertEqual(empty["mask_roi_probs"].shape, (0, 1, 28, 28))
        image = scene([(20, 30, 15, 12), (100, 60, 20, 20)], size=(100, 150))
        tiled = TiledInstancePredictor(SquareDetector(), TilingConfig(tile_size=512)).eval()([image])[0]
        direct = SquareDetector()([image])[0]
        self.assertEqual(self.boxes_of(tiled), self.boxes_of(direct))

    def test_training_mode_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.predictor().train()([scene([])])


class RealModelTests(unittest.TestCase):
    def test_tiny_mask_rcnn_tiles_are_upscaled_and_outputs_are_scene_valid(self):
        from disaster_lens.detection.model import DetectorConfig, build_detector
        torch.manual_seed(0)
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        config = DetectorConfig(
            min_size=96, max_size=96, anchor_sizes=(8, 16, 32, 64, 128), rpn_pre_nms_top_n=64, rpn_post_nms_top_n=32,
            detections_per_image=16, rpn_batch_size=16, box_batch_size=16, score_threshold=0.0, mask_representation="compact")
        model = build_detector(config).eval()
        predictor = TiledInstancePredictor(model, TilingConfig(tile_size=48, min_overlap=16, tile_batch_size=2, score_threshold=0.0)).eval()
        image = torch.rand(3, 96, 80)
        with torch.inference_mode():
            output = predictor([image])[0]
            tile_direct = model([image[:, :48, :48]])[0]
        self.assertEqual(tuple(tile_direct["mask_image_size"]), (48, 48))      # boxes come back in tile coordinates
        self.assertEqual(output["mask_image_size"], (96, 80))
        if len(output["boxes"]):
            self.assertTrue((output["boxes"][:, [0, 2]] <= 80).all() and (output["boxes"][:, [1, 3]] <= 96).all())
            self.assertTrue((output["boxes"][:, 2:] > output["boxes"][:, :2]).all())
            self.assertEqual(output["mask_roi_probs"].shape[1:], (1, 28, 28))


if __name__ == "__main__":
    unittest.main()
