import unittest

import numpy as np
import torch

from disaster_lens.detection.masks import CompactMasks
from disaster_lens.data.footprints import MaskWindow
from disaster_lens.detection.crops import CropInstanceDataset, crop_target, crop_window
from disaster_lens.detection.instance_data import InstanceSegmentationSample


def window(x1, y1, x2, y2, repair=None):
    return MaskWindow((x1, y1, x2, y2), np.ones((y2 - y1, x2 - x1), dtype=np.uint8), repair)


def target(windows, size=(100, 100)):
    boxes = torch.tensor([list(map(float, w.bounds)) for w in windows])
    n = len(windows)
    return {"boxes": boxes, "labels": torch.ones(n, dtype=torch.int64), "image_id": torch.tensor([3]),
            "area": torch.tensor([w.area for w in windows], dtype=torch.float32), "iscrowd": torch.zeros(n, dtype=torch.int64),
            "compact_masks": CompactMasks(size, tuple(windows))}


class CropWindowTests(unittest.TestCase):
    def test_window_is_clipped_and_shifted_into_crop_coordinates(self):
        pixels = np.arange(1, 21, dtype=np.uint8).reshape(4, 5) % 2
        original = MaskWindow((10, 20, 15, 24), pixels, {"note": "kept"})
        cropped = crop_window(original, 12, 21, 40, 40)
        self.assertEqual(cropped.bounds, (0, 0, 3, 3))
        np.testing.assert_array_equal(cropped.pixels, pixels[1:4, 2:5])
        self.assertEqual(cropped.repair, {"note": "kept"})
        self.assertIsNone(crop_window(original, 15, 20, 30, 30))        # touching edges only
        self.assertIsNone(crop_window(original, 0, 0, 5, 5))

    def test_visibility_rule_keeps_mostly_visible_and_drops_slivers(self):
        windows = [window(10, 10, 20, 20),     # fully inside
                   window(45, 10, 55, 20),     # 50% visible (x 45..50): kept at >= .5
                   window(48, 30, 58, 40),     # 20% visible: dropped
                   window(70, 70, 80, 80)]     # outside
        cropped, kept = crop_target(target(windows), 0, 0, 50, .5)
        self.assertEqual(kept, [0, 1])
        self.assertEqual(cropped["compact_masks"].image_size, (50, 50))
        self.assertEqual(cropped["boxes"].tolist(), [[10., 10., 20., 20.], [45., 10., 50., 20.]])
        self.assertEqual(cropped["area"].tolist(), [100., 50.])
        self.assertEqual(len(cropped["compact_masks"]), 2)
        self.assertEqual(cropped["image_id"].tolist(), [3])
        # Offsets: box and mask move together.
        shifted, kept = crop_target(target(windows), 5, 8, 50, .5)
        self.assertEqual(shifted["boxes"][0].tolist(), [5., 2., 15., 12.])
        self.assertEqual(shifted["compact_masks"].windows[0].bounds, (5, 2, 15, 12))

    def test_empty_result_has_valid_empty_targets(self):
        cropped, kept = crop_target(target([window(80, 80, 90, 90)]), 0, 0, 40, .5)
        self.assertEqual(kept, [])
        self.assertEqual(cropped["boxes"].shape, (0, 4))
        self.assertEqual(cropped["labels"].shape, (0,))
        self.assertEqual(len(cropped["compact_masks"]), 0)


class FakeBase:
    mask_representation = "compact"

    def __init__(self):
        self.scenes = [{"height": 100, "width": 100}, {"height": 100, "width": 100}]
        self.windows = [window(10, 10, 20, 20), window(60, 60, 70, 70), window(80, 15, 95, 30)]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        image = torch.arange(3 * 100 * 100, dtype=torch.float32).reshape(3, 100, 100) + index
        return InstanceSegmentationSample(f"s{index}", image, target(self.windows), ["a", "b", "c"], [])


class CropDatasetTests(unittest.TestCase):
    def make(self, **kwargs):
        return CropInstanceDataset(FakeBase(), 50, 3, seed=7, **kwargs)

    def test_length_and_image_crop_matches_the_target_geometry(self):
        data = self.make()
        self.assertEqual(len(data), 6)
        for item in range(len(data)):
            sample = data[item]
            self.assertEqual(sample.image.shape, (3, 50, 50))
            self.assertEqual(len(sample.reference_uids), len(sample.target["boxes"]))
            # The crop is a true sub-image: its first pixel value identifies its origin in the source scene.
            base_index = item // 3
            origin = int(sample.image[0, 0, 0].item()) - base_index
            y0, x0 = divmod(origin, 100)
            for box, built in zip(sample.target["boxes"], sample.target["compact_masks"].windows, strict=True):
                self.assertEqual(tuple(int(v) for v in box.tolist()), built.bounds)
                self.assertTrue(0 <= box[0] < box[2] <= 50 and 0 <= box[1] < box[3] <= 50)
            self.assertTrue(0 <= x0 <= 50 and 0 <= y0 <= 50)

    def test_deterministic_per_epoch_and_different_across_epochs(self):
        a, b = self.make(), self.make()
        for item in range(len(a)):
            torch.testing.assert_close(a[item].image, b[item].image)
        first = [a[item].image[0, 0, 0].item() for item in range(len(a))]
        a.set_epoch(1)
        second = [a[item].image[0, 0, 0].item() for item in range(len(a))]
        self.assertNotEqual(first, second)
        b.set_epoch(1)
        self.assertEqual(second, [b[item].image[0, 0, 0].item() for item in range(len(b))])

    def test_empty_crops_are_redrawn_when_buildings_exist(self):
        data = CropInstanceDataset(FakeBase(), 36, 20, seed=1, resample_empty=8)
        empties = sum(len(data[i].target["boxes"]) == 0 for i in range(len(data)))
        none = CropInstanceDataset(FakeBase(), 36, 20, seed=1, resample_empty=0)
        empties_without = sum(len(none[i].target["boxes"]) == 0 for i in range(len(none)))
        self.assertLess(empties, empties_without)

    def test_validation_of_arguments(self):
        for kwargs in ({"crop_size": 16}, {"crops_per_scene": 0}, {"min_visible": 0.}, {"crop_size": 200}):
            full = {"crop_size": 50, "crops_per_scene": 1, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CropInstanceDataset(FakeBase(), full.pop("crop_size"), full.pop("crops_per_scene"), 0, **full)
        with self.assertRaises(IndexError):
            self.make()[99]


if __name__ == "__main__":
    unittest.main()
