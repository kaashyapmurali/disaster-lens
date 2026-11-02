"""Drone alignment helpers on synthetic data (skipped without the `drone` dependency group)."""

import importlib.util
import unittest

import numpy as np

HAVE_DRONE = all(importlib.util.find_spec(m) for m in ("cv2", "rasterio", "scipy"))


@unittest.skipUnless(HAVE_DRONE, "needs `uv sync --group drone`")
class AlignmentTests(unittest.TestCase):
    def setUp(self):
        from disaster_lens.drone import alignment
        self.al = alignment
        self.rng = np.random.default_rng(0)

    def test_affine_fit_recovers_a_shift_despite_outliers(self):
        a = self.rng.uniform(0, 1000, (400, 2))
        b = a + [7.0, -4.0] + self.rng.normal(0, 0.5, a.shape)
        b[:80] = self.rng.uniform(0, 1000, (80, 2))            # 20% wrong matches
        A, inliers = self.al.fit_affine(a, b)
        np.testing.assert_allclose(self.al.apply_affine(A, np.array([[500.0, 500.0]])), [[507.0, 496.0]], atol=0.5)
        self.assertGreater(inliers[80:].mean(), 0.95)
        self.assertLess(inliers[:80].mean(), 0.1)

    def test_local_consistency_rejects_isolated_outliers(self):
        a = self.rng.uniform(0, 500, (300, 2))
        r = np.full((300, 2), 2.0) + self.rng.normal(0, 0.3, (300, 2))
        r[:15] += 15.0                                            # outliers far from their neighbours' residual
        keep = self.al.consistent(a, r)
        self.assertFalse(keep[:15].any())
        self.assertGreater(keep[15:].mean(), 0.95)

    def test_local_field_recovers_a_smooth_offset(self):
        a = self.rng.uniform(0, 400, (2000, 2))
        offset = lambda p: np.c_[1.0 + p[:, 0] / 400, -0.5 + 0 * p[:, 1]]   # varies slowly in x
        q = np.array([[100.0, 200.0], [300.0, 200.0]])
        np.testing.assert_allclose(self.al.local_field(a, offset(a), q, sigma_px=20), offset(q), atol=0.1)

    def test_warp_with_a_constant_field_moves_the_image(self):
        img = np.zeros((64, 64, 3), np.uint8)
        img[20:30, 20:30] = 255
        field = np.zeros((64, 64, 2), np.float32)
        field[..., 0] = 5                                        # sample 5 px to the right: content moves left
        out = self.al.warp(img, field)
        self.assertEqual(out[25, 17, 0], 255)
        self.assertEqual(out[25, 27, 0], 0)

    def test_nearest_field_assigns_each_pixel_its_closest_building(self):
        field = self.al.nearest_field(np.array([[10.0, 10.0], [50.0, 50.0]]), np.array([[1.0, 2.0], [-3.0, 4.0]]), (64, 64))
        np.testing.assert_allclose(field[8, 8], [1.0, 2.0])
        np.testing.assert_allclose(field[52, 52], [-3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
