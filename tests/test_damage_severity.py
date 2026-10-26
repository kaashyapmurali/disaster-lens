"""Ordinal severity summaries exported next to class probabilities."""

import unittest

import torch

from disaster_lens.damage.severity import SEVERITY_COLUMNS, severity_outputs


class SeverityTests(unittest.TestCase):
    def test_known_example_and_one_hot_extremes(self):
        probabilities = torch.tensor([[.1, .2, .6, .1]], dtype=torch.float64)
        torch.testing.assert_close(severity_outputs(probabilities),
                                   torch.tensor([[1.7 / 3, .9, .7, .1]], dtype=torch.float64))
        outputs = severity_outputs(torch.eye(4, dtype=torch.float64))
        torch.testing.assert_close(outputs[:, 0], torch.tensor([0., 1 / 3, 2 / 3, 1.], dtype=torch.float64))
        self.assertEqual(len(SEVERITY_COLUMNS), outputs.shape[1])

    def test_rejects_wrong_shapes(self):
        for bad in (torch.ones(4), torch.ones(2, 3)):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError):
                severity_outputs(bad)


if __name__ == "__main__":
    unittest.main()
