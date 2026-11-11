import unittest
import numpy as np

from windowed_detrend import Config, detrend
from windowed_detrend.scalp_benchmark import case_definitions, scalp_layout, scalp_signal


class ScalpBenchmarkTests(unittest.TestCase):
    def test_fixture_geometry_and_noise(self):
        labels, xyz, _, _ = scalp_layout()
        self.assertEqual(len(set(labels)), 64)
        self.assertIn('Cz', labels)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), .095)
        cases = case_definitions()
        self.assertEqual(len(cases), 214)
        self.assertEqual(sum(c[5] == 'grid' for c in cases), 200)
        case_id = next(i for i, case in enumerate(cases) if case[3] == 2.)
        t, y, truth, artifact = scalp_signal(cases[case_id], case_id)
        self.assertEqual(y.shape, (64, 2500))
        np.testing.assert_array_equal(y, scalp_signal(cases[case_id], case_id)[1])
        self.assertFalse(np.array_equal(y, scalp_signal(cases[case_id], case_id, seed=43)[1]))
        self.assertEqual(np.unique(y, axis=0).shape[0], 64)
        noise = y-truth-artifact
        self.assertLess(abs(np.corrcoef(noise[:2])[0, 1]), .1)
        self.assertAlmostEqual(float(noise.std()), 2., delta=.03)

    def test_multichannel_batch_matches_single_channel_and_preserves_input(self):
        t, y, _, _ = scalp_signal(('windowed', .003, .2, .5, 0, 'extreme'), 0)
        original = y.copy()
        cfg = Config(peak_s=.015)
        batch, trend, info = detrend(y[None], t, cfg)
        self.assertEqual(len(info['fits']), 64)
        np.testing.assert_array_equal(y, original)
        np.testing.assert_allclose(batch+trend, y[None], atol=1e-12)
        np.testing.assert_array_equal(batch[:, :, t<.01], y[None][:, :, t<.01])
        for ch in (0, 31, 63):
            single, _, _ = detrend(y[None, ch:ch+1], t, cfg)
            np.testing.assert_array_equal(batch[:, ch:ch+1], single)


if __name__ == '__main__':
    unittest.main()
