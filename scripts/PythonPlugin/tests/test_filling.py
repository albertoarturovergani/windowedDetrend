import unittest
from dataclasses import replace
import numpy as np
from windowed_detrend import Config, detrend
from windowed_detrend.filling import fill_offset, _fourier_resample


class FillingTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(-200, 501)/2000
        background = .7*np.sin(2*np.pi*11*self.t)+.2*np.cos(2*np.pi*23*self.t)
        self.data = np.stack([background, 2*background+1])[None]
        self.cfg = Config(peak_s=.015, fill_mode='resample', channels=(0,))

    def test_optional_fill_and_source_are_separate_from_fit(self):
        original = self.data.copy()
        raw, raw_trend, _ = detrend(self.data, self.t, replace(self.cfg, fill_mode='none'))
        filled, trend, info = detrend(self.data, self.t, self.cfg)
        rec = info['fits'][0]['fill']
        target = slice(rec['target_first'], rec['target_last']+1)
        source = slice(rec['source_first'], rec['source_last']+1)
        np.testing.assert_array_equal(trend, raw_trend)
        np.testing.assert_array_equal(self.data, original)
        np.testing.assert_array_equal(filled[:,1], self.data[:,1])
        expected = raw.copy()
        expected[0,0,target] = _fourier_resample(original[0,0,source], rec['target_count'])
        np.testing.assert_array_equal(filled, expected)
        self.assertFalse(np.allclose(filled[0,0,target], raw[0,0,target]))

    def test_zero_margin_off_sample_peak_and_fallback(self):
        rec = fill_offset(self.data[0,0], self.t, .010, .0153, 0)
        target = np.flatnonzero((self.t>=.010)&(self.t<.0153))
        self.assertEqual((rec['target_first'],rec['target_last']), (target[0],target[-1]))
        fallback = fill_offset(self.data[0,0], self.t, .010, .0153, .5)
        self.assertTrue(fallback['source_fallback'])
        self.assertEqual(fallback['target_count'],self.t.size)
        self.assertEqual(fallback['source_count'],int(np.sum(self.t<.010)))

    def test_pre_fit_does_not_change_filling_source(self):
        a, _, ai = detrend(self.data,self.t,self.cfg)
        b, _, bi = detrend(self.data,self.t,replace(self.cfg,pre_degree=1))
        np.testing.assert_array_equal(ai['fits'][0]['fill']['values'],bi['fits'][0]['fill']['values'])

    def test_constant_and_nyquist_resampling(self):
        for n in (1,2,3,8,9,32):
            for m in (1,2,3,7,8,31,32):
                np.testing.assert_allclose(_fourier_resample(np.full(n,2.5),m),2.5,atol=1e-14)
        np.testing.assert_allclose(_fourier_resample((-1.)**np.arange(8),16),
                                   np.cos(np.pi*np.arange(16)/2),atol=1e-14)

    def test_scipy_reference_when_available(self):
        try:
            from scipy.signal import resample
        except ImportError:
            self.skipTest('SciPy is an optional reference dependency')
        rng=np.random.default_rng(13)
        for n in (1,2,3,4,7,8,15,16,101):
            x=rng.normal(size=n)
            for m in (1,2,3,4,7,8,31,32):
                np.testing.assert_allclose(_fourier_resample(x,m),resample(x,m),atol=2e-13)

    def test_invalid_fill_requests(self):
        for cfg in (replace(self.cfg,fill_mode='bad'), replace(self.cfg,fill_margin_s=-1),
                    replace(self.cfg,fill_margin_s=np.nan),replace(self.cfg,mode='global')):
            with self.assertRaises(ValueError):
                detrend(self.data,self.t,cfg)
        t=self.t.copy(); t[10]+=.00001
        with self.assertRaisesRegex(ValueError,'uniform'):
            detrend(self.data,t,self.cfg)
        t=np.arange(100)/1000
        with self.assertRaisesRegex(ValueError,'pre-onset'):
            detrend(np.ones((1,1,100)),t,replace(self.cfg,onset_s=0,peak_s=.02))

    def test_gaussian_defaults_reproducibility_and_parameters(self):
        self.assertEqual(Config().fill_mode,'gaussian')
        cfg=replace(self.cfg,fill_mode='gaussian')
        a,trend,info=detrend(self.data,self.t,cfg)
        b,_,_=detrend(self.data,self.t,cfg)
        np.testing.assert_array_equal(a,b)
        c,_,ci=detrend(self.data,self.t,replace(cfg,fill_seed=43))
        self.assertFalse(np.array_equal(a,c))
        f=info['fits'][0]['fill']; source=self.data[0,0,f['source_first']:f['source_last']+1]
        self.assertEqual(f['mean'],float(np.mean(source)))
        self.assertEqual(f['std'],max(float(np.std(source))*.5,1e-8))
        expected=np.random.default_rng(np.random.SeedSequence([42,0,0])).normal(f['mean'],f['std'],f['target_count'])
        np.testing.assert_array_equal(f['values'],expected)
        both=replace(cfg,channels=(0,1))
        _,_,all_info=detrend(self.data,self.t,both)
        _,_,one_info=detrend(self.data,self.t,replace(cfg,channels=(1,)))
        np.testing.assert_array_equal(all_info['fits'][1]['fill']['values'],one_info['fits'][0]['fill']['values'])

    def test_gaussian_distribution_and_flat_source(self):
        t=np.arange(-1000,50001)/10000
        y=np.sin(2*np.pi*17*t)
        f=fill_offset(y,t,0,5,0,'gaussian')
        standardized=(np.array(f['values'])-f['mean'])/f['std']
        self.assertLess(abs(float(np.mean(standardized))),.03)
        self.assertLess(abs(float(np.std(standardized))-1),.03)
        flat=fill_offset(np.zeros_like(y),t,0,5,0,'gaussian')
        self.assertEqual(flat['std'],1e-8)


if __name__=='__main__':
    unittest.main()
