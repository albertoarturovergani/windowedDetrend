import unittest
from dataclasses import replace
import numpy as np
from windowed_detrend import Config, detrend
from windowed_detrend.benchmark import synthetic

class PluginTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(2500)/10000
        self.gt,self.a=synthetic(self.t,.003,.2)
        self.cfg=Config(fill_mode='none', onset_s=.010,peak_s=.015,rise_model='singlerise',
                        decay_model='singledecay',constrain_rise=True)
    def test_default_lagrange_cubic(self):
        cfg=Config(fill_mode='none', peak_s=.015)
        self.assertEqual((cfg.rise_model,cfg.decay_model,cfg.decay_degree),('lagrange','poly',3))
        signal=(self.gt+self.a)[None,None,:]
        z,trend,info=detrend(signal,self.t,cfg)
        rise=(self.t>=cfg.onset_s)&(self.t<cfg.peak_s)
        decay=self.t>=cfg.peak_s
        np.testing.assert_array_equal(z[0,0,rise],0)
        np.testing.assert_array_equal(trend[0,0,rise],signal[0,0,rise])
        self.assertEqual(info['fits'][0]['rise']['degree'],int(rise.sum())-1)
        self.assertEqual(info['fits'][0]['decay']['model'],'poly')
        self.assertEqual(len(info['fits'][0]['decay']['coefficients']),4)
        np.testing.assert_allclose(np.diff(trend[0,0,decay],n=4),0,atol=1e-10)
        np.testing.assert_array_equal(z[0,0,self.t<cfg.onset_s],signal[0,0,self.t<cfg.onset_s])
        with self.assertRaises(ValueError):detrend(signal,self.t,replace(cfg,constrain_rise=True))
    def test_exact_artifact_and_identity(self):
        cfg=replace(self.cfg,tau_min_s=.003,tau_max_s=.2)
        data=self.a[None,None,:]
        z,a,_=detrend(data,self.t,cfg)
        np.testing.assert_allclose(z,0,atol=1e-9)
        np.testing.assert_allclose(z+a,data,atol=1e-12)
    def test_recovery_and_preservation(self):
        data=np.stack([self.gt+self.a,np.full(self.t.size,17.)])[None]
        original=data.copy()
        z,a,_=detrend(data,self.t,replace(self.cfg,channels=(0,),end_s=.2))
        np.testing.assert_array_equal(data,original)
        np.testing.assert_array_equal(z[:,1],data[:,1])
        np.testing.assert_array_equal(z[:,:,self.t<.010],data[:,:,self.t<.010])
        np.testing.assert_array_equal(z[:,:,self.t>.2],data[:,:,self.t>.2])
        mask=(self.t>=.010)&(self.t<=.2)
        self.assertLess(np.mean((z[0,0,mask]-self.gt[mask])**2),np.mean(self.a[mask]**2))
    def test_unit_scaling(self):
        data=(self.gt+self.a)[None,None,:]
        a,_,_=detrend(data,self.t,self.cfg);b,_,_=detrend(data*1e-6,self.t,self.cfg)
        np.testing.assert_allclose(b,a*1e-6,atol=1e-12)
    def test_auto_peak(self):
        _,_,info=detrend(self.a[None,None,:],self.t,replace(self.cfg,peak_s=None,peak_search_end_s=.02))
        self.assertAlmostEqual(info['fits'][0]['peak_s'],.015)
    def test_invalid_inputs(self):
        data=(self.gt+self.a)[None,None,:]
        for t,cfg in [(self.t[::-1],self.cfg),(self.t,replace(self.cfg,peak_s=.010)),(self.t,replace(self.cfg,channels=(5,)))]:
            with self.assertRaises(ValueError):detrend(data,t,cfg)
        data[0,0,0]=np.nan
        with self.assertRaises(ValueError):detrend(data,self.t,self.cfg)
    def test_models_and_baseline(self):
        data=(self.gt+self.a)[None,None,:]
        for model in ('biexp','doubledecay','alpha','poly'):
            cfg=replace(self.cfg,mode='global',global_model=model,constrain_rise=False,fit_offset=True,pre_degree=1)
            z,a,_=detrend(data,self.t,cfg)
            self.assertTrue(np.isfinite(z).all())
            np.testing.assert_allclose(z+a,data,atol=1e-12)

if __name__=='__main__': unittest.main()
