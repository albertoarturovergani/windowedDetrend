import unittest
from dataclasses import replace
import numpy as np
from windowed_detrend import Config,detrend


class TepAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(-100,201)/1000
        u=np.maximum(self.t,0)
        self.artifact=-np.minimum(u/.01,1)*np.exp(-np.maximum(u-.01,0)/.05)

    def test_published_defaults(self):
        cfg=Config()
        self.assertEqual(cfg.pre_degree,2)
        self.assertFalse(cfg.constrain_decay)
        self.assertEqual((cfg.onset_s,cfg.peak_search_end_s,cfg.slope_threshold), (0,.020,.5))
        self.assertEqual((cfg.mask_convention,cfg.channel_policy),('pipeline','auto'))
        self.assertFalse(cfg.only_offset_channels)
        self.assertEqual(cfg.overall_degree,1)

    def test_quadratic_signal_removed_in_all_three_windows(self):
        y=(2+3*self.t+4*self.t**2)[None,None]
        cfg=Config(channel_policy='all',peak_s=.010,fill_mode='none')
        cleaned,_,info=detrend(y,self.t,cfg)
        np.testing.assert_allclose(cleaned,0,atol=3e-14)
        r=info['fits'][0];i0=np.flatnonzero(self.t==0)[0];ip=np.flatnonzero(self.t==.010)[0]
        self.assertEqual((r['windows']['pre']['last'],r['windows']['rise']['first']),(i0,i0))
        self.assertEqual((r['windows']['rise']['last'],r['windows']['decay']['first']),(ip,ip+1))
        self.assertIsNone(r['decay']['anchor'])

    def test_automatic_offset_selection_and_channel_degrees(self):
        data=(np.array([40,2,2,2,2,2])[:,None]*self.artifact)[None]
        _,_,info=detrend(data,self.t,Config(fill_mode='none'))
        self.assertEqual(info['channel_selection']['offset_channels'],[0])
        self.assertEqual([r['decay']['degree'] for r in info['fits']],[3,2,2,2,2,2])

    def test_manual_only_offset_channels_preserves_other_input(self):
        data=np.stack([10000*self.t**3,10000*self.t**3])[None]
        cfg=Config(channel_policy='manual',offset_channels=(0,),only_offset_channels=True,
                   peak_s=.010,fill_mode='none')
        cleaned,_,info=detrend(data,self.t,cfg)
        np.testing.assert_array_equal(cleaned[:,1],data[:,1])
        np.testing.assert_allclose(cleaned[0,0,self.t>.010],0,atol=1e-12)
        self.assertEqual(len(info['fits']),1)

    def test_no_offset_channels_use_whole_epoch_linear_fallback(self):
        data=np.stack([np.tile(self.artifact,(3,1)),np.tile(-self.artifact,(3,1))])
        cfg=Config()
        cleaned,_,info=detrend(data,self.t,cfg)
        self.assertEqual(info['operation'],'overall')
        self.assertEqual(info['channel_selection']['offset_channels'],[])
        for e in range(2):
            for c in range(3):
                expected=data[e,c]-np.polyval(np.polyfit(self.t,data[e,c],1),self.t)
                np.testing.assert_allclose(cleaned[e,c],expected,atol=1e-14)
        self.assertTrue(all('fill' not in r for r in info['fits']))
        unchanged,_,info=detrend(data,self.t,replace(cfg,overall_degree=None))
        np.testing.assert_array_equal(unchanged,data)
        self.assertEqual(info['operation'],'none')

    def test_off_grid_split_and_inclusive_fill_bounds(self):
        cfg=Config(channel_policy='all',peak_s=.0104,fill_margin_s=0,fill_mode='resample')
        _,_,info=detrend(self.artifact[None,None],self.t,cfg)
        r=info['fits'][0];self.assertEqual(r['peak_s'],.010)
        self.assertEqual(r['fill']['source_last'],r['windows']['pre']['last'])
        self.assertEqual(r['fill']['target_last'],r['windows']['rise']['last'])

    def test_invalid_selection_settings(self):
        data=self.artifact[None,None]
        for cfg in (Config(channel_policy='bad'),Config(mask_convention='bad'),
                    Config(offset_channels=(2,)),Config(slope_threshold=-1)):
            with self.assertRaises(ValueError):detrend(data,self.t,cfg)


if __name__=='__main__':unittest.main()
