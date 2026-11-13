"""Compare the plugin with the published TEP masks, screening and detrending.

Python 3.12+, NumPy, SciPy and pandas are needed for reference execution.
The core itself requires only NumPy. Run MATLAB run_tep_validation afterwards.
"""
from pathlib import Path
import argparse
from dataclasses import asdict, replace
import contextlib
import io
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.stats import linregress
from scipy.spatial.distance import euclidean

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from windowed_detrend import Config,detrend
from paper_reference import tep_functions as reference


class Epochs:
    """Storage-only stand-in: the original numerical functions are unchanged."""
    def __init__(self,data,times):
        self._data=np.array(data,copy=True);self.times=np.array(times,copy=True)
        self.ch_names=[f'CH{c}' for c in range(data.shape[1])]
        self.info=dict(ch_names=self.ch_names,chs=[dict(loc=np.array([c,0.,1.])) for c in range(data.shape[1])])
    def get_data(self):return self._data
    def copy(self):return Epochs(self._data,self.times)


def matlab_config(cfg):
    result={k:np.empty((0,0)) if v is None else float(v) if isinstance(v,int) and not isinstance(v,bool) else v
            for k,v in asdict(cfg).items()}
    for key in ('channels','offset_channels'):
        value=getattr(cfg,key)
        result[key]=np.empty((0,0)) if value is None else np.asarray(value,dtype=float)+1
    return result


def parameters(cfg):
    return dict(detrend_minTimeWindowOffset=cfg.onset_s,detrend_maxTimeWindowOffset=cfg.peak_search_end_s,
                detrend_polOrder_preOffset=cfg.pre_degree,detrendExtremeTechinque='max',
                detrend_offsetStart=True,detrend_noise_seed=cfg.fill_seed,seedChans=['CH0'])


def setup_reference():
    reference.np=np;reference.pd=pd;reference.os=os
    reference.tqdm=lambda values:values
    reference.linregress=linregress;reference.euclidean=euclidean
    reference.supported_models=['Gaussian']
    captured=[]
    reference.plot_detrend_example_v3=lambda **kw:captured.append(kw)
    return captured


def original_offsets(epochs,cfg,pars):
    # Original computeSlopes_v4; same pandas aggregation and condition as
    # check_detrend_need lines 2925-2930 in the published commit.
    df=reference.computeSlopes_v4(epochs,pars,'','synthetic')
    means=df.groupby(['id_twindow','chan'],as_index=False)['Zslope'].mean()
    outliers=means[np.abs(means['Zslope'])>=cfg.slope_threshold]
    offsets=[i for i,name in enumerate(epochs.ch_names) if name in outliers['chan'].unique()]
    slopes=np.empty((epochs.get_data().shape[0],len(epochs.ch_names),3))
    mean_z=np.empty((len(epochs.ch_names),3))
    for k,label in enumerate(('preOffset','offset','postOffset')):
        for c,name in enumerate(epochs.ch_names):
            slopes[:,c,k]=df[(df.id_twindow==label)&(df.chan==name)].sort_values('id_trial').slope
            mean_z[c,k]=means[(means.id_twindow==label)&(means.chan==name)].Zslope.iloc[0]
    return offsets,slopes,mean_z


def fixture_data(fs,identical=False):
    times=np.arange(round(-.15*fs),round(.25*fs))/fs
    onset=0.;peak=.008;u=np.maximum(times-onset,0)
    artifact=-(1-np.exp(-u/.002))
    tail=times>=peak
    artifact[tail]=-(1-np.exp(-peak/.002))*np.exp(-(times[tail]-peak)/.08)
    common=0.2*np.sin(2*np.pi*11*times)
    neural=np.exp(-25*u)*np.sin(2*np.pi*40*u)
    gains=np.array([40.,2.,2.,2.,2.,2.])
    if identical:
        data=np.stack([np.tile(common+neural+10*artifact,(6,1)),np.tile(-(common+neural+10*artifact),(6,1))])
    else:
        data=np.stack([gains[:,None]*artifact+neural+sign*common for sign in (-1,1)])
    return data,times


def validate(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    captured=setup_reference();fixtures=[];details=[]
    specs=[]
    for fs in (1000,1024):
        for mode in ('none','gaussian','resample'):
            specs.append((f'auto_{fs}_{mode}',fs,Config(fill_mode=mode),False))
    for mode in ('none','gaussian','resample'):
        specs.append((f'manual_constrained_{mode}',1024,Config(channel_policy='manual',offset_channels=(0,),constrain_decay=True,fill_mode=mode),False))
    specs.append(('offset_channels_only',1000,Config(channel_policy='manual',offset_channels=(0,),only_offset_channels=True),False))
    specs.append(('no_offsets_overall',1024,Config(),True))
    specs.append(('no_offsets_preserve',1000,Config(overall_degree=None),True))
    max_difference=0.;max_outside_rise=0.;max_slope=0.;max_mean_z=0.;max_fill=0.
    offset_count=0;other_count=0;warning_count=0
    for name,fs,cfg,identical in specs:
        data,times=fixture_data(fs,identical);epochs=Epochs(data,times);pars=parameters(cfg)
        cleaned,trend,info=detrend(data,times,cfg)
        if cfg.channel_policy=='auto':
            offsets,slopes,mean_z=original_offsets(epochs,cfg,pars)
            assert offsets==info['channel_selection']['offset_channels'],name
            max_slope=max(max_slope,float(np.max(np.abs(slopes-np.asarray(info['channel_selection']['slopes'])))))
            max_mean_z=max(max_mean_z,float(np.max(np.abs(mean_z-np.asarray(info['channel_selection']['mean_z'])))))
        else:
            offsets=list(cfg.offset_channels)
        original_dir=out/'reference_tables'/f'{len(fixtures):02d}';(original_dir/'3.detrend').mkdir(parents=True,exist_ok=True)
        captured.clear()
        with contextlib.redirect_stdout(io.StringIO()),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if offsets:
                result,_,_=reference.computeDetrend_v6(epochs,pars,str(original_dir),'synthetic',
                    typeOffsetRise='wind_poly_lagrange',typeOffsetDecay='wind_poly_3',
                    fitConstraint=cfg.constrain_decay,correctMode=False if cfg.fill_mode=='none' else cfg.fill_mode,
                    oddSamples=cfg.fill_margin_s*1000,offsetChans=[epochs.ch_names[c] for c in offsets],
                    lag_correction=False,doDetrendOnlyOffsetChans=cfg.only_offset_channels)
                expected=result.get_data()
            else:
                # Original run_detrend_pipeline's nonlinear_detrend fallback:
                # sample-index polynomial on the complete epoch, no filling.
                expected=data.copy()
                if cfg.overall_degree is not None:
                    x=np.arange(len(times))
                    for e in range(data.shape[0]):
                        for c in range(data.shape[1]):
                            expected[e,c]-=np.polyval(np.polyfit(x,data[e,c],cfg.overall_degree),x)
        warning_count+=len(caught)
        outside_rise=np.ones_like(data,dtype=bool);fill_bounds=[];mask_bounds=[];degree_rows=[];noise_parameters=[]
        reference_by_key={(x['epoch_idx'],epochs.ch_names.index(x['chan'])):x for x in captured}
        for rec in info['fits']:
            e,c=rec['epoch'],rec['channel']
            if 'rise' not in rec:continue
            raw=reference_by_key[(e,c)];masks=raw['timeMask']
            outside_rise[e,c,masks[1]]=False
            bounds=[]
            for label,mask in zip(('pre','rise','decay'),masks):
                ids=np.flatnonzero(mask);ours=rec['windows'][label]
                assert (ours['first'],ours['last'],ours['count'])==(int(ids[0]),int(ids[-1]),int(ids.size)),(name,label)
                bounds.extend([int(ids[0])+1,int(ids[-1])+1])
            mask_bounds.append([e+1,c+1,*bounds])
            assert rec['decay']['degree']==raw['OPTPARS_C'],name
            degree_rows.append([e+1,c+1,rec['decay']['degree']])
            offset_count+=int(c in offsets);other_count+=int(c not in offsets)
            if 'fill' in rec:
                f=rec['fill'];target=slice(f['target_first'],f['target_last']+1)
                max_fill=max(max_fill,float(np.max(np.abs(cleaned[e,c,target]-expected[e,c,target]))))
                fill_bounds.append([e+1,c+1,f['source_first']+1,f['source_last']+1,f['target_first']+1,f['target_last']+1,int(f['source_fallback'])])
                noise_parameters.append([f.get('mean',0.),f.get('std',0.)])
        difference=float(np.max(np.abs(cleaned-expected)))
        outside=float(np.max(np.abs(cleaned[outside_rise]-expected[outside_rise])))
        max_difference=max(max_difference,difference);max_outside_rise=max(max_outside_rise,outside)
        assert outside<2e-9,(name,outside)
        # Exact input-node Lagrange intentionally avoids the original high-degree
        # monomial residual. With filling off, isolate that known difference;
        # with filling on, the entire output must agree with the original.
        if cfg.fill_mode!='none':
            assert difference<2e-9,(name,difference)
        fixtures.append(dict(name=name,data=data,times=times,cfg=matlab_config(cfg),
                             python_cleaned=cleaned,python_trend=trend,pipeline_cleaned=expected,
                             pipeline_compare_mask=outside_rise if cfg.fill_mode=='none' else np.ones_like(data,dtype=bool),
                             operation=info['operation'],offset_channels=np.asarray(offsets,dtype=float)+1,
                             mask_bounds=np.asarray(mask_bounds),degree_rows=np.asarray(degree_rows),
                             fill_bounds=np.asarray(fill_bounds),noise_parameters=np.asarray(noise_parameters),
                             slopes=np.asarray(info['channel_selection'].get('slopes',[])),
                             mean_z=np.asarray(info['channel_selection'].get('mean_z',[]))))
        details.append(dict(name=name,offset_channels=offsets,operation=info['operation'],
                            max_output_difference=difference,max_difference_outside_rise=outside,
                            original_warnings=[str(w.message) for w in caught]))
    assert max_slope<1e-9 and max_mean_z<1e-10 and max_fill<1e-12
    assert offset_count and other_count
    savemat(out/'shared_fixture.mat',dict(cases=np.asarray(fixtures,dtype=object)),do_compression=True)
    report=dict(passed=True,reference_commit='534e926e659885d97bd873a7a31efbac17ebdd96',
                verified_windowed_constraint=False,cases=len(fixtures),offset_trace_fits=offset_count,
                non_offset_trace_fits=other_count,max_python_pipeline_output_difference=max_difference,
                max_difference_outside_lagrange_rise=max_outside_rise,max_slope_difference=max_slope,
                max_mean_z_difference=max_mean_z,max_fill_difference=max_fill,original_warning_count=warning_count,
                details=details,scope='Original TEP masks, slope screening, per-channel polynomial degrees, full computeDetrend_v6, both constraint settings, Gaussian/resample filling and no-offset overall fallback. Without filling, report the original ill-conditioned Lagrange residual separately; with filling, compare the complete output. Synthetic inputs; excludes preprocessing/ICA and real-EEG validation.')
    (out/'python_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT.parent/'results/tep_alignment'))
    validate(parser.parse_args().output)
