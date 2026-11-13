"""Generate small shared fixtures and compare filling with the pipeline/SciPy.

Run python validate_filling.py, then MATLAB run_fill_validation. SciPy is a
validation dependency only. Complete arrays are regenerated, not versioned.
"""
from pathlib import Path
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import sys
import numpy as np
from scipy.io import savemat
from scipy.signal import resample

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from windowed_detrend import Config, detrend
from windowed_detrend.filling import _fourier_resample
from paper_reference.offset_correction import apply_offset_correction


def validate(output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    cases=[]; fft_cases=[]; max_pipeline_error=0.; max_fft_error=0.
    specs=[(-64,100,.01,.018,0.,None,False),(-63,101,.01,.018,.005,None,False),
           (-64,100,.01,.0185,.005,None,False),(-64,100,.01,.018,.1,None,False),
           (0,101,.01,.018,.005,None,False),(0,101,.001,.009,.005,None,False),
           (-64,100,.01,None,.005,None,False),(-64,100,.01,.018,.005,1,False),
           (-64,100,.01,.018,.005,None,True)]
    rng=np.random.default_rng(20261113)
    records=0
    for index,(spec,fill_mode) in enumerate((s,m) for m in ('resample','gaussian') for s in specs):
        start,stop,onset,peak,margin,pre_degree,exponentials=spec
        t=np.arange(start,stop)/1000
        u=np.maximum(t-onset,0); actual_peak=.018 if onset==.01 else .009
        artifact=-30*(1-np.exp(-u/.003))
        mask=t>=actual_peak
        artifact[mask]=-30*(1-np.exp(-(actual_peak-onset)/.003))*np.exp(-(t[mask]-actual_peak)/.06)
        background=.4*np.sin(2*np.pi*13*t)+rng.normal(0,.02,len(t))
        data=np.stack([np.stack([background+artifact,background+2,1.7*(background+artifact)])+e*.1 for e in range(2)])
        cfg=Config(onset_s=onset,peak_s=peak,peak_search_end_s=.03,end_s=.075,
                   fill_mode=fill_mode,fill_margin_s=margin,pre_degree=pre_degree,channels=(0,2))
        if exponentials:
            cfg=replace(cfg,rise_model='singlerise',decay_model='singledecay',constrain_rise=True)
        filled,trend,info=detrend(data,t,cfg)
        raw,raw_trend,_=detrend(data,t,replace(cfg,fill_mode='none'))
        np.testing.assert_array_equal(trend,raw_trend)
        reference=raw.copy(); bounds=[]; noise_parameters=[]
        for rec in info['fits']:
            e,c=rec['epoch'],rec['channel']; split=rec['peak_s']
            masks=[t<onset,(t>=onset)&(t<split),t>=split]
            epochs=type('EpochTimes',(),{'times':t})()
            reference[e,c]=apply_offset_correction(raw[e,c],data[e,c],masks,fill_mode,margin*1000,epochs,['Gaussian'],
                rng=np.random.default_rng(np.random.SeedSequence([cfg.fill_seed,c,e])))
            f=rec['fill']; bounds.append([e+1,c+1,f['source_first']+1,f['source_last']+1,
                                          f['target_first']+1,f['target_last']+1,int(f['source_fallback'])])
            noise_parameters.append([f.get('mean',0.),f.get('std',0.)])
            records+=1
        error=float(np.max(np.abs(filled-reference))); max_pipeline_error=max(error,max_pipeline_error)
        np.testing.assert_allclose(filled,reference,rtol=1e-12,atol=1e-12)
        matlab_cfg={key:(np.empty((0,0)) if value is None else float(value) if isinstance(value,int) and not isinstance(value,bool) else value)
                    for key,value in asdict(cfg).items()}
        matlab_cfg['channels']=np.array(cfg.channels,dtype=float)+1
        cases.append(dict(name=f'case_{index}',data=data,times=t,cfg=matlab_cfg,
                          python_cleaned=filled,python_trend=trend,pipeline_cleaned=reference,
                          fill_bounds=np.array(bounds),noise_parameters=np.array(noise_parameters)))
    for n in (1,2,3,4,7,8,15,16,101):
        for kind in ('noise','nyquist'):
            values=rng.normal(size=n) if kind=='noise' else (-1.)**np.arange(n)
            for m in (1,2,3,4,7,8,31,32):
                expected=resample(values,m); actual=_fourier_resample(values,m)
                max_fft_error=max(max_fft_error,float(np.max(np.abs(actual-expected))))
                np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=1e-12)
                fft_cases.append(dict(values=values,count=m,expected=expected))
    savemat(out/'shared_fixture.mat',dict(cases=np.array(cases,dtype=object),fft_cases=np.array(fft_cases,dtype=object)),do_compression=True)
    report=dict(passed=True,cases=len(cases),processed_traces=records,fourier_cases=len(fft_cases),
                max_pipeline_difference=max_pipeline_error,max_scipy_fourier_difference=max_fft_error,
                pipeline_reference_sha256=hashlib.sha256((ROOT/'paper_reference/offset_correction.py').read_bytes()).hexdigest(),
                notes='Unmodified pipeline Gaussian and resample branches are the Python references. MATLAB uses its native RNG; Gaussian masks, fitted trend and distribution parameters are compared, not equality of independent noise draws.')
    (out/'python_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default=str(ROOT.parent/'results/filling_validation'))
    validate(parser.parse_args().output)
