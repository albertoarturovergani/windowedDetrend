"""Supporting Fig.1 panel B: artifact-only fits, no neural signal/no noise.
Original notebook solver fits A, tau_rise, tau_decay AND onset in milliseconds.
The bounded plugin fixes onset=10 ms; these are intentionally separate solvers.
"""
import argparse,csv,json,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.io import savemat
from .core import Config,detrend
from .benchmark import synthetic


def legacy_model(t_ms,amplitude,rise_ms,decay_ms,onset_ms):
    u=np.clip(np.asarray(t_ms)-onset_ms,0,None)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        return amplitude*(1-np.exp(-np.clip(u/rise_ms,0,700)))*np.exp(-np.clip(u/decay_ms,0,700))


def run(output,size=10):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    t=np.arange(2500)/10000; mask=t>.010
    grid=[(r,d,'grid') for r in np.linspace(.001,.050,size) for d in np.linspace(.025,.250,size)]+[(.003,.200,'extreme')]
    records=[];artifacts=[];bounded=[];params=[];curves={};diagnostics=[]
    cfg=Config(mode='global',onset_s=.010)
    for family in ('windowed','biexp'):
        for r,d,group in grid:
            _,artifact=synthetic(t,r,d,family)
            case=len(artifacts);artifacts.append(artifact);params.append([r,d,0 if family=='windowed' else 1])
            _,trend,info=detrend(artifact[None,None,:],t,cfg)
            fit=trend.ravel(); bounded.append(fit)
            diagnostics.append(dict(case=case,**info))
            if group=='extreme':curves[family]={'time_ms':(t[::10]*1000).tolist(),'artifact':artifact[::10].tolist(),'bounded':fit[::10].tolist()}
            rec=dict(case=case,family=family,group=group,tau_rise_ms=r*1000,tau_decay_ms=d*1000,
                     solver='bounded',mse_artifact_fit=float(np.mean((artifact[mask]-fit[mask])**2)),
                     status='ok',nonpositive_tau=False,onset_ms=10.,warning='')
            records.append(rec)
            legacy=dict(rec,solver='notebook',mse_artifact_fit=None,status='failed',onset_ms=None)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    popt,pcov=curve_fit(legacy_model,t[mask]*1000,artifact[mask],p0=[1.,50.,100.,10.])
                pred=legacy_model(t*1000,*popt)
                if not np.isfinite(pred).all():raise ValueError('Nonfinite fit')
                legacy.update(mse_artifact_fit=float(np.mean((artifact[mask]-pred[mask])**2)),
                              status='ok',nonpositive_tau=bool(popt[1]<=0 or popt[2]<=0),onset_ms=float(popt[3]),
                              warning='; '.join(str(w.message) for w in caught))
                if group=='extreme':curves[family]['notebook']=pred[::10].tolist()
                diagnostics.append(dict(case=case,solver='notebook',parameters=popt.tolist(),warning=legacy['warning']))
            except (RuntimeError,ValueError,FloatingPointError) as err:
                legacy['warning']=str(err)
                if group=='extreme':curves[family]['notebook']=[None]*len(t[::10])
            records.append(legacy)
        print(f'{family}: {len(grid)} artifacts, two solvers',flush=True)
    with (out/'artifact_only_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    savemat(out/'artifact_only_fixture.mat',dict(times=t,data=np.array(artifacts)[:,None,:],parameters=np.array(params),python_trend=np.array(bounded)))
    summary={'grid_size':size,'artifacts':len(artifacts),'fits':len(records),'metric':'mean((isolated_artifact - fitted_artifact)^2)',
        'evaluation_window_s':[float(t[mask][0]),float(t[-1])],'neural_signal':False,'noise':False,
        'notebook_initial_parameters_ms':[1.,50.,100.,10.],'notebook_onset':'free parameter',
        'bounded_onset_s':.010,'records':records,'extreme_curves':curves,
        'failures':sum(r['status']=='failed' for r in records),
        'nonpositive_tau_fits':sum(r['nonpositive_tau'] for r in records)}
    (out/'artifact_only_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (out/'artifact_only_diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False),encoding='utf-8')
    assert len(records)==4*(size*size+1)
    assert not any(r['mse_artifact_fit'] is not None and r['mse_artifact_fit']<0 for r in records)
    print(json.dumps({k:summary[k] for k in ['artifacts','fits','failures','nonpositive_tau_fits']}))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results');p.add_argument('--grid-size',type=int,default=10)
    a=p.parse_args()
    if a.grid_size<2:p.error('Grid size must be >=2')
    run(a.output,a.grid_size)

if __name__=='__main__':main()
