"""Recompute Supplementary Figure 1, notebook equations with explicit repairs.

Only model function definitions are loaded from the reviewed source extraction.
E fixes unequal interpolation vectors. Insets use exact 3/200 ms, not nearest grid.
B inset intentionally retains the notebook's fixed-onset fit, unlike its map.
"""
import ast, csv, json, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

ROOT=Path(__file__).resolve().parent
source=ROOT/'source_notebook_functions.py'
names={'capacitive_artifact','generate_damped_oscillator','biexp','exp_rise','exp_decay_constrained','fit_constrained_poly'}
tree=ast.parse(source.read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(source),'exec'))
t=np.arange(0,250,.1); mask=t>10; rise=(t>=10)&(t<15); decay=t>=15
truth=generate_damped_oscillator(t)
rr=np.linspace(1,50,10); dd=np.linspace(25,250,10)
idx=np.unique(np.r_[np.arange(0,2500,10),149,150,2499])
out=ROOT/'regenerated';out.mkdir(parents=True,exist_ok=True)
data={'time_ms':t[idx].tolist(),'truth':truth[idx].tolist(),'rise_ms':rr.tolist(),'decay_ms':dd.tolist(),'curves':[], 'maps':{k:[] for k in 'BDEF'},'extreme':{},'diagnostics':[]}
metrics=[]
for n,(r,d) in enumerate([(r,d) for r in rr for d in dd]+[(3.,200.)]):
    artifact=capacitive_artifact(t,10,15,r,d); signal=artifact+truth
    fits={}; mse={}; vp=None
    for panel in 'BDEF':
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if panel=='B':
                    p,_=curve_fit(biexp,t[mask],artifact[mask],p0=[1.,50.,100.,10.])
                    fit=biexp(t,*p); score=np.mean((artifact[mask]-fit[mask])**2)
                elif panel=='D':
                    p,_=curve_fit(lambda x,a,b,c:biexp(x,a,b,c,10),t[mask],signal[mask],p0=[5.,20.,100.],maxfev=10000)
                    fit=biexp(t,*p,10);score=np.mean((truth[mask]-(signal-fit)[mask])**2)
                elif panel=='E':
                    pr,_=curve_fit(lambda x,a,b:exp_rise(x,a,b,10),t[rise],signal[rise],p0=[5.,10.])
                    vp=exp_rise(15,*pr,10)
                    # Notebook incorrectly passes full t with decay-only y.
                    pd,_=curve_fit(lambda x,tau:exp_decay_constrained(x,tau,15,t,signal),t[decay],signal[decay],p0=[100.])
                    fit=np.zeros_like(t);fit[rise]=exp_rise(t[rise],*pr,10)
                    fit[decay]=exp_decay_constrained(t[decay],*pd,15,t,signal)
                    p=np.r_[pr,pd];score=np.mean((truth-(signal-fit))**2)
                else:
                    if vp is None:raise ValueError('Rise fit required for polynomial boundary')
                    fit=np.zeros_like(t);fit[rise]=np.polyval(np.polyfit(t[rise],signal[rise],3),t[rise])
                    fit[decay]=fit_constrained_poly(t[decay],signal[decay],vp,degree=5,t_peak=15)
                    p=[];score=np.mean((truth-(signal-fit))**2)
                if not np.isfinite(fit).all() or not np.isfinite(score):raise ValueError('Nonfinite fit')
            fits[panel]=fit; mse[panel]=float(score)
            data['diagnostics'].append({'case':n,'panel':panel,'parameters':np.asarray(p).tolist(),'warnings':[str(w.message) for w in caught]})
        except (RuntimeError,ValueError,FloatingPointError) as e:
            fits[panel]=None;mse[panel]=None
            data['diagnostics'].append({'case':n,'panel':panel,'error':str(e)})
        metrics.append({'case':n,'group':'grid' if n<100 else 'extreme','panel':panel,'rise_ms':r,'decay_ms':d,'mse':mse[panel],'status':'ok' if fits[panel] is not None else 'failed'})
    if n<100:
        data['curves'].append({'rise':r,'decay':d,'artifact':artifact[idx].tolist(),'mse':mse['B']})
        for panel in 'BDEF':data['maps'][panel].append(mse[panel])
    else:
        # Original B inset has different optimization settings from B map.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p,_=curve_fit(lambda x,a,b,c:biexp(x,a,b,c,10),t[mask],artifact[mask],p0=[5.,20.,100.])
        b_inset=biexp(t,*p,10)
        data['extreme']={'artifact':artifact[idx].tolist(),'signal':signal[idx].tolist(),
          'fit':{k:(v[idx].tolist() if v is not None else None) for k,v in fits.items()},'mse':mse,
          'b_inset':b_inset[idx].tolist(),'b_inset_mse':float(np.mean((artifact[mask]-b_inset[mask])**2))}
        np.savez_compressed(out/'extreme_full.npz',time_ms=t,artifact=artifact,truth=truth,signal=signal,b_inset=b_inset,**{k:v for k,v in fits.items() if v is not None})
data['notes']={'grid':'10 x 10, same SWEEP=10 as notebook','B':'4 parameters, free onset, p0=[1,50,100,10]; MSE t>10 ms',
'B_inset':'3 parameters, onset fixed at 10 ms, p0=[5,20,100], as notebook inset',
'D':'3 parameters, onset 10 ms, p0=[5,20,100], maxfev=10000; MSE t>10 ms',
'E':'Repaired interpolation lengths; decay anchored to observed peak; MSE full epoch',
'F':'Rise degree 3, decay degree 5; decay anchored to fitted exponential rise peak, as notebook sweep; MSE full epoch',
'insets':'Exact 3/200 ms, not nearest grid point. D/E/F use same algorithm as their maps.',
'failure':'Missing values, never synthetic penalty MSE', 'display':'Curves downsampled to 1 ms plus 14.9,15,249.9; MSE uses original 0.1 ms samples'}
(out/'data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False),encoding='utf-8')
with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
assert len(metrics)==404 and len(data['curves'])==100
for k in 'BDEF':assert len(data['maps'][k])==100
print(json.dumps({'fits':len(metrics),'failures':sum(x['status']=='failed' for x in metrics),'extreme_MSE':data['extreme']['mse'],'B_inset_MSE':data['extreme']['b_inset_mse']}))
