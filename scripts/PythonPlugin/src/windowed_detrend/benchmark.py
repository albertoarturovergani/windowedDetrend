"""Reproducible synthetic benchmark; see PAPER_ALIGNMENT.md for provenance."""
import argparse, csv, json
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
from .core import Config, detrend


def synthetic(times, tau_rise_s, tau_decay_s, family='windowed',
              onset_s=.010, peak_s=.015, amplitude=5., frequency_hz=50.,
              damping_per_s=25., artifact_gain=-100.):
    """Notebook equations, converted from milliseconds to seconds.
    Amplitudes are arbitrary units (not a claimed microvolt calibration).
    """
    if tau_rise_s<=0 or tau_decay_s<=0 or peak_s<=onset_s: raise ValueError('Invalid time constants')
    t=np.asarray(times); u=np.maximum(t-onset_s,0)
    neural=amplitude*np.exp(-damping_per_s*u)*np.sin(2*np.pi*frequency_hz*u)
    if family=='windowed':
        a=-np.expm1(-u/tau_rise_s)
        decay=t>=peak_s
        a[decay]=-np.expm1(-(peak_s-onset_s)/tau_rise_s)*np.exp(-(t[decay]-peak_s)/tau_decay_s)
    elif family=='biexp': a=-np.expm1(-u/tau_rise_s)*np.exp(-u/tau_decay_s)
    else: raise ValueError('Unknown artifact family')
    return neural,artifact_gain*a


def configs(peak=.015):
    base=Config(onset_s=.010,peak_s=peak,peak_search_end_s=.100,
                rise_model='singlerise',decay_model='singledecay',constrain_rise=True)
    return dict(window_exp=base,
                window_poly3=replace(base,rise_model='poly',decay_model='poly',constrain_rise=False,decay_degree=3),
                window_poly5=replace(base,rise_model='poly',decay_model='poly',constrain_rise=False,decay_degree=5),
                global_biexp=replace(base,mode='global'))


def metrics(original, observed, recovered, mask):
    gt=original[mask]; raw=observed[mask]; rec=recovered[mask]
    mse=float(np.mean((rec-gt)**2)); before=float(np.mean((raw-gt)**2)); power=float(np.mean(gt**2))
    corr=float(np.corrcoef(gt,rec)[0,1]) if np.std(gt)>0 and np.std(rec)>0 else float('nan')
    return dict(mse_recovery=mse,mse_before=before,nmse=mse/power if power else float('nan'),
                improvement_pct=100*(1-mse/before) if before else float('nan'),correlation=corr)


def run(output, size=10, seed=42):
    from scipy.io import savemat
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    # 10 kHz reproduces dt=0.1 ms of the figure notebook, 0..249.9 ms.
    t=np.arange(2500)/10000; mask=t>=.010
    rises=np.linspace(.001,.050,size); decays=np.linspace(.025,.250,size)
    cases=[]
    for family in ('windowed','biexp'):
        cases += [(family,r,d,0.,0,'grid') for r in rises for d in decays]
        cases += [(family,.003,.200,n,rep,'extreme') for n in (0.,.5,2.) for rep in range(1 if n==0 else 3)]
    rows=[]; observations=[]; truths=[]; artifacts=[]; params=[]; diagnostics=[]
    recovered_all=[]; example={}
    for idx,(family,r,d,noise,rep,group) in enumerate(cases):
        gt,a=synthetic(t,r,d,family)
        rng=np.random.default_rng(np.random.SeedSequence([seed,idx]))
        y=gt+a+rng.normal(0,noise,t.size)
        observations.append(y); truths.append(gt); artifacts.append(a)
        params.append([r,d,noise,rep,0 if family=='windowed' else 1])
        peak=.015 if family=='windowed' else .010+r*np.log1p(d/r)
        outputs=[]
        for name,cfg in configs(peak).items():
            rec,trend,info=detrend(y[None,None,:],t,cfg); z=rec.ravel(); outputs.append(z)
            m=metrics(gt,y,z,mask)
            rows.append(dict(case=idx,family=family,group=group,tau_rise_ms=r*1000,tau_decay_ms=d*1000,
                             noise_sd=noise,replicate=rep,method=name,peak_mode='known',**m,
                             mse_early=float(np.mean((z[(t>=.010)&(t<.030)]-gt[(t>=.010)&(t<.030)])**2)),
                             mse_late=float(np.mean((z[t>=.030]-gt[t>=.030])**2)),
                             residual_power=float(np.mean(z[mask]**2))))
            diagnostics.append(dict(case=idx,method=name,**info))
            if group=='extreme' and noise==0 and family=='windowed': example[name]=z
        recovered_all.append(outputs)
        if group=='extreme' and noise==0:
            for name,cfg in configs(None).items():
                if cfg.mode=='global': continue
                rec,_,info=detrend(y[None,None,:],t,cfg)
                m=metrics(gt,y,rec.ravel(),mask)
                rows.append(dict(case=idx,family=family,group='auto_peak',tau_rise_ms=r*1000,tau_decay_ms=d*1000,
                    noise_sd=noise,replicate=rep,method=name,peak_mode='detected',**m,
                    mse_early=float('nan'),mse_late=float('nan'),residual_power=float(np.mean(rec.ravel()[mask]**2))))
            # Sensitivity to polynomial order as in supporting Fig.4; synthetic GT here.
            for degree in range(1,9):
                cfg=replace(configs(peak)['window_poly3'],decay_degree=degree)
                rec,_,_=detrend(y[None,None,:],t,cfg)
                rows.append(dict(case=idx,family=family,group='degree_sweep',tau_rise_ms=r*1000,tau_decay_ms=d*1000,
                    noise_sd=noise,replicate=rep,method=f'poly{degree}',peak_mode='known',**metrics(gt,y,rec.ravel(),mask),
                    mse_early=float('nan'),mse_late=float('nan'),residual_power=float(np.mean(rec.ravel()[mask]**2))))
        if (idx+1)%25==0: print(f'{idx+1}/{len(cases)} synthetic cases',flush=True)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    savemat(out/'shared_fixture.mat',dict(times=t,data=np.array(observations)[:,None,:],
       truth=np.array(truths),artifact=np.array(artifacts),parameters=np.array(params),
       python_recovered=np.array(recovered_all),method_names=np.array(list(configs()),dtype=object)))
    (out/'fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    summary={'seed':seed,'grid_size':size,'cases':len(cases),'metric_rows':len(rows),
             'amplitude_unit':'a.u.','time_unit':'seconds','evaluation_window_s':[.010,float(t[-1])],
             'primary_peak_mode':'known simulation peak; separate detected-peak extreme checks',
             'methods':{k:asdict(v) for k,v in configs().items()},'extreme_windowed':[
                 r for r in rows if r['family']=='windowed' and r['group']=='extreme' and r['noise_sd']==0]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot_results(out,rows,t,example,rises,decays)
    print(json.dumps({k:summary[k] for k in ['cases','metric_rows','extreme_windowed']},indent=2))
    return summary


def plot_results(out,rows,t,example,rises,decays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    methods=list(configs())
    fig,axes=plt.subplots(2,4,figsize=(15,7.5),constrained_layout=True)
    values=[r['mse_recovery'] for r in rows if r['group']=='grid']
    norm=LogNorm(max(min(values),1e-8),max(values))
    for i,family in enumerate(('windowed','biexp')):
        for j,method in enumerate(methods):
            selected=[r for r in rows if r['group']=='grid' and r['family']==family and r['method']==method]
            z=np.array([r['mse_recovery'] for r in selected]).reshape(len(rises),len(decays))
            im=axes[i,j].pcolormesh(rises*1000,decays*1000,z.T,shading='nearest',norm=norm,cmap='viridis')
            axes[i,j].set(title=f'{family} / {method}',xlabel='Rise tau (ms)',ylabel='Decay tau (ms)')
    fig.colorbar(im,ax=axes,label='Recovery MSE (a.u. squared)');fig.savefig(out/'recovery_mse.png',dpi=160);plt.close(fig)
    gt,a=synthetic(t,.003,.200)
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    axes[0].plot(t*1000,gt+a,color='gray',label='Contaminated');axes[0].plot(t*1000,gt,color='black',label='Ground truth')
    axes[0].set(title='Extreme windowed artifact: rise 3 ms, decay 200 ms, peak 15 ms',ylabel='Amplitude (a.u.)');axes[0].legend()
    axes[1].plot(t*1000,gt,color='black',lw=2,label='Ground truth')
    for name,z in example.items(): axes[1].plot(t*1000,z,label=name,alpha=.8)
    axes[1].set(xlabel='Time (ms)',ylabel='Amplitude (a.u.)');axes[1].legend(ncol=2)
    fig.savefig(out/'extreme_recovery.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results');p.add_argument('--grid-size',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.grid_size<2: p.error('--grid-size must be >=2')
    run(args.output,args.grid_size,args.seed)

if __name__=='__main__': main()
