"""Compare original numerical branches and the plugin on identical toy arrays.

Run from any directory: python compare_paper_plugin.py
Dependencies: NumPy, SciPy, Matplotlib. No plugin installation is required.
Original functions and source hashes are preserved in paper_reference.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from windowed_detrend import Config, detrend
from paper_reference.figure_functions import (
    capacitive_artifact, generate_damped_oscillator, exp_rise,
    exp_decay_constrained, fit_constrained_poly,
)
from paper_reference.pipeline_functions import polyfit_constrained_start

# Shared settings: source functions use milliseconds; plugin uses seconds.
TIME_MS = np.arange(-1000, 2500, dtype=float) / 10
TIME_S = TIME_MS / 1000
ONSET_MS, PEAK_MS = 10., 15.
RISE = (TIME_MS >= ONSET_MS) & (TIME_MS < PEAK_MS)
DECAY = TIME_MS >= PEAK_MS
EVAL = TIME_MS >= ONSET_MS
TAU_BOUNDS_MS = (0.5, 1000.)
SEED = 42
OUT = ROOT.parent/'results'/'paper_plugin_comparison'
OUT.mkdir(parents=True, exist_ok=True)
LAGRANGE_CODE = compile((ROOT/'paper_reference'/'lagrange_branch.py').read_text(),
                        'original_lagrange_branch', 'exec')
METHODS = {
 'lagrange_cubic_matched': ('Lagrange / cubic: matched anchor','lagrange',3),
 'poly33_matched': ('Polynomial 3/3: matched anchor','poly',3),
 'poly35_matched': ('Polynomial 3/5: matched anchor','poly',5),
 'exp_original': ('Exponentials: original curve_fit','singlerise',3),
 'exp_same_bounds': ('Exponentials: same tau bounds','singlerise',3),
 'lagrange_cubic_previous_anchor': ('Lagrange / cubic: original prior-sample anchor','lagrange',3),
 'poly35_supplement_anchor': ('Polynomial 3/5: supplementary anchor','poly',5),
 'poly35_main_figure': ('Polynomial 3/5: main-figure convention','poly',5),
}

def original_rise_lagrange(signal):
    namespace = dict(np=np, n_offset=int(RISE.sum()), x_offset=TIME_S[RISE],
                     y_offset=signal[RISE])
    exec(LAGRANGE_CODE, namespace)
    return namespace['trend_line_B']

def exponential_fit(signal, bounded=False):
    rise_options = dict(p0=[5.,10.])
    decay_options = dict(p0=[100.])
    if bounded:
        rise_options['bounds'] = ([-np.inf,TAU_BOUNDS_MS[0]], [np.inf,TAU_BOUNDS_MS[1]])
        decay_options['bounds'] = ([TAU_BOUNDS_MS[0]], [TAU_BOUNDS_MS[1]])
    pr,_ = curve_fit(lambda t,a,tau:exp_rise(t,a,tau,ONSET_MS),
                     TIME_MS[RISE],signal[RISE],**rise_options)
    pd,_ = curve_fit(lambda t,tau:exp_decay_constrained(t,tau,PEAK_MS,TIME_MS,signal),
                     TIME_MS[DECAY],signal[DECAY],**decay_options)
    fit=np.zeros_like(signal)
    fit[RISE]=exp_rise(TIME_MS[RISE],*pr,ONSET_MS)
    fit[DECAY]=exp_decay_constrained(TIME_MS[DECAY],*pd,PEAK_MS,TIME_MS,signal)
    return fit,dict(rise_amplitude=float(pr[0]),rise_tau_ms=float(pr[1]),
                    decay_tau_ms=float(pd[0]),
                    extrapolated_rise_peak=float(exp_rise(PEAK_MS,*pr,ONSET_MS)))

def reference_fit(signal, method):
    fit=np.zeros_like(signal)
    observed_peak=float(np.interp(PEAK_MS,TIME_MS,signal))
    detail=dict(anchor_time_ms=PEAK_MS,anchor_value=observed_peak)
    if method.startswith('exp_'):
        return exponential_fit(signal,bounded=method=='exp_same_bounds')
    if method.startswith('lagrange'):
        fit[RISE]=original_rise_lagrange(signal)
        x0,y0=PEAK_MS/1000,observed_peak
        if method.endswith('previous_anchor'):
            x0,y0=TIME_S[RISE][-1],signal[RISE][-1]
        fit[DECAY],_=polyfit_constrained_start(TIME_S[DECAY],signal[DECAY],3,x0,y0)
        detail.update(anchor_time_ms=float(x0*1000),anchor_value=float(y0))
    else:
        degree=3 if method=='poly33_matched' else 5
        fit[RISE]=np.polyval(np.polyfit(TIME_MS[RISE],signal[RISE],3),TIME_MS[RISE])
        anchor=observed_peak
        if method in ('poly35_supplement_anchor','poly35_main_figure'):
            # Same original rise call as source figure/supplement cells.
            pr,_=curve_fit(lambda t,a,tau:exp_rise(t,a,tau,ONSET_MS),
                          TIME_MS[RISE],signal[RISE],p0=[5.,10.])
            anchor=float(exp_rise(PEAK_MS,*pr,ONSET_MS))
        fit[DECAY]=fit_constrained_poly(TIME_MS[DECAY],signal[DECAY],anchor,degree,PEAK_MS)
        if method=='poly35_main_figure':
            # Cell 5 subtracts the fitted rise peak before LS but adds the observed peak.
            fit[DECAY]+=observed_peak-anchor
        detail.update(anchor_value=anchor,polynomial_degree=degree,
                      added_intercept=observed_peak if method=='poly35_main_figure' else anchor)
    return fit,detail

def plugin_fit(signal, rise_model, degree):
    cfg=Config(fill_mode='none', onset_s=ONSET_MS/1000,peak_s=PEAK_MS/1000,end_s=TIME_S[-1],
               rise_model=rise_model,decay_model='singledecay' if rise_model=='singlerise' else 'poly',
               rise_degree=3,decay_degree=degree,constrain_rise=rise_model=='singlerise',
               constrain_decay=True,fit_offset=False,tau_min_s=TAU_BOUNDS_MS[0]/1000,
               tau_max_s=TAU_BOUNDS_MS[1]/1000,grid_size=21)
    corrected,trend,info=detrend(signal[None,None,:],TIME_S,cfg)
    return corrected[0,0],trend[0,0],info

def mse(a,b,mask): return float(np.mean((a[mask]-b[mask])**2))

def main():
    truth=generate_damped_oscillator(TIME_MS)
    # The original generator is evaluated once per case. Both fits receive copies of the same array.
    cases=[dict(name=f'grid_{i:02d}_{j:02d}',rise_ms=float(r),decay_ms=float(d),noise_sd=0.)
           for i,r in enumerate(np.linspace(1,50,10)) for j,d in enumerate(np.linspace(25,250,10))]
    cases += [dict(name='extreme',rise_ms=3.,decay_ms=200.,noise_sd=0.),
              dict(name='moderate',rise_ms=25.,decay_ms=50.,noise_sd=0.),
              dict(name='extreme_noise05',rise_ms=3.,decay_ms=200.,noise_sd=.5),
              dict(name='extreme_noise2',rise_ms=3.,decay_ms=200.,noise_sd=2.)]
    rng=np.random.default_rng(SEED)
    rows=[];details=[];arrays=dict(time_s=TIME_S,truth=truth)
    for ci,case in enumerate(cases):
        artifact=capacitive_artifact(TIME_MS,ONSET_MS,PEAK_MS,case['rise_ms'],case['decay_ms'])
        noise=rng.normal(0,case['noise_sd'],len(TIME_MS)) if case['noise_sd'] else np.zeros_like(TIME_MS)
        signal=truth+artifact+noise
        arrays[f'input_{ci}']=signal;arrays[f'artifact_{ci}']=artifact
        cache={}
        for method,(label,rise_model,degree) in METHODS.items():
            row=dict(case=case['name'],case_index=ci,**{k:case[k] for k in ('rise_ms','decay_ms','noise_sd')},
                     method=method,status='ok',paper_mse=None,plugin_mse=None,delta_mse=None,
                     output_difference_mse=None,max_abs_difference=None,paper_decay_mse=None,
                     plugin_decay_mse=None,paper_fit_residual_mse=None,plugin_fit_residual_mse=None,
                     warning_count=0,error='')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    key=(rise_model,degree)
                    if key not in cache: cache[key]=plugin_fit(signal.copy(),*key)
                    recovered_plugin,trend_plugin,plugin_info=cache[key]
                    fit,reference_info=reference_fit(signal.copy(),method)
                    recovered_paper=signal-fit
                    assert np.isfinite(fit).all() and np.isfinite(recovered_plugin).all()
                    np.testing.assert_array_equal(recovered_plugin[~EVAL],signal[~EVAL])
                    np.testing.assert_array_equal(recovered_paper[~EVAL],signal[~EVAL])
                    row.update(paper_mse=mse(recovered_paper,truth,EVAL),plugin_mse=mse(recovered_plugin,truth,EVAL),
                      output_difference_mse=mse(recovered_paper,recovered_plugin,EVAL),
                      max_abs_difference=float(np.max(np.abs(recovered_paper[EVAL]-recovered_plugin[EVAL]))),
                      paper_decay_mse=mse(recovered_paper,truth,DECAY),plugin_decay_mse=mse(recovered_plugin,truth,DECAY),
                      paper_fit_residual_mse=mse(signal,fit,EVAL),plugin_fit_residual_mse=mse(signal,trend_plugin,EVAL))
                    row['delta_mse']=row['plugin_mse']-row['paper_mse']
                    arrays[f'paper_{ci}_{method}']=recovered_paper
                    arrays[f'plugin_{ci}_{method}']=recovered_plugin
                    details.append(dict(case=case['name'],method=method,reference=reference_info,plugin=plugin_info))
                except Exception as exc:
                    row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                row['warning_count']=len(caught)
                if caught:
                    details.append(dict(case=case['name'],method=method,
                                        warnings=[f'{w.category.__name__}: {w.message}' for w in caught]))
            rows.append(row)
        if ci%25==0: print(f'Completed {ci+1}/{len(cases)} cases',flush=True)
    np.savez_compressed(OUT/'signals_and_outputs.npz',**arrays)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'diagnostics.json').write_text(json.dumps(details,indent=2,allow_nan=False),encoding='utf-8')
    summary={}
    for method in METHODS:
        group=[r for r in rows if r['method']==method and r['status']=='ok']
        summary[method]=dict(successes=len(group),failures=len(cases)-len(group),
            median_output_difference_mse=float(np.median([r['output_difference_mse'] for r in group])),
            max_output_difference_mse=max(r['output_difference_mse'] for r in group),
            max_abs_difference=max(r['max_abs_difference'] for r in group),
            min_delta_mse=min(r['delta_mse'] for r in group),max_delta_mse=max(r['delta_mse'] for r in group),
            warning_rows=sum(r['warning_count']>0 for r in group))
    metadata=dict(cases=cases,summary=summary,seed=SEED,
      evaluation='10.0 <= t <= 249.9 ms; same 2400 samples for every method',
      versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
      plugin_sha256=hashlib.sha256((ROOT/'src/windowed_detrend/core.py').read_bytes()).hexdigest(),
      reference_manifest=json.loads((ROOT/'paper_reference/manifest.json').read_text()))
    (OUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    refinement_control(arrays,cases)
    make_figure(rows,arrays,cases)
    make_report(rows,metadata)
    print(json.dumps({'extreme':[r for r in rows if r['case']=='extreme'],'summary':summary},indent=2))

def refinement_control(arrays,cases):
    index=next(i for i,c in enumerate(cases) if c['name']=='extreme')
    signal=arrays[f'input_{index}']
    reference,info=exponential_fit(signal,True)
    records=[]
    for count in (3,5,7):
        cfg=Config(fill_mode='none', onset_s=.010,peak_s=.015,rise_model='singlerise',decay_model='singledecay',
                   constrain_rise=True,constrain_decay=True,refinements=count)
        result,_,diagnostics=detrend(signal[None,None,:],TIME_S,cfg)
        records.append(dict(refinements=count,mse=mse(result[0,0],arrays['truth'],EVAL),
            output_difference_mse=mse(result[0,0],signal-reference,EVAL),
            max_abs_difference=float(np.max(np.abs(result[0,0,EVAL]-(signal-reference)[EVAL]))),
            diagnostics=diagnostics))
    (OUT/'solver_refinement_control.json').write_text(json.dumps(dict(reference=info,plugin=records),indent=2),encoding='utf-8')

def make_figure(rows,arrays,cases):
    methods=['lagrange_cubic_matched','poly35_matched','exp_original','lagrange_cubic_previous_anchor']
    index=next(i for i,c in enumerate(cases) if c['name']=='extreme')
    fig,axes=plt.subplots(4,2,figsize=(13,13),layout='constrained')
    for i,method in enumerate(methods):
        row=next(r for r in rows if r['case']=='extreme' and r['method']==method)
        paper=arrays[f'paper_{index}_{method}'];plugin=arrays[f'plugin_{index}_{method}']
        ax=axes[i,0]
        ax.plot(TIME_MS,arrays['truth'],color='0.4',lw=1,label='Known neural signal')
        ax.plot(TIME_MS,paper,color='#2166ac',lw=1.7,label='Original numerical branch')
        ax.plot(TIME_MS,plugin,color='#b2182b',ls='--',lw=1.4,label='Python plugin')
        ax.set_title(METHODS[method][0],fontsize=11)
        ax.set_xlim(10,250);ax.set_ylabel('Amplitude (a.u.)')
        ax.text(.98,.95,f"Recovery MSE\nOriginal: {row['paper_mse']:.6f}\nPlugin: {row['plugin_mse']:.6f}",
                transform=ax.transAxes,ha='right',va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        if i==0:ax.legend(loc='lower right',fontsize=8)
        ax=axes[i,1]
        ax.plot(TIME_MS,plugin-paper,color='#762a83',lw=1.3)
        ax.axhline(0,color='0.6',lw=.6);ax.set_xlim(10,250)
        ax.set_title(f"Plugin minus original; max |difference| = {row['max_abs_difference']:.3g}",fontsize=10)
        ax.set_ylabel('Difference (a.u.)')
        for a in axes[i]:a.set_xlabel('Time relative to TMS pulse (ms)')
    fig.suptitle('Direct comparison on identical toy data: rise 3 ms / decay 200 ms',fontsize=15)
    fig.savefig(OUT/'paper_plugin_comparison.png',dpi=160)
    fig.savefig(OUT/'paper_plugin_comparison.pdf')
    plt.close(fig)

def make_report(rows,metadata):
    text=['# Direct comparison: original paper branches and Python plugin','',
    'This experiment executes numerical functions extracted unchanged from the original figure notebook and the Lagrange branch/polynomial helper from `computeDetrend_v6`. Controlled wrappers provide identical inputs and explicit fit settings. It does not run the entire preprocessing pipeline or claim equivalence to every historical script version. Source names, cell/line locations and SHA-256 hashes are in `PythonPlugin/paper_reference/manifest.json`.','',
    '## Shared inputs and evaluation','',
    '- 10 kHz; epoch -100 to 249.9 ms; onset 10 ms; supplied split 15 ms.',
    '- Original damped oscillator: amplitude 5, frequency 50 Hz, damping 25/s. Original capacitive artifact scale -100.',
    '- 100 noiseless grid cases: rise 1-50 ms and decay 25-250 ms (10 values each), plus 3/200 ms, 25/50 ms, and two 3/200 ms cases with Gaussian noise SD 0.5 and 2 (one realization each, seed 42).',
    '- Every pair receives exactly the same array. Reference routines keep native milliseconds or seconds; the plugin receives seconds. Rise includes 10 ms and excludes 15 ms; decay includes 15 ms.',
    '- All recovery MSE values use the same 2400 samples, 10.0-249.9 ms, and the clean oscillator as truth. Native figure captions sometimes use full-epoch scores or t>10 ms; those different masks are not mixed here.',
    '- `metrics.csv` separately records recovery MSE, decay-only MSE, fit residual MSE, output-difference MSE, maximum absolute difference, failures and warnings. Delta MSE means plugin minus reference; negative favors plugin for that case.','',
    '## What is matched and what is deliberately different','',
    '| Comparison | Reference behavior | Plugin behavior |','|---|---|---|',
    '| Lagrange/cubic, matched | Original high-degree np.polyfit rise; original constrained cubic helper; anchor aligned to observed 15 ms | Exact rise-node interpolation; scaled constrained cubic; same anchor |',
    '| Polynomial 3/3 and 3/5, matched | Original np.polyfit rise and fit_constrained_poly decay; observed 15 ms anchor | Same degrees, no rise constraint, observed 15 ms decay anchor |',
    '| Exponentials, original solver | Original curve_fit calls, p0 rise [5,10 ms], decay [100 ms], no bounds | Grid/refinement, positive taus 0.5-1000 ms, 21 initial values and 5 refinements; same functions and observed anchor |',
    '| Exponentials, same bounds control | Same original functions and initial guesses, with 0.5-1000 ms bounds added explicitly | Same positive tau bounds; different optimizer |',
    '| Lagrange/cubic, prior-sample anchor | Original pipeline decay anchor at last rise sample (14.9 ms with these fixed masks) | Observed split anchor at 15 ms |',
    '| Polynomial 3/5, supplementary convention | Decay anchored to exponential-rise extrapolation at 15 ms | Observed split anchor |',
    '| Polynomial 3/5, main-figure convention | Original cell 5 subtracts fitted rise peak before LS but adds observed peak after LS | Consistent observed anchor in subtraction and reconstruction |','',
    'The first three comparisons align boundary conventions to isolate numerical implementation. The convention comparisons intentionally preserve their original differences and must not be described as identical mathematical settings. The bounded exponential reference is an explicit control, not the unmodified paper solver.','',
    'The supplemental exponential sweep contains an inconsistent interpolation call using a local decay-time vector with the full signal. This benchmark uses the valid call from main-figure cell 5 with matching full time/signal vectors; it does not silently reproduce the sweep error or its penalty value.','',
    '## Extreme toy: 3/200 ms','',
    '| Comparison | Original recovery MSE | Plugin recovery MSE | Delta MSE | Max absolute output difference |',
    '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['case']=='extreme':
            text.append(f"| {METHODS[row['method']][0]} | {row['paper_mse']:.9g} | {row['plugin_mse']:.9g} | {row['delta_mse']:.6g} | {row['max_abs_difference']:.6g} |" if row['status']=='ok' else f"| {METHODS[row['method']][0]} | Failed | | | |")
    text+=['','![Identical-input comparison](results/paper_plugin_comparison/paper_plugin_comparison.png)','',
           '## Across all 104 toy cases','',
           '| Comparison | Successful / failed | Median output-difference MSE | Worst absolute output difference | Warning cases |',
           '|---|---:|---:|---:|---:|']
    for method,s in metadata['summary'].items():
        text.append(f"| {METHODS[method][0]} | {s['successes']} / {s['failures']} | {s['median_output_difference_mse']:.6g} | {s['max_abs_difference']:.6g} | {s['warning_rows']} |")
    text+=['','## Interpretation','',
    'Numerical agreement must be judged separately for matched settings and native conventions. Lagrange monomial fits may issue RankWarning and fail to interpolate high-frequency noise exactly, whereas the plugin zeroes the rise samples by construction. That behavior is not proof of early neural recovery. A lower recovery MSE in one case is not a universal ranking, and one noise realization per level is not a noise robustness study.','',
    '### Matched Lagrange/cubic: clean versus noisy input','',
    '| Input group | Original MSE | Plugin MSE | Maximum output difference |',
    '|---|---:|---:|---:|']
    for case in ('extreme','extreme_noise05','extreme_noise2'):
        row=next(r for r in rows if r['case']==case and r['method']=='lagrange_cubic_matched')
        text.append(f"| {case} | {row['paper_mse']:.9g} | {row['plugin_mse']:.9g} | {row['max_abs_difference']:.6g} |")
    clean=[r for r in rows if r['method']=='lagrange_cubic_matched' and r['noise_sd']==0]
    text+=['',f"Across the 102 noiseless inputs, the largest Lagrange/cubic recovery-MSE difference is {max(abs(r['delta_mse']) for r in clean):.6g} a.u.^2 and the maximum sample difference is {max(r['max_abs_difference'] for r in clean):.6g} a.u. All original degree-49 Lagrange fits issue RankWarning, including those with close numerical agreement on smooth input.",'',
    '### Exponential solver refinement control','',
    'The same-bounds reference isolates optimizer effects. The current default is 5 refinements; 3 is the previous default and 7 is an additional control. On the 3/200 ms toy, changing only plugin refinements gives:','',
    '| Plugin refinements | Recovery MSE | Output-difference MSE versus same-bounds curve_fit |','|---|---:|---:|']
    control=json.loads((OUT/'solver_refinement_control.json').read_text())
    for record in control['plugin']:
        text.append(f"| {record['refinements']} | {record['mse']:.9g} | {record['output_difference_mse']:.6g} |")
    text+=['','This control does not change the shipped defaults. It shows that much of the default exponential discrepancy in this example is due to finite search resolution, rather than a different rise/decay model.','',
    '## Reproduce','',
    'From `scripts/PythonPlugin`, in the documented Python environment:',
    '```bash','python compare_paper_plugin.py','```','',
    'Outputs are written to `scripts/results/paper_plugin_comparison`. The script imports the local plugin directly. It preserves all input/output arrays, source provenance, dependency versions, fit diagnostics and warnings. No changes to the plugin algorithm were made for this comparison.','',
    'Reference source extracts retain their original comments for auditability. New wrapper code and reporting are in English.']
    (ROOT.parent/'PAPER_PLUGIN_COMPARISON.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__': main()
