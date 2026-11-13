"""Conceptual framework: simulated signals, fitted trends and neural recovery.
Polynomial comparison uses Lagrange rise interpolation and cubic decay.
"""
from pathlib import Path
import ast,json,warnings,sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'PythonPlugin/src'))
from windowed_detrend import Config, detrend
source=root/'results/supplement1/source_notebook_functions.py'
names={'generate_damped_oscillator','capacitive_artifact','biexp','exp_rise','exp_decay_constrained'}
tree=ast.parse(source.read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(source),'exec'))
t=np.arange(0,250,.1);t0=10.;peak=15.
truth=generate_damped_oscillator(t);artifact=capacitive_artifact(t,t0,peak,3.,200.);observed=truth+artifact
mask=t>t0;rise=(t>=t0)&(t<peak);decay=t>=peak
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 pglobal,_=curve_fit(lambda x,a,b,c:biexp(x,a,b,c,t0),t[mask],observed[mask],p0=[3.,50.,100.])
 global_fit=biexp(t,*pglobal,t0)
 pr,_=curve_fit(lambda x,a,tau:exp_rise(x,a,tau,t0),t[rise],observed[rise],p0=[5.,10.])
 pd,_=curve_fit(lambda x,tau:exp_decay_constrained(x,tau,peak,t,observed),t[decay],observed[decay],p0=[100.])
 exp_fit=np.zeros_like(t);exp_fit[rise]=exp_rise(t[rise],*pr,t0);exp_fit[decay]=exp_decay_constrained(t[decay],*pd,peak,t,observed)
 # Use the actual plugin default models, with the known simulation split.
 cfg=Config(fill_mode='none', peak_s=peak/1000)
 _,poly_trend,poly_info=detrend(observed[None,None,:],t/1000,cfg)
 poly_fit=poly_trend[0,0]
 np.testing.assert_array_equal(poly_fit[rise],observed[rise])
 assert len(poly_info['fits'][0]['decay']['coefficients'])==4
 less=truth+capacitive_artifact(t,t0,30.,25.,50.)
 pless,_=curve_fit(lambda x,a,b,c:biexp(x,a,b,c,t0),t[mask],less[mask],p0=[3.,50.,100.])
 less_fit=biexp(t,*pless,t0)
fits={'global':global_fit,'windowed_exp':exp_fit,'windowed_lagrange_3':poly_fit}
residuals={k:observed-v for k,v in fits.items()}
mse={k:float(np.mean((v[mask]-truth[mask])**2)) for k,v in residuals.items()}
early=mask & (t<30);late=t>=30
interval_mse={k:{'early_10_30_ms':float(np.mean((v[early]-truth[early])**2)),'late_30_250_ms':float(np.mean((v[late]-truth[late])**2))} for k,v in residuals.items()}
for key in fits:
 assert np.isfinite(fits[key]).all()
 np.testing.assert_allclose(residuals[key]+fits[key],observed,atol=1e-12)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'normal','svg.fonttype':'none'})
fig=plt.figure(figsize=(14,13),layout='constrained');gs=fig.add_gridspec(3,6,height_ratios=[1,1.5,1.15])
axes=[fig.add_subplot(gs[0,i:i+2]) for i in (0,2,4)]
axes += [fig.add_subplot(gs[1,:3]),fig.add_subplot(gs[1,3:]),fig.add_subplot(gs[2,:3]),fig.add_subplot(gs[2,3:])]
black='#171717';blue='#2864ae';red='#cf3434';green='#358554'
def panel(ax,letter,title):
 ax.set_title(letter+'  '+title,loc='left',pad=10);ax.set_xlim(0,250);ax.set_xlabel('Time (ms)');ax.set_ylabel('Amplitude (a.u.)');ax.axhline(0,color='.8',lw=.6,zorder=0)
def windows(ax):
 ax.axvspan(10,15,color=blue,alpha=.09,lw=0);ax.axvspan(15,250,color=red,alpha=.04,lw=0);ax.axvline(15,color='.55',lw=.7,ls=':')
for a,letter,title,y in zip(axes[:3],'ABC',['Damped oscillator','Extreme capacitive artifact','Oscillator + capacitive artifact'],[truth,artifact,observed]):
 panel(a,letter,title);a.plot(t,y,color=black,lw=1.5)
axes[1].text(.32,.15,'Rise tau = 3 ms\nDecay tau = 200 ms\nPeak = 15 ms',transform=axes[1].transAxes,fontsize=9)
axes[1].set_ylim(-90,5);axes[2].set_ylim(-90,5)
ax=axes[3];panel(ax,'D','External comparator: bi-exponential fit');ax.plot(t,observed,color=black,label='Observed',lw=1.3);ax.plot(t,global_fit,'--',color=green,label='External bi-exponential',lw=1.5);ax.set_ylim(-90,5);ax.legend(loc='upper right')
ins=ax.inset_axes([.54,.08,.41,.29]);ins.plot(t,less,color=black,lw=1);ins.plot(t,less_fit,'--',color=green,lw=1);ins.set(xlim=(0,150),xticks=[],yticks=[],title='Less extreme: 25/50 ms');ins.title.set_fontsize(9)
ax=axes[4];panel(ax,'E','Our method: separate rise and decay fits');windows(ax)
ax.plot(t,observed,color=black,lw=1.3,label='Observed');ax.plot(t,exp_fit,'--',color=blue,lw=1.4,label='Exponential rise / decay');ax.plot(t,poly_fit,':',color=red,lw=1.6,label='Lagrange rise / cubic decay');ax.set_ylim(-90,5);ax.legend(loc='upper right',fontsize=9)
ins=ax.inset_axes([.52,.06,.44,.35]);windows(ins)
ins.plot(t,observed,color=black,lw=1);ins.plot(t,exp_fit,'--',color=blue,lw=1);ins.plot(t,poly_fit,':',color=red,lw=1.5);ins.set(xlim=(10,55),ylim=(-86,-57),xticks=[10,30,55],yticks=[]);ins.set_title('Early-window detail (ms)',fontsize=9)
ax=axes[5];panel(ax,'F','Recovery: external bi-exponential comparator');ax.plot(t,truth,color=black,lw=1.5,label='Original oscillator');ax.plot(t,residuals['global'],color=green,lw=1.5,label='External bi-exponential');ax.legend(loc='upper right',fontsize=9);ax.text(.03,.04,f"Recovery MSE = {mse['global']:.3f} a.u.²",transform=ax.transAxes)
ax=axes[6];panel(ax,'G','Recovery: our windowed method');ax.plot(t,truth,color=black,lw=1.5,label='Original oscillator');ax.plot(t,residuals['windowed_exp'],color=blue,lw=1.2,label='Recovered: exponentials');ax.plot(t,residuals['windowed_lagrange_3'],color=red,lw=1.2,label='Recovered: Lagrange / cubic');ax.legend(loc='upper right',fontsize=9)
ax.text(.03,.04,f"MSE: exp {mse['windowed_exp']:.3f}; poly {mse['windowed_lagrange_3']:.3f} a.u.²",transform=ax.transAxes)
fig.suptitle('Conceptual framework of windowed detrending\nSeparate rise and decay: exponential or polynomial fits',fontsize=16)
out=root/'images';out.mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(out/f'paper_concept_computed.{ext}',dpi=180)
plt.close(fig)
results=root/'results/paper_concept';results.mkdir(exist_ok=True)
np.savez_compressed(results/'signals_and_fits.npz',time_ms=t,truth=truth,artifact=artifact,observed=observed,**fits,**{'recovered_'+k:v for k,v in residuals.items()})
report={'recovery_mse':mse,'interval_recovery_mse':interval_mse,'evaluation_window_ms':[float(t[mask][0]),float(t[mask][-1])],'fit_samples':int(mask.sum()),'global_parameters':pglobal.tolist(),'exp_rise_parameters':pr.tolist(),'exp_decay_parameters':pd.tolist(),'poly_degrees':[int(rise.sum())-1,3],'polynomial_models':['lagrange','poly'],'warnings':[str(w.message) for w in caught],'source':'2025_12_01_FigurePaper.ipynb, cells 3 and 5','note':'Conceptual framework computed from synthetic signals. Lagrange rise interpolation and cubic decay use the plugin default models with a fixed simulation peak. The bi-exponential is an external comparator, not a step of the windowed algorithm. The displayed comparator uses an unconstrained fit and has a nonphysical decay tau; this example is not a general ranking of bi-exponential methods. No fit uses neural truth. Lagrange removes all observed rise samples, including neural activity.'}
(results/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(report,indent=2))
