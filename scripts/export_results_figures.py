"""Export the Supplementary Figure 1 reconstruction and Python plugin results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'; OUT=ROOT/'images';OUT.mkdir(exist_ok=True)
reds=LinearSegmentedColormap.from_list('white_red',['white','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'])
norm=LogNorm(.1,1000,clip=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
d=json.loads((RESULTS/'supplement1/data.json').read_text());t=np.array(d['time_ms']);e=d['extreme'];gt=np.array(d['truth'])
fig,ax=plt.subplots(3,2,figsize=(13,16),layout='constrained')
for panel,a in zip('AC',[ax[0,0],ax[1,0]]):
 for c in d['curves']:
  y=np.array(c['artifact'])+(gt if panel=='C' else 0)
  a.plot(t,y,color=plt.get_cmap('brg')(norm(c['mse'])) if panel=='A' else '.5',lw=.6,alpha=.25)
 a.plot(t,e['artifact' if panel=='A' else 'signal'],color='black',lw=2,label='Exact 3/200 ms; peak 15 ms')
 a.set(xlim=(0,250),ylim=(-108,8),xlabel='Time (ms)',ylabel='Amplitude (a.u.)')
 a.set_title(panel+'  '+('Simulated capacitive dynamics' if panel=='A' else 'Damped oscillator + capacitive artifacts'),loc='left')
 a.legend(loc='lower right',fontsize=9)
 if panel=='C':
  ins=a.inset_axes([.54,.69,.43,.27]);ins.plot(t,gt,color='black');ins.set(xlim=(0,200),xticks=[],yticks=[],title='Original oscillator')
titles={'B':'Non-windowed bi-exponential fit','D':'Non-windowed bi-exponential detrending','E':'Windowed exponential detrending','F':'Windowed polynomial detrending (3/5)'}
for panel,a in zip('BDEF',[ax[0,1],ax[1,1],ax[2,0],ax[2,1]]):
 z=np.array(d['maps'][panel],dtype=float).reshape(10,10).T
 im=a.pcolormesh(d['rise_ms'],d['decay_ms'],z,shading='nearest',norm=norm,cmap='brg' if panel=='B' else reds)
 a.set(xlim=(1,50),ylim=(25,250),xlabel='Tau rise (ms)',ylabel='Tau decay (ms)');a.set_title(panel+'  '+titles[panel],loc='left')
 fig.colorbar(im,ax=a,label='MSE (a.u.²), logarithmic',shrink=.9)
 a.plot(3,200,'ks',ms=4)
 inset=a.inset_axes([.53,.53,.44,.42]);obs=e['artifact'] if panel=='B' else e['signal'];fit=np.array(e['b_inset'] if panel=='B' else e['fit'][panel])
 inset.plot(t,obs,'k',lw=1,label='Artifact' if panel=='B' else 'Contaminated');inset.plot(t,fit,'r--',lw=1,label='Fit')
 inset.set(xlim=(0,80 if panel=='B' else 200),xticks=[],yticks=[]);inset.legend(fontsize=7,loc='upper right')
 if panel!='B':
  ins=a.inset_axes([.53,.05,.44,.35]);ins.plot(t,gt,'k',lw=1,label='Original');ins.plot(t,np.array(e['signal'])-fit,'r',lw=1,label='Recovered')
  ins.set(xlim=(0,200),xticks=[],yticks=[],title='Detrend');ins.legend(fontsize=7,loc='lower right')
fig.suptitle('Supplementary Figure 1 - reconstruction from the original notebook\nExact 3/200 ms insets; interpolation repair in E; D-F: white = low MSE',fontsize=14)
fig.savefig(OUT/'supplementary_figure_1_python.png',dpi=180);fig.savefig(OUT/'supplementary_figure_1_python.pdf');plt.close(fig)

s=loadmat(RESULTS/'shared_fixture.mat');t=s['times'].ravel()*1000
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
names=['Windowed exponential','Windowed polynomial 3/3','Windowed polynomial 3/5','Global bi-exponential']
for col,(cid,start,family) in enumerate([(100,0,'Windowed capacitive artifact'),(207,107,'Bi-exponential artifact')]):
 a=axs[0,col];a.plot(t,s['truth'][cid],'k',lw=2,label='Original oscillator')
 for m,name in enumerate(names):a.plot(t,s['python_recovered'][cid,m],lw=1,label=name)
 a.set(title=family+' - 3/200 ms',xlabel='Time (ms)',ylabel='Amplitude (a.u.)',xlim=(0,250));a.legend(fontsize=8)
 ids=np.arange(start,start+100);mask=t>=10
 mse=np.mean((s['python_recovered'][ids,1][:,mask]-s['truth'][ids][:,mask])**2,axis=1).reshape(10,10).T
 im=axs[1,col].pcolormesh(np.linspace(1,50,10),np.linspace(25,250,10),mse,norm=norm,cmap=reds,shading='nearest')
 axs[1,col].set(title='Recovery MSE - windowed polynomial 3/3',xlabel='Tau rise (ms)',ylabel='Tau decay (ms)')
 fig.colorbar(im,ax=axs[1,col],label='MSE (a.u.²), logarithmic')
fig.suptitle('Python plugin results - bounded solver; known peak; no noise',fontsize=14)
fig.savefig(OUT/'plugin_results_python.png',dpi=180);fig.savefig(OUT/'plugin_results_python.pdf');plt.close(fig)
print('Exported Python supplementary and plugin figures (PNG and PDF).')
