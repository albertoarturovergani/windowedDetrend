from pathlib import Path
import json
import numpy as np
from scipy.optimize import curve_fit
from windowed_detrend import Config,detrend,detrend_mne
from windowed_detrend.benchmark import synthetic,configs,metrics
import mne
root=Path(__file__).resolve().parents[1];t=np.arange(2500)/10000;gt,a=synthetic(t,.003,.2);data=(gt+a)[None,None,:]
# MNE metadata and non-EEG preservation test.
info=mne.create_info(['Cz','EOG'],10000,['eeg','eog'])
x=np.concatenate([data,np.ones_like(data)*3],axis=1)
epochs=mne.EpochsArray(x,info,tmin=0,verbose=False)
out,trend,diag=detrend_mne(epochs,Config(onset_s=.01,peak_s=.015))
assert out.ch_names==epochs.ch_names
np.testing.assert_array_equal(out.get_data()[:,1],x[:,1])
np.testing.assert_array_equal(epochs.get_data(),x)
# Paper notebook's global fit in milliseconds, preserving initial parameters.
tms=t*1000;mask=t>.010
model=lambda tm,A,tr,td: A*(1-np.exp(-np.clip(np.maximum(tm-10,0)/tr,0,700)))*np.exp(-np.clip(np.maximum(tm-10,0)/td,0,700))
popt,_=curve_fit(model,tms[mask],(gt+a)[mask],p0=[5,20,100],maxfev=10000)
legacy=np.zeros_like(t);legacy[mask]=model(tms[mask],*popt)
rows=[]
for name,cfg in configs().items():
 clean,_,_=detrend(gt[None,None,:],t,cfg)
 rows.append({'method':name,'clean_only_distortion_mse':float(np.mean((clean.ravel()[t>=.01]-gt[t>=.01])**2))})
report={'mne_adapter':'passed','legacy_global_initial_parameters_ms':[5,20,100],
 'legacy_global_fitted_parameters':popt.tolist(),
 'legacy_global_recovery_mse':float(np.mean(((gt+a-legacy)[t>=.01]-gt[t>=.01])**2)),
 'note':'Single extreme-case diagnostic only. Original figure notebook solver, not a full paper reproduction.',
 'clean_only_controls':rows}
(root/'results'/'additional_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
