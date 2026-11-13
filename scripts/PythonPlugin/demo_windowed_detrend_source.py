# %% Multichannel EEG: threshold first
# Run all cells in order. Synthetic EEG, not experimental validation.
# No plugin installation needed; source is loaded from this repository.
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import matplotlib.pyplot as plt

candidates = []
for parent in [Path.cwd(), *Path.cwd().parents]:
    candidates.extend([parent / 'src', parent / 'PythonPlugin' / 'src',
                       parent / 'scripts' / 'PythonPlugin' / 'src'])
plugin_src = next((p for p in candidates if (p/'windowed_detrend/__init__.py').is_file()), None)
if plugin_src is None:
    raise FileNotFoundError('Start inside the windowedDetrend repository')
sys.path[:0] = [str(plugin_src), str(plugin_src.parent)]
from windowed_detrend import Config, detrend
from demo_multichannel_data import make_demo_eeg

# %% A. Controls in order of importance
cfg = Config(
    # 1. Slope Threshold: main decision parameter for extreme-artifact detection.
    # A channel score >= threshold is classified as extreme: enable windowed fits.
    # Threshold above ALL channel scores: apply the basic whole-epoch linear fit.
    # Scores are absolute epoch-mean standardized slopes. Lower => more flagged.
    slope_threshold=0.5,
    # 2. Peak search interval (seconds relative to the recorded TMS trigger).
    # The detected peak splits rise/decay; slopes use pre, rise AND decay.
    onset_s=0.0,
    peak_search_end_s=0.020,
    # 3. Models, kept at the published Windowed TEP defaults.
    pre_degree=2,
    rise_model='lagrange',  # Interpolates every rise sample; degree N-1.
    decay_model='poly',
    decay_degree=3,        # Offset channels: cubic; others: degree 2.
    fill_mode='gaussian',  # Default: replace the zeroed rise plus margin with Gaussian noise.
)
# All other fields keep their library defaults: channel_policy='auto',
# constrain_decay=False, only_offset_channels=False, overall_degree=1.
# No flagged channel => whole-epoch linear fit, no Lagrange and no filling.
# At least one flagged => windowed fits on ALL selected channels by default.
# Changing cfg does not change the synthetic data below.

# %% B. Synthetic EEG: channels, stimuli and jitter
toy = dict(n_epochs=12, n_channels=16, fs_hz=2000, seed=42,
           stimulus_jitter_s=0.0007,  # Common latency shift per stimulus: +/-0.7 ms.
           channel_jitter_s=0.00035,  # Additional latency per channel: +/-0.35 ms.
           amplitude_jitter=0.10,    # Artifact amplitude varies +/-10% per trace.
           noise_sd=0.05)           # Independent background noise, arbitrary units.
data, times, truth, artifact, metadata = make_demo_eeg(**toy)
labels = metadata['channel_labels']
# Two channels have stronger artifacts, with weaker contamination elsewhere.
# A damped response and ongoing background provide known clean ground truth.
# Jitter changes signal latency relative to fixed event markers at t=0;
# it does not resample the epochs or introduce the removed pulse artifact.
print('Input shape (epochs, channels, samples):', data.shape)
print('Stimulus jitter (ms):', np.round(1000*metadata['stimulus_jitter_s'], 3))

# %% 1. Apply automatic screening and Polynomial detrending
corrected, trend, diagnostics = detrend(data, times, cfg)
selection = diagnostics['channel_selection']
scores = np.max(np.abs(np.asarray(selection['mean_z'])), axis=1)
offset_channels = selection['offset_channels']
operation = diagnostics['operation']
mask = times >= 0
mse_full = float(np.mean((corrected[:, :, mask]-truth[:, :, mask])**2))
print('Operation:', operation)
print('Offset channels:', [labels[c] for c in offset_channels])
print('Channel  max|mean z-slope|  classification')
for c, label in enumerate(labels):
    print(f'{label:7s}  {scores[c]:8.4f}          {"offset" if c in offset_channels else "below threshold"}')
print(f'Post-trigger recovery MSE: {mse_full:.6f} a.u.^2')
# Inspect operation before accessing rise/decay: fallback records have overall.
if operation == 'windowed':
    print('Decay degree by channel:', [r['decay']['degree'] for r in diagnostics['fits'] if r['epoch']==0])
else:
    print('No windowed detrending: whole-epoch linear fit; no gap filling.')

# %% 2. Change only the threshold on the SAME data
# This sweep illustrates decisions; it does not optimize the threshold.
sweep_thresholds = np.unique(np.r_[.25, .5, 1., 2., 3., cfg.slope_threshold])
sweep_counts = []
sweep_operations = []
sweep_mse = []
print('Threshold  flagged channels  operation             recovery MSE')
for threshold in sweep_thresholds:
    sweep_output, _, sweep_info = detrend(data, times, replace(cfg, slope_threshold=float(threshold)))
    count = len(sweep_info['channel_selection']['offset_channels'])
    error = float(np.mean((sweep_output[:, :, mask]-truth[:, :, mask])**2))
    sweep_counts.append(count); sweep_operations.append(sweep_info['operation']); sweep_mse.append(error)
    print(f'{threshold:8.3f}  {count:16d}  {sweep_info["operation"]:20s}  {error:.6f}')

# %% 3. Inspect screening and recovered waveforms
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
colors = ['#b55229' if c in offset_channels else '#75899a' for c in range(toy['n_channels'])]
ax = axes[0, 0]
ax.bar(np.arange(toy['n_channels']), scores, color=colors)
ax.axhline(cfg.slope_threshold, color='#b55229', ls='--', label=f'Threshold = {cfg.slope_threshold:g}')
ax.set_xticks(np.arange(toy['n_channels']), labels, rotation=60, ha='right', fontsize=8)
ax.set_ylabel('max |mean slope z-score|'); ax.set_title('1. Channel decision'); ax.legend()
ax = axes[0, 1]
ax.plot(sweep_thresholds, sweep_counts, 'o', color='#245877')
for x, count, name in zip(sweep_thresholds, sweep_counts, sweep_operations):
    ax.annotate('Linear' if name=='overall' else 'Windowed', (x,count),
                xytext=(0,8), textcoords='offset points', ha='center', fontsize=8)
ax.axvline(cfg.slope_threshold, color='#b55229', ls='--')
ax.set(xlabel='Slope threshold', ylabel='Flagged channels', ylim=(-1,toy['n_channels']+3),
       xlim=(0,float(sweep_thresholds[-1])+.3), title='2. Same EEG, different threshold')
display_channels = [int(np.argmax(scores)), int(np.argmin(scores))]
for ax, c in zip(axes[1], display_channels):
    ax.plot(times*1000, truth[0,c], color='black', lw=1, label='Known clean signal')
    ax.plot(times*1000, corrected[0,c], color='#b55229', lw=1, label='Polynomial')
    ax.axvspan(cfg.onset_s*1000,cfg.peak_search_end_s*1000,color='#dce7ee',alpha=.5,label='Peak search')
    rec = next((r for r in diagnostics['fits'] if r['epoch']==0 and r['channel']==c), None)
    if rec and 'peak_s' in rec:
        ax.axvline(1000*rec['peak_s'],color='#245877',ls=':',label='Detected split')
        subtitle=f'decay degree {rec["decay"]["degree"]}'
    else:
        subtitle='whole-epoch linear'
    ax.set(title=f'{labels[c]}, epoch 1: {subtitle}',xlabel='Time from trigger (ms)',
           ylabel='Amplitude (a.u.)',xlim=(-20,200))
# Share one legend below the layout so it never covers either waveform.
fig.legend(*axes[1, 0].get_legend_handles_labels(), loc='outside lower center',
           ncol=2, fontsize=9, frameon=False)
fig.suptitle(f'Polynomial | {toy["n_channels"]} channels x {toy["n_epochs"]} stimuli | {operation}')

# %% 4. Inspect the simulated EEG and timing jitter
fig_input, axes_input = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
ax = axes_input[0]
im = ax.imshow(data[0],aspect='auto',origin='lower',
               extent=[1000*times[0],1000*times[-1],.5,toy['n_channels']+.5],cmap='RdBu_r')
ax.set(xlabel='Time from trigger (ms)',ylabel='Channel',title='Observed EEG: epoch 1',xlim=(-20,200))
fig_input.colorbar(im,ax=ax,label='Amplitude (a.u.)')
ax = axes_input[1]
epoch_numbers=np.arange(1,toy['n_epochs']+1)
for c in range(toy['n_channels']):
    ax.plot(epoch_numbers,1000*metadata['artifact_onsets_s'][:,c],'.',color='#75899a',alpha=.6)
ax.plot(epoch_numbers,1000*np.mean(metadata['artifact_onsets_s'],axis=1),'-o',color='#245877',label='Mean across channels')
ax.axhline(3,color='black',ls=':',label='Nominal onset: 3 ms')
ax.set(xlabel='Stimulus / epoch',ylabel='Residual artifact onset (ms)',title='Stimulus jitter + channel latency')
fig_input.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
                 ncol=2, fontsize=9, frameon=False)
plt.show()

# %% 5. Continue the host pipeline
# Replace data/times with real preprocessed epochs; keep matching good EEG
# channels in MATLAB and Python because screening depends on that population.
# Gaussian filling remains enabled; inspect diagnostics, then continue ICA.
data_for_ica = corrected
