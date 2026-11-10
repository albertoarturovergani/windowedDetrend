"""Run from the repository's scripts directory with dependencies available."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

plugin_src = Path.cwd() / "PythonPlugin" / "src"
if not (plugin_src / "windowed_detrend" / "__init__.py").is_file():
    raise FileNotFoundError("Run this example from the repository's scripts directory")
sys.path.insert(0, str(plugin_src))
from windowed_detrend import Config, detrend

# Toy data represent an already-epoched signal; pulse artifact is absent.
fs = 10000
times = np.arange(-1000, 2500) / fs  # seconds relative to the TMS pulse at zero
u = np.maximum(times - 0.010, 0)
neural = 5 * np.exp(-25 * u) * np.sin(2 * np.pi * 50 * u)
artifact = -100 * (1 - np.exp(-u / 0.003))
decay = times >= 0.015
artifact[decay] = -100 * (1 - np.exp(-0.005 / 0.003)) * np.exp(-(times[decay] - 0.015) / 0.200)
data = (neural + artifact)[None, None, :]  # epochs x channels x samples

cfg = Config(onset_s=0.010, peak_s=0.015, end_s=times[-1],
             rise_model='lagrange', decay_model='poly', rise_degree=3,
             decay_degree=3, constrain_rise=False, constrain_decay=True,
             pre_degree=None)
corrected, trend, diagnostics = detrend(data, times, cfg)
mask = times >= cfg.onset_s
mse = float(np.mean((corrected[0, 0, mask] - neural[mask])**2))
assert corrected.shape == data.shape
np.testing.assert_array_equal(corrected[..., ~mask], data[..., ~mask])
print(f'Toy recovery MSE: {mse:.6f} a.u.^2')

out = Path.cwd() / 'images'
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
axes[0].plot(times * 1000, data[0, 0], label='Input: neural + slow artifact')
axes[0].plot(times * 1000, trend[0, 0], '--', label='Estimated trend')
axes[1].plot(times * 1000, neural, 'k', label='Known toy neural signal')
axes[1].plot(times * 1000, corrected[0, 0], label='Detrended; input to subsequent ICA')
for ax in axes:
    ax.axvline(10, color='.5', ls=':', label='Detrend onset: 10 ms')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.legend(loc='best')
axes[1].set_xlabel('Time relative to TMS pulse (ms)')
axes[1].set_xlim(-20, 250)
fig.suptitle(f'Python post-epoching toy - Lagrange rise / cubic decay - MSE {mse:.4f} a.u.²')
fig.savefig(out / 'toy_post_epoching_python.png', dpi=160)
plt.close(fig)
results = Path.cwd() / 'results' / 'toy'
results.mkdir(parents=True, exist_ok=True)
np.savez(results / 'python_toy.npz', times=times, data=data, truth=neural,
         corrected=corrected, trend=trend)
(results / 'python_summary.json').write_text(json.dumps({'mse': mse, 'shape': list(data.shape)}))
# In real data, pass corrected (or the MNE/EEGLAB adapter output) to ICA next.
