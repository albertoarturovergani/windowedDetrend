# windowedDetrend - MATLAB

Standalone post-epoching TMS-EEG plugin, local development version 0.1.0.
Requires MATLAB R2023a+ for packaging; tested with R2023b. No Optimization,
Signal Processing, or Statistics toolbox is required by the core.

```matlab
addpath('path/to/MatlabPlugin')
cfg = tmspath_config();
cfg.onset_s = 0.010;
cfg.peak_s = 0.015; % []: detect maximum absolute amplitude in search interval
cfg.rise_model = 'lagrange'; cfg.decay_model = 'poly';
cfg.rise_degree = 3; cfg.decay_degree = 3;
cfg.constrain_rise = false;
[cleaned, artifact, diagnostics] = tmspath_detrend(data, times, cfg);
```

`data` is **epochs x channels x samples**, times in **seconds**. Amplitude units
are preserved. `cfg.channels` uses MATLAB 1-based indices; empty means all.
Set selected EEG channels explicitly if EOG or other channel types are present.
`cfg.end_s` limits the correction window; `cfg.pre_degree=[]` preserves baseline.
`cfg.constrain_decay=true` anchors to the observed value at the split, including
neural activity; this can bias recovery. See PAPER_ALIGNMENT.md.

For an already epoched EEGLAB structure (channels x samples x trials, ms):

```matlab
[EEGout, artifact, diagnostics] = tmspath_eeglab(EEG, cfg);
```

Only EEG.data is changed in the copy; the adapter does not require EEGLAB to run.
No filtering, epoching or ICA is performed. For experimental recordings, see the filling note below.

## Interactive Live Script

The default exponential search uses `cfg.refinements=5` (previously 3) and
`cfg.grid_size=21`. This improves tau search resolution with additional
computation; Lagrange and polynomial fits are unaffected.

Open [demo_windowed_detrend.mlx](demo_windowed_detrend.mlx) in MATLAB and select
**Run All**. It locates the plugin in its own directory and requires no
installation. Edit sections A (toy generation) and B (detrending) at the
beginning; comments explain all parameters and alternative options. It needs no
additional toolbox. The sections generate the toy signal,
call `tmspath_detrend`, check the output, calculate recovery MSE and plot the
results. The recovered curve is labelled **Polynomial**.
The demo uses Lagrange rise interpolation, cubic decay and no filling. It detects
the peak in a user-configurable search window. Lagrange subtraction zeroes the rise window,
including any neural activity there.

```matlab
open("C:\path\to\windowedDetrend\scripts\MatlabPlugin\demo_windowed_detrend.mlx");
```

The editable plain-text companion is `demo_windowed_detrend_source.m`.
The source was executed with MATLAB R2023b. Select Run All in the Live Script
to generate the inline figures. Current demo figures and reports are in
`../results/filling_validation`.

## Native benchmark and cross-language validation

For the **214-epoch, 64-channel, 2,500-sample scalp dataset**, first generate
the shared fixtures with `PythonPlugin/run_scalp_benchmark.py`.

The waveform fixtures and complete reconstructed arrays are excluded from Git
and must be generated locally after cloning. Source code, figures, CSV/JSON
metrics and the compact `matlab_metrics.mat` summary are versioned.

Then run:

```matlab
run_scalp_benchmark('../results/scalp64', '../results/scalp64/matlab')
```

MATLAB processes all 64 distinct channels with the current Lagrange/cubic
default and four comparator methods, checking **68,480 reconstructed channel
traces** against Python. Recovery and agreement metrics are written per
case/channel/method. Inputs, including noise realizations, and configurations
are shared exactly. The data use BioSemi 64 labels and idealized positions;
the spatial signal model is illustrative, not a physiological head model.
See [the full dataset description](../SCALP64_BENCHMARK.md).

The earlier single-channel benchmark remains available:

```matlab
test_plugin
run_benchmark % Generates native noiseless 10x10 grids for two artifact families
% Or use the Python fixture, including exactly the same noise realizations:
run_benchmark('../results/shared_fixture.mat', '../results/matlab')
build_toolbox % Creates ../dist/windowedDetrend-0.1.0.mltbx
```

The shared-fixture run verifies every primary recovered waveform against Python
with maximum absolute error tolerance 1e-6 a.u. MATLAB does not invoke Python.
The native MATLAB benchmark generates its own inputs; Python additionally
provides the degree-sensitivity and automatic-peak analyses and figures.

## Distribution

Install the generated .mltbx locally using MATLAB Add-Ons or
`matlab.addons.toolbox.installToolbox('windowedDetrend-0.1.0.mltbx')`.
For File Exchange use **Connect to GitHub** and the windowedDetrend repository,
preserving AGPL-3.0. Do not use a direct upload that automatically adds BSD.
Publication is a separate step; no public submission has been made.

Citation: Vergani AA et al. (2026), DOI 10.1088/1741-2552/ae668e.


Artifact-only sweep parity: run `run_artifact_sweep` after generating the
Python artifact-only fixture. It evaluates the bounded plugin on 202 artifacts,
with no oscillator and no noise, and writes separate artifact-fit MSE results.
The historical SciPy notebook comparator remains Python-only.


## Current defaults

Windowed polynomial detrending: `rise_model=lagrange`, `decay_model=poly`, `decay_degree=3`, `constrain_rise=false`, and decay anchored to the observed peak. Peak detection remains automatic unless a split is supplied. Lagrange degree is N-1 for N rise samples; `rise_degree` is ignored for this model. At the input nodes, the interpolant equals the input, so the rise window is zero before the subsequent filling step, including removal of any neural signal. Evaluation uses that exact identity rather than a high-degree monomial solve. The historical four-method benchmark explicitly retains its exponential and polynomial 3/3 and 3/5 configurations; it is not a benchmark of the new default. The updated toy demonstrates the new default models.

## Experimental recordings

The Python and MATLAB demos disable filling (`fill_mode='none'`) to show the
detrending result. For experimental recordings, enable random filling of the
zeroed interval with `fill_mode='gaussian'` (the library default), and choose
the peak-search and correction windows for your data. See [filling options](../FILLING.md).
