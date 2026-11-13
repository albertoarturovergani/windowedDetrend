# windowedDetrend - MATLAB

Standalone post-epoching TMS-EEG plugin, local development version 0.1.0.
Requires MATLAB R2023a+ for packaging; tested with R2023b. No Optimization,
Signal Processing, or Statistics toolbox is required by the core.

## Controls to adjust first

1. **`slope_threshold=0.5`**: the main decision parameter for recognizing
   channels with extreme artifacts and choosing the detrending strategy.
   A channel is classified as extreme when its slope score reaches or exceeds
   this threshold. Lower values identify more channels; higher values fewer.
2. **`onset_s=0`, `peak_search_end_s=0.020`**: peak search window in seconds.
   The peak divides rise/decay; slope screening uses pre, rise and decay.
3. **Pre degree 2 / Lagrange rise / decay degree 3**: keep these defaults.
   Other processed channels use degree 2 in decay. Lagrange degree is N-1;
   `rise_degree` does not change it. Decay is unconstrained by default.

With auto screening, a threshold higher than every channel score selects the
**basic, whole-epoch linear detrend**, without Lagrange zeroing or gap filling.
If at least one channel reaches the threshold, **windowed detrending** runs on
all selected channels by default, including those below threshold. The score
is the largest absolute epoch-mean standardized slope across pre/rise/decay;
equality counts as extreme (`>=`). See the
[threshold guide and demo outcomes](../../README.md#start-with-these-three-controls).

```matlab
addpath('path/to/MatlabPlugin')
cfg = tmspath_config();
cfg.slope_threshold = 0.5;
cfg.onset_s = 0.0; cfg.peak_search_end_s = 0.020; % Choose for your recording.
[cleaned, artifact, diagnostics] = tmspath_detrend(data, times, cfg);
```

`data` is **epochs x channels x samples**, times in **seconds**. Amplitude units
are preserved. `cfg.channels` uses MATLAB 1-based indices; empty means all.
Set selected EEG channels explicitly if EOG or other channel types are present.
`cfg.end_s` limits the decay fit. The default quadratic pre-onset fit is disabled with `cfg.pre_degree=[]`.
The default `cfg.constrain_decay=false` matches the Windowed TEP notebook; true optionally anchors to the observed peak.

For an already epoched EEGLAB structure (channels x samples x trials, ms):

```matlab
[EEGout, artifact, diagnostics] = tmspath_eeglab(EEG, cfg);
```

Only EEG.data is changed in the copy; the adapter does not require EEGLAB to run.
No filtering, epoching or ICA is performed. For experimental recordings, see the filling note below.

## Interactive Live Script

Open [demo_windowed_detrend.mlx](demo_windowed_detrend.mlx) in MATLAB and select
**Run All**. It locates the plugin in its own directory and requires no
installation. Edit section A (threshold, windows, models), then B (synthetic EEG) if needed;
advanced settings retain their defaults. It needs no
additional toolbox. The sections generate 16-channel, 12-stimulus EEG with timing/amplitude jitter,
call `tmspath_detrend`, check the output, calculate recovery MSE and plot the
results. The recovered curve is labelled **Polynomial**.
The demo uses automatic screening, quadratic pre, Lagrange rise, free cubic/quadratic
decay and Gaussian filling. A threshold sweep shows the switch to whole-epoch linear
detrending when no channels are flagged. Peak search defaults to 0-20 ms. Lagrange subtraction zeroes the rise window,
including any neural activity there.

```matlab
open("C:\path\to\windowedDetrend\scripts\MatlabPlugin\demo_windowed_detrend.mlx");
```


The final figure places butterflies of **all observed channels before correction**
and **all corrected channels** side by side, for the same epoch. The two strongest
simulated artifact channels keep the same colors; other channels are grey.
Amplitudes remain in their original units, with separately labelled vertical
scales. The threshold figure retains one clean-versus-corrected channel example.
Legends are outside the signal axes.

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

The current defaults follow the published Windowed TEP notebook: pre_degree=2, Lagrange rise including the peak, free decay (constrain_decay=false), automatic offset-channel screening, cubic decay on offset channels and quadratic decay on other processed channels. If no offsets are found, the default fallback is whole-epoch linear detrending. See [TEP alignment](../TEP_ALIGNMENT.md) for source provenance, configuration and comparisons.

## Experimental recordings

The Python and MATLAB demos keep Gaussian filling enabled (`fill_mode='gaussian'`),
matching the library default. The zeroed rise plus its margin is replaced with
Gaussian noise. For experimental recordings, choose
the peak-search and correction windows for your data. See [filling options](../FILLING.md).
