# Using windowed detrending after epoching

The plugin processes already-preprocessed epochs. Its current defaults follow
the published **Windowed TEP** notebook: quadratic pre-onset fit, Lagrange rise,
free cubic decay on offset channels and free quadratic decay on other processed
channels. See [TEP alignment and validation](TEP_ALIGNMENT.md) for source versions,
the exact channel decision and numerical comparisons.

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
[threshold guide and demo outcomes](../README.md#start-with-these-three-controls).

## Place in the pipeline

```mermaid
flowchart LR
 A[Pulse handling and filtering] --> B[Trial/channel cleaning and epoching]
 B --> C[Detrending and configured filling]
 C --> D[ICA and component rejection]
 D --> E[Host pipeline final processing and features]
```

Input arrays are **epochs x channels x samples**; times are **seconds relative
to TMS**, including negative baseline times. Amplitude units are retained.
MNE commonly uses volts; EEGLAB commonly uses microvolts; the toy uses arbitrary
units. The EEGLAB adapter converts EEG.times from milliseconds.

Pulse handling, filtering, whole-epoch resampling, ICA, re-referencing and
feature extraction remain in the host pipeline. Fit subsequent ICA on the
corrected data, rather than reusing a decomposition fitted before correction.

## Algorithm

1. Screen channels using the pipeline's epoch-mean z-slopes in pre, rise and
   decay. Threshold 0.5 identifies offset channels.
2. If offsets are found, fit pre through onset with degree 2, rise through the
   peak with Lagrange, and samples after peak with a free polynomial: degree 3
   for offset channels and degree 2 for other processed channels.
3. Assemble and subtract the trend. If onset is on the grid, both pre and rise
   include it while fitting; rise overwrites pre there during assembly.
4. Apply configured filling. Gaussian filling is enabled in both the library and the demos.

If no offsets are found, `overall_degree=1` applies the notebook's whole-epoch
linear detrending fallback, without filling. `None` / `[]` preserves the input
instead. Inspect diagnostics.operation and diagnostics.channel_selection.

Lagrange removes all signal in rise, including neural activity. A low fit
residual alone does not establish neural recovery. The original high-degree
monomial solve can leave numerical residuals; the plugin evaluates the exact
interpolation identity at the samples.

## Configuration

Config() and tmspath_config() have matching TEP defaults. All times are seconds.

| Setting | Default | Meaning |
|---|---|---|
| slope_threshold | 0.5 | Main decision threshold for extreme-artifact channels: reaching it enables windowed detrending; if all scores are lower, use the basic linear fallback. |
| mode | windowed | TEP workflow; global is an external comparator. |
| onset_s | 0 | Rise start; pre also includes this boundary. Choose for the recording. |
| peak_s | None / [] | Detect maximum absolute amplitude per epoch/channel; optionally supply a split. |
| peak_search_end_s | 0.020 | Search end, clipped by correction end. |
| end_s | None / [] | Fit decay through epoch end unless limited explicitly. |
| rise_model / decay_model | lagrange / poly | Exact rise interpolation and polynomial decay. |
| pre_degree | 2 | Quadratic pre fit; None / [] disables. |
| rise_degree | 3 | Only used by rise_model=poly; ignored by Lagrange. |
| decay_degree | 3 | Offset-channel degree; other polynomial decays use degree minus one, minimum zero. |
| constrain_rise / constrain_decay | false / false | Keep rise unconstrained for Lagrange; free decay matches the TEP notebook. |
| channels | None / [] | All input channels; select good EEG channels for processing. |
| channel_policy | auto | Slope screening; manual uses offset_channels; all marks every selected channel. |
| offset_channels | empty | Explicit offset indices when policy is manual. |
| only_offset_channels | false | Also process other selected channels. True preserves them. |
| overall_degree | 1 | No-offset whole-epoch fallback; None / [] skips. |
| mask_convention | pipeline | Pre <= onset; rise includes onset/peak; decay starts after peak. |
| fill_mode | gaussian | Alternatives: resample, none. |
| fill_margin_s / fill_seed | 0.005 / 42 | Replacement margin and reproducible native random stream. |
| global_model | biexp | Only used in global comparator mode, with fill_mode=none. |
| fit_offset | false | Optional constant for exponential models. |
| tau_min_s / tau_max_s | 0.0005 / 1 | Exponential bounds, inactive for polynomial fits. |
| grid_size / refinements | 21 / 5 | Exponential search settings. |

An off-grid manual peak resolves to the last sampled rise time at or before
the request. Diagnostics give the actual sampled split. Rise needs at least
one sample, each exponential segment four, and a polynomial segment at least
max(4, degree+2). Automatic screening needs two samples in every window.
Invalid inputs raise errors rather than silently selecting another model.

## Interactive toy

Python: open [demo_windowed_detrend.ipynb](PythonPlugin/demo_windowed_detrend.ipynb)
using [ENVIRONMENT.md](PythonPlugin/ENVIRONMENT.md). MATLAB: add MatlabPlugin to
the path, open [demo_windowed_detrend.mlx](MatlabPlugin/demo_windowed_detrend.mlx)
and select Run All.

Section A puts slope_threshold first, temporal windows second, and pre/rise/post
models third. Section B generates 16 EEG channels and 12 stimuli with channel
latency differences, stimulus jitter, amplitude variation and noise. The fit
uses channel_policy='auto', 0-20 ms peak search and all library defaults,
including fill_mode='gaussian'. It displays only **Polynomial** and evaluates five thresholds
on the same data, including the no-offset linear fallback. The detected peak
can differ from the true artifact peak because it sees neural activity/noise.

For experimental recordings, enable fill_mode='gaussian', choose suitable
windows and use auto screening or an explicit manual channel classification.

## Python/MNE

Add scripts/PythonPlugin/src to the Python path, or install locally with
`python -m pip install .` from scripts/PythonPlugin. NumPy is the core dependency;
the adapter additionally requires MNE.

```python
from windowed_detrend import Config, detrend_mne
cfg = Config(onset_s=0.010, peak_search_end_s=0.030)  # Choose for your recording.
corrected_epochs, trend, diagnostics = detrend_mne(epochs, cfg)
print(diagnostics['operation'], diagnostics['channel_selection']['offset_channels'])
```

MNE defaults to EEG channels excluding bads; explicit picks overrides the
selection. Indices are zero-based. Automatic screening depends on the channel
population, so use matching selections across environments.

## MATLAB/EEGLAB

The core needs base MATLAB. Add scripts/MatlabPlugin to the path. The adapter
accepts numeric epoched EEG.data in channels x samples x trials.

```matlab
cfg = tmspath_config();
cfg.onset_s = 0.010; cfg.peak_search_end_s = 0.030; % Choose for your recording.
cfg.channels = eeg_channel_indices; % One-based good EEG indices.
[EEGout, trend, diagnostics] = tmspath_eeglab(EEG, cfg);
disp(diagnostics.operation);
disp(diagnostics.channel_selection.offset_channels);
```

The adapter does not infer EEG/bad-channel types from EEGLAB metadata. Supply
the intended indices. The original dataset is preserved. Continue the host
pipeline using the corrected copy.

## Outputs and validation

corrected is the final signal; trend is the fitted model before filling.
data-trend recovers the unfilled residual. Pre fitting and the fill margin may
modify pre-onset samples. Unselected channels remain unchanged. See
[OUTPUTS.md](OUTPUTS.md), [FILLING.md](FILLING.md) and [TEP_ALIGNMENT.md](TEP_ALIGNMENT.md).

Native Gaussian draws differ across Python and MATLAB; masks and distribution
parameters agree. Deterministic fits and Fourier-resampled outputs are compared
sample by sample against the original functions and masks.

## Historical paper figures

![Conceptual example](images/paper_concept_computed.png)

The conceptual figure and older benchmark reports retain their recorded
configurations through Config.figure_defaults(...) and tmspath_config('figure').
Their scores do not describe the current TEP defaults. See [paper alignment](PAPER_ALIGNMENT.md),
[controlled branch comparison](PAPER_PLUGIN_COMPARISON.md) and the
[64-channel synthetic benchmark](SCALP64_BENCHMARK.md). The spatial projection
is illustrative, not a physiological forward model or experimental validation.
