# Offset filling after windowed detrending

The default workflow is **Lagrange rise, cubic decay, automatic peak detection,
then Gaussian filling**. The core does not change the epoch's sampling rate.
Filling replaces the unreliable interval; it does not reconstruct its neural
activity. The default Gaussian rule follows TMSpathPipeline's
`apply_offset_correction` and `generate_noise_from_distribution`.

## Configure the workflow

```python
cfg = Config(
    onset_s=0.010,             # Set after your pulse-handling interval.
    peak_s=None,              # Detect per epoch/channel.
    peak_search_end_s=0.030,  # Example only: choose from your recordings.
    rise_model="lagrange", decay_model="poly", decay_degree=3,
    fill_mode="gaussian", fill_margin_s=0.005, fill_seed=42,
)
corrected, trend, diagnostics = detrend(data, times, cfg)
```

```matlab
cfg = tmspath_config();
cfg.onset_s = 0.010;
cfg.peak_s = [];                 % Automatic per epoch/channel.
cfg.peak_search_end_s = 0.030;    % Example; inspect your residual artifact.
cfg.fill_mode = 'gaussian';
cfg.fill_margin_s = 0.005;
cfg.fill_seed = 42;
[corrected, trend, diagnostics] = tmspath_detrend(data, times, cfg);
```

The library's default search end remains 0.100 s; the interactive demos use
0.030 s to illustrate a user-selected window. Search starts at `onset_s` and
ends at the smaller of `peak_search_end_s` and `end_s`. It finds the largest
absolute **observed** amplitude, which may differ from the true artifact peak.
Inspect the detected split. Set `peak_s` explicitly to override detection.
All times are seconds. The EEG sampling rate is unchanged.

| Parameter | Default | Meaning |
|---|---|---|
| `fill_mode` | `gaussian` | Random Gaussian replacement; alternatives `resample` and `none`. |
| `fill_margin_s` | 0.005 | Extend the sampled rise window by this duration on each side. Zero fills just the rise. |
| `fill_seed` | 42 | Integer 0 to 2^32-1. Change it for another reproducible noise realization. |

Filling works with windowed fits. The optional global comparator requires
`fill_mode='none'` / `"none"` because it has no rise window.

## Source and target intervals

The rise is `onset_s <= times < detected_or_supplied_peak`. The replacement
interval extends from the **first sampled rise time minus the margin** to the
**last sampled rise time plus the margin**, inclusive and clipped to the epoch.
It can therefore replace pre-onset and post-peak samples, including samples
beyond `end_s` when the margin reaches them. This matches the pipeline's mask
construction, rather than silently clipping the fill to the fitted interval.

The source is the original signal before `onset_s`, ending at its last sample
minus the same margin. If trimming would leave it empty, all pre-onset samples
are used, matching the pipeline. The source is taken **before detrending or
filling**, independently for each epoch/channel. It is not automatically
restricted to negative TMS times; upstream pulse handling and an appropriate
onset are required. If there are no pre-onset samples at all, the plugin raises
an explicit error rather than silently skipping the requested fill.

## Gaussian default

The replacement distribution is:

```text
mean  = mean(original source samples)
sigma = max(0.5 * population_standard_deviation(source), 1e-8)
```

The minimum sigma is in the input amplitude unit, exactly as in the pipeline.
It creates tiny noise even for a constant source; that floor is not invariant
under an arbitrary change of amplitude unit. This is uncorrelated Gaussian
noise with source-derived parameters, not preservation of the source spectrum.
Finite generated samples need not have exactly the requested sample mean/SD.

Python uses a local NumPy PCG64 generator seeded by
`SeedSequence([fill_seed, channel_index, epoch_index])`, matching the pipeline's
zero-based convention. MATLAB uses a local `RandStream('mt19937ar')`, with
`mod(fill_seed + 104729*(epoch-1) + 130363*(channel-1), 2^32)` as its trace seed.
Neither implementation changes the session's global random state. Channel
selection or processing order does not change a given trace's realization.

**The same seed is repeatable within each implementation; it does not produce
the same random samples across languages.** Cross-language checks compare
the fitted trend, source/target masks and Gaussian distribution parameters,
plus native repeatability and distribution checks. Exact sample equality is
appropriate for deterministic `resample`, not for these independent draws.

## Alternative modes

`resample` Fourier-resamples the original source to the number of replacement
samples. NumPy and base MATLAB implementations agree with the pipeline's
`scipy.signal.resample(..., window=None)`, including odd/even Nyquist handling.
No Signal Processing Toolbox or SciPy installation is needed by the core.
MATLAB's polyphase `resample` is not used, because it is a different algorithm.
Fourier resampling assumes a periodic source and can create edge oscillations;
it does not guarantee a smooth join to neighboring samples.

`none` disables replacement and returns the detrended waveform. Historical
paper comparisons and the 64-channel benchmark now explicitly select `none`
so their saved results remain reproducible. They test detrending before fill.

## Output contract and diagnostics

- `corrected` is the final, possibly filled signal.
- `trend` remains the fitted model before filling. `data - trend` recovers the
  detrended waveform **before replacement**. When filling is active,
  `corrected + trend == data` is generally false inside the replacement mask.
- `diagnostics.fits[k].fill` (MATLAB: `diagnostics.fits{k}.fill`) records mode,
  method, margin, inclusive source/target index bounds and times, sample counts,
  fallback use and actual replacement values. Gaussian records additionally
  contain `mean`, `std`, `seed` and RNG description (MATLAB also `trace_seed`).
- Indices are zero-based in Python and one-based in MATLAB. Fit coefficients,
  taus and segment `residual_mse` describe the fit **before replacement**.
- Unselected channels are unchanged. Outside the union of fitting and filling
  intervals, samples are unchanged (unless optional pre-onset detrending is on).

## Demos and validation

The notebook and Live Script use automatic peak detection and explicitly set
`fill_mode='none'` to demonstrate detrending without filling. For experimental
recordings, set `fill_mode='gaussian'` to enable random replacement of the
zeroed interval. The library default remains Gaussian filling. The original
no-background toy remains available as `toy_post_epoching.py` / `.m`.

Run the Python tests and generate compact shared fixtures:

```bash
cd scripts/PythonPlugin
python -m unittest discover -s tests -v
python validate_filling.py
```

Then, from `scripts/MatlabPlugin`:

```matlab
test_plugin; test_filling;
run_fill_validation;
```

`validate_filling.py` requires the optional benchmark dependencies. It uses
unmodified pipeline reference functions with recorded source provenance. The
suite covers Gaussian and resampled filling, manual/off-sample/automatic
peaks, source fallback, selected channels, pre-onset fits, and 144 Fourier
length/signal combinations. Reports are in `results/filling_validation`.
Generated waveform fixtures are excluded from Git. No real EEG is included.
