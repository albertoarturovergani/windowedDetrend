# Alignment with the published TEP notebook

The Python and MATLAB defaults follow the **Windowed** configuration in
[`20260901_TEP_v1.ipynb`](https://github.com/albertoarturovergani/TMSpathPipeline/blob/534e926e659885d97bd873a7a31efbac17ebdd96/scripts/20260901_TEP_v1.ipynb),
TMSpathPipeline commit `534e926e659885d97bd873a7a31efbac17ebdd96`.
Unmodified function extracts and source hashes are in
`PythonPlugin/paper_reference/tep_functions.py` and `tep_provenance.json`.

## Verified decay constraint

The published notebook contains:

```python
detrend_fitConstraint = False if "nowind" not in typeOffsetDecay else True
```

For `typeOffsetDecay="wind_poly_3"`, this is **False**. True belongs to the
non-windowed configuration. Accordingly, `constrain_decay=False` / `false`
is the default. The constrained option remains available and is tested.

## Four aligned behaviors

| Element | Published TEP behavior now used in both plugins |
|---|---|
| Pre-onset detrending | Quadratic fit, `pre_degree=2`. |
| Decay constraint | Free polynomial fit, `constrain_decay=False`. |
| Sample boundaries | Pre `t <= onset`; rise `onset <= t <= peak`; decay `t > peak`. |
| Channel policy | Automatic slope screening; cubic decay for offset channels and quadratic decay for other processed channels. |

When onset is on the sampling grid, pre and rise both include it during fitting.
Assembly writes pre first, then rise, so the rise result wins at that sample,
exactly as in `computeDetrend_v6`. The peak belongs only to rise. Filling uses
these same masks for its source and target, including their endpoint samples.

Peak detection uses maximum absolute observed amplitude within the configured
search interval, independently for every epoch/channel. Default onset/search
end are 0 / 0.020 s, as in the notebook and current multichannel demo.
Choose them for the recording. A manually supplied off-grid peak uses the last
sample at or before the requested time; diagnostics give the actual sampled
split. `end_s` optionally limits the decay fit; its default is the epoch end.

## Channel screening and fallback

**Slope Threshold is the main decision parameter for recognizing channels with
extreme artifacts and choosing the detrending strategy.** A channel is
classified as extreme when its slope score reaches or exceeds the threshold.
If the threshold is higher than every channel score, the default is the basic
whole-epoch linear detrend. Otherwise, windowed detrending is enabled.

`channel_policy="auto"` / `'auto'` follows `computeSlopes_v4` and the numerical
decision in `check_detrend_need`:

1. Estimate linear slopes in pre, rise and decay for every epoch/channel.
2. Within each window, standardize slopes over selected channels and epochs
   with population SD. A zero SD produces zero z-scores.
3. Average z-scores across epochs for each channel. A channel is an offset
   channel if any window has `abs(mean_z) >= slope_threshold` (default 0.5).
4. Use `decay_degree` for offset channels and `max(decay_degree-1, 0)` for
   the other processed polynomial decays.

Electrode distances in the original reporting table do not affect selection.
The adapter/core channel selection defines the population being screened:
pass the same channels to both implementations for equivalent decisions.

`only_offset_channels=False` / `false` matches the notebook: other selected
channels are also processed. True leaves non-offset channels entirely unchanged.

If no offsets are found, the notebook disables windowed detrending and applies
a whole-epoch linear polynomial fit (`detrend_overall=True`,
`detrend_noWindowedOrder=1`). The plugin matches this with `overall_degree=1`;
no gap filling occurs in this branch. Set `overall_degree=None` / `[]` to
preserve the input instead. Invalid or undersampled windows still raise errors.

For known classifications, use `channel_policy="manual"` and
`offset_channels=(0, 2)` in Python, or `'manual'` and `[1 3]` in MATLAB.
`channel_policy="all"` treats all selected channels as known offset channels.
The interactive demo uses `auto` on 16 channels and 12 stimuli. Its threshold
sweep demonstrates both windowed detrending and the no-offset linear fallback.
The older minimal `toy_post_epoching.*` examples explicitly use `all`.

## Filling and intentional numerical differences

The library default is Gaussian filling; Fourier resampling of the original
pre-onset source and no filling are alternatives. Both demos keep
`fill_mode="gaussian"` / `'gaussian'` and label the result **Polynomial**.
The zeroed rise plus margin is replaced with Gaussian noise.
Details: [FILLING.md](FILLING.md).

Lagrange is evaluated exactly at its input nodes, so the unfilled rise is zero.
The original code solves a potentially ill-conditioned degree-N-1 monomial
polynomial and can leave residuals or fail. Comparisons report that difference
rather than reproducing instability. With filling enabled, replacement removes
this difference from the final signal.

Native Python and MATLAB Gaussian draws differ; source/target masks,
distribution parameters and seeded repeatability agree. Deterministic outputs
and fitted trends are compared sample by sample. Optional exponential solvers
remain adaptations and are not used by the default polynomial workflow.

Pulse handling, filtering, whole-epoch resampling, ICA, bad-channel interpolation,
re-referencing and feature extraction remain in the host pipeline. Derivative
peak detection, optimized order selection and additional spline/multi-exponential
models are not part of this fixed Lagrange/polynomial preset.

## Reproduce the comparisons

Use Python 3.12+ to execute the original function extracts; the plugin core
still supports Python 3.10+. Install optional validation dependencies:

```bash
cd scripts/PythonPlugin
python -m pip install ".[validation,benchmark]"
python -m unittest discover -s tests -v
python validate_filling.py
python validate_tep_pipeline.py
```

Then in MATLAB, from `scripts/MatlabPlugin`:

```matlab
test_plugin; test_filling; test_tep_alignment;
run_fill_validation;
run_tep_validation;
```

The TEP comparison executes the original mask builder, slope routine, channel
decision and full `computeDetrend_v6` on identical synthetic inputs. It covers
12 cases at 1000 and 1024 Hz: automatic/manual selection, both decay-constraint
settings, all three fill modes, offset-only processing and both no-offset
fallback settings. Only storage/plot stand-ins are used. Reports:

- [Python versus published TEP](results/tep_alignment/python_report.json)
- [MATLAB versus Python and TEP](results/tep_alignment/matlab_report.json)
- [Filling/Fourier checks](results/filling_validation/python_report.json)

The current run passed all 24 Python unit tests and the MATLAB test suites.
Across the 12 TEP cases, MATLAB checked 110 original window masks; the maximum
Python/MATLAB trend and deterministic-output difference was 1.21e-13 a.u.
The Python/original difference outside unfilled Lagrange rise was 1.78e-13 a.u.;
the original ill-conditioned unfilled rise left up to 0.002715 a.u. residual
on these inputs. Gaussian parameter differences were below 7e-18 a.u.
The multichannel notebook and MATLAB demo also pass matched-input checks:
16 channels, 12 epochs, five thresholds, identical selected channels and branch
decisions. With Gaussian filling enabled, fitted trends and samples outside
replacement agree within 1.32e-13 a.u. Replacement masks and Gaussian parameters
agree; native random samples and recovery MSE need not match exactly. Current
MSE values are recorded in each demo report. Latency/amplitude jitter is identical
across languages; synthetic input waveforms agree within floating-point error.

The filling suite additionally covers 18 cases / 72 traces and 144 Fourier
cases. Waveform fixtures are excluded from Git and regenerated by the commands
above. These are synthetic numerical checks, not evidence of neural recovery
on experimental recordings.

## Historical figures and migration

Existing figure/scalp results use explicit historical settings:
`Config.figure_defaults(...)` and `tmspath_config('figure')`. That profile
preserves pre-onset data, puts peak in decay, constrains decay and treats all
selected channels as offset channels, without filling. Historical metrics do
not describe the new TEP defaults. [PAPER_PLUGIN_COMPARISON.md](PAPER_PLUGIN_COMPARISON.md)
remains a controlled numerical-branch comparison, separate from the full TEP check.
