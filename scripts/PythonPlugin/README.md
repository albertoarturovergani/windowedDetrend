# windowed-detrend (Python)

Standalone post-epoching TMS-EEG detrending. Local development release 0.1.0;
not yet published on PyPI. Source repository: https://github.com/albertoarturovergani/windowedDetrend

## Environment and direct source use

See [ENVIRONMENT.md](ENVIRONMENT.md) for Conda/Miniforge and standard Python
instructions. `environment.yml` creates a Python environment with NumPy,
SciPy and Matplotlib; MNE is optional. The plugin can be imported directly
from its `src` folder without installing the plugin itself.

## Install locally (optional)

Python 3.10+; NumPy is the only core dependency:

```bash
python -m pip install .
python -m pip install ".[benchmark,mne]"
```

After a future PyPI publication, the intended command is `pip install windowed-detrend`.
The distribution name has not been reserved or verified as available.

## API

For an executable demonstration, open [demo_windowed_detrend.ipynb](demo_windowed_detrend.ipynb).
It includes the toy signal, polynomial detrending without filling, recovery metrics
and plots. The recovered curve is labelled **Polynomial**.
Edit the opening sections A (toy generation) and B (detrending), then restart
the kernel and run all cells. All core parameters, alternatives
and evaluation controls are explained in English at the beginning.
See [ENVIRONMENT.md](ENVIRONMENT.md) for JupyterLab setup and launch commands.

```python
from windowed_detrend import Config, detrend
cfg = Config(onset_s=0.010, peak_s=0.015,
             rise_model="lagrange", decay_model="poly",
             rise_degree=3, decay_degree=3, constrain_rise=False)
cleaned, artifact, diagnostics = detrend(data, times, cfg)
```

`data` must be `[epochs, channels, samples]`; `times` is in seconds.
Output amplitudes have the input unit. Arrays are copied. Set `channels=(0, 2)`
to select channels (zero-based); None selects all, empty tuple selects none.

`peak_s=None` detects the maximum absolute amplitude within the configured
search interval independently per trial/channel. Inspect that peak on real data:
physiological components or noise can move it. `end_s` limits correction;
`pre_degree=None` preserves baseline, otherwise fits a pre-onset polynomial.
`constrain_decay=True` anchors the decay to the observed signal at the split,
which includes any neural activity at that point. This can bias recovery.

Models: `lagrange`, `poly`, `singlerise`, `singledecay`, `doubledecay`, `biexp`, `alpha`.
Use `mode="global", global_model="biexp"` for the non-windowed comparator.
Exponential amplitudes and optionally offsets are fitted by least squares;
time constants use a bounded logarithmic grid plus refinement, not upstream
SciPy curve_fit. See PAPER_ALIGNMENT.md for important differences.

```python
from windowed_detrend import detrend_mne
clean_epochs, artifact, info = detrend_mne(epochs, cfg)
```

The optional MNE adapter selects EEG channels excluding bads by default.
The ndarray API performs no channel-type inference. The pulse artifact must
already have been handled upstream; no filtering, epoching, ICA or whole-epoch
resampling is performed here. For experimental recordings, see the filling note below.

## Benchmark and tests

### 64-channel scalp benchmark

```bash
python run_scalp_benchmark.py --output ../results/scalp64 --workers 8
```

This generates **214 epochs x 64 channels x 2,500 samples at 10 kHz**. Channels
use BioSemi 64 labels and idealized spherical positions, with distinct smooth
neural/artifact gain fields and independent channel noise. The clean signal
is a known 50 Hz damped oscillator. The 214 cases comprise two 10 x 10
artifact grids plus 14 extreme/noisy cases. The current Lagrange/cubic default
and four comparators produce **68,480 reconstructed channel traces**. Saved
per-epoch MAT fixtures allow MATLAB to process exactly the same inputs.
These waveform fixtures, `metadata.mat` and MATLAB `case_*_results.mat` arrays
are generated locally and excluded from Git. Run the command above after
cloning to recreate them; code, figures, CSV/JSON reports and the compact
`matlab_metrics.mat` summary remain versioned.

See [dataset construction, limits, file shapes and MATLAB commands](../SCALP64_BENCHMARK.md).
Worker count defaults to 1. This is an illustrative spatial simulation,
without a biophysical head model. The original single-channel benchmark below
and its historical figures retain their own results.

### Single-channel and original-paper comparisons

For a direct comparison against the original numerical branches on identical
toy arrays, run `python compare_paper_plugin.py`. Source extracts and hashes
are in `paper_reference`; the report is `../PAPER_PLUGIN_COMPARISON.md`.
Matched windows/anchors and native-convention differences are reported
separately. This is not an end-to-end reproduction of the original pipeline.

```bash
windowed-detrend-benchmark --output results --grid-size 10
python -m unittest discover -s tests -v
```

The benchmark writes CSV metrics, complete configurations, fit diagnostics,
figures and a shared MATLAB fixture. Primary metrics compare recovered signal
to known neural ground truth, never to the contaminated input. Grid cases use
known simulation peaks; separate extreme cases test automatic peak detection.
Includes both windowed and multiplicative bi-exponential artifacts, noisy
replicates, and decay polynomial degrees 1-8 on the extreme case.

## Citation and license

Vergani AA et al. (2026), Effective correction of extreme capacitive artifacts
in TMS-EEG via windowed detrending. DOI: 10.1088/1741-2552/ae668e.
AGPL-3.0; see LICENSE. Research software; this is a new standalone adaptation,
not a bit-for-bit reproduction of the paper's original numerical workflow.


Artifact-only sweep (supporting panel B):

```bash
python -m windowed_detrend.artifact_sweep --output ../results
```

Compares the notebook's four-parameter, free-onset SciPy solver with the
bounded standalone plugin. Stores convergence/physicality flags and MSE
between the isolated artifact and its fit. See PAPER_ALIGNMENT.md.


## Current defaults

Exponential time-constant searches use `refinements=5` by default (previously
3), retaining `grid_size=21`. This improves search resolution with additional
computation. It has no effect on Lagrange or polynomial fits. The current
notebook and direct paper comparison have been recomputed with this default.

Windowed polynomial detrending: `rise_model=lagrange`, `decay_model=poly`, `decay_degree=3`, `constrain_rise=false`, and decay anchored to the observed peak. Peak detection remains automatic unless a split is supplied. Lagrange degree is N-1 for N rise samples; `rise_degree` is ignored for this model. At the input nodes, the interpolant equals the input, so the rise window is zero before the subsequent filling step, including removal of any neural signal. Evaluation uses that exact identity rather than a high-degree monomial solve. The historical four-method benchmark explicitly retains its exponential and polynomial 3/3 and 3/5 configurations; it is not a benchmark of the new default. The updated toy demonstrates the new default models.


## Experimental recordings

The Python and MATLAB demos disable filling (`fill_mode='none'`) to show the
detrending result. For experimental recordings, enable random filling of the
zeroed interval with `fill_mode='gaussian'` (the library default), and choose
the peak-search and correction windows for your data. See [filling options](../FILLING.md).
