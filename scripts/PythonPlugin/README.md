# windowed-detrend (Python)

Standalone post-epoching TMS-EEG detrending. Local development release 0.1.0;
not yet published on PyPI. Source repository: https://github.com/albertoarturovergani/windowedDetrend

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
It includes 16-channel, 12-stimulus EEG with timing/amplitude jitter, automatic
screening, a threshold sweep, Polynomial detrending with Gaussian filling and plots. The recovered curve is labelled **Polynomial**.
Edit section A (threshold, windows, models), then B (synthetic EEG) if needed; restart
the kernel and run all cells. The main controls are ordered by importance; advanced settings retain their defaults.

The final figure places butterflies of **all observed channels before correction**
and **all corrected channels** side by side, for the same epoch. The two strongest
simulated artifact channels keep the same colors; other channels are grey.
Amplitudes remain in their original units, with separately labelled vertical
scales. The threshold figure retains one clean-versus-corrected channel example.
Legends are outside the signal axes.

See [ENVIRONMENT.md](ENVIRONMENT.md) for JupyterLab setup and launch commands.
The executable plain-text companion is
[demo_windowed_detrend_source.py](demo_windowed_detrend_source.py).

```python
from windowed_detrend import Config, detrend
cfg = Config(slope_threshold=0.5, onset_s=0.0, peak_search_end_s=0.020)
cleaned, artifact, diagnostics = detrend(data, times, cfg)
```

`data` must be `[epochs, channels, samples]`; `times` is in seconds.
Output amplitudes have the input unit. Arrays are copied. Set `channels=(0, 2)`
to select channels (zero-based); None selects all, empty tuple selects none.

`peak_s=None` detects the maximum absolute amplitude within the configured
search interval independently per trial/channel. Inspect that peak on real data:
physiological components or noise can move it. `end_s` limits correction;
The default pre-onset fit is quadratic (`pre_degree=2`); `None` disables this fit. The fill margin can still replace pre samples.
The default `constrain_decay=False` matches the Windowed TEP notebook; True optionally anchors to the observed peak.

Models: `lagrange`, `poly`, `singlerise`, `singledecay`, `doubledecay`, `biexp`, `alpha`.
Use `mode="global", global_model="biexp", fill_mode="none"` for the non-windowed comparator.
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

The current defaults follow the published Windowed TEP notebook: pre_degree=2, Lagrange rise including the peak, free decay (constrain_decay=false), automatic offset-channel screening, cubic decay on offset channels and quadratic decay on other processed channels. If no offsets are found, the default fallback is whole-epoch linear detrending. See [TEP alignment](../TEP_ALIGNMENT.md) for source provenance, configuration and comparisons.


## Experimental recordings

The Python and MATLAB demos keep Gaussian filling enabled (`fill_mode='gaussian'`),
matching the library default. The zeroed rise plus its margin is replaced with
Gaussian noise. For experimental recordings, choose
the peak-search and correction windows for your data. See [filling options](../FILLING.md).
