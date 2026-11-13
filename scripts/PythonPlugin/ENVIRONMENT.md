# Python environment

Run the commands below from `scripts/PythonPlugin` in your downloaded repository.
The plugin is used directly from source; installing the plugin or publishing it
on PyPI is not required.

## Option 1: Conda or Miniforge

```bash
conda env create -f environment.yml
conda activate windowed-detrend
```

The environment includes Python 3.12, NumPy (the core dependency), and SciPy
and Matplotlib (synthetic benchmarks and figures). To use MNE Epochs, uncomment
the MNE entry in `environment.yml` before creating the environment, or run:

```bash
conda install -c conda-forge "mne>=1.6"
```

If the environment already exists after editing its configuration:

```bash
conda env update -n windowed-detrend -f environment.yml
```

## Option 2: Standard Python on Windows

With Python 3.10 or newer installed, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional MNE support:

```powershell
.\.venv\Scripts\python.exe -m pip install "mne>=1.6"
```

Activation is not required when using the explicit interpreter path above.
Select that interpreter in your editor, or the `windowed-detrend` Conda
interpreter if using Option 1.

## Interactive demonstration notebook

`demo_windowed_detrend.ipynb` includes 16-channel, 12-stimulus synthetic EEG
with timing/amplitude jitter, automatic screening and a threshold sweep.
Plots are labelled Polynomial: quadratic pre, Lagrange rise and free
cubic/quadratic decay according to channel classification. Gaussian filling
of the zeroed rise plus margin is enabled. Saved outputs are included; run all cells to recompute them.
The executable plain-text companion is `demo_windowed_detrend_source.py`.

The Conda environment includes JupyterLab and ipykernel. For an existing
environment, update it with `conda env update -n windowed-detrend -f environment.yml`.
From `scripts/PythonPlugin`, with that environment activated:

```bash
python -m jupyterlab demo_windowed_detrend.ipynb
```

For the venv alternative, from the same directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyterlab demo_windowed_detrend.ipynb
```

Select the matching Python environment as the notebook kernel. The multichannel notebook orders controls as threshold, search windows and
pre/rise/post models; it shows channel decisions across five thresholds. It
locates the source relative to the working directory, without personal paths
or plugin installation.

## Source-path setup

In your Python pipeline, set the absolute path to the downloaded source folder:

```python
import sys
from pathlib import Path

plugin_src = Path(r"C:\path\to\windowedDetrend\scripts\PythonPlugin\src")
sys.path.insert(0, str(plugin_src))

from windowed_detrend import Config, detrend

cfg = Config(slope_threshold=0.5, onset_s=0.0, peak_search_end_s=0.020)
# data: epochs x channels x samples; times: seconds.
# Pulse handling, bad trial/channel removal and epoching precede this step.
cleaned, artifact, diagnostics = detrend(data, times, cfg)
# Continue with ICA and feature extraction as appropriate.
```

The configuration class is named `Config`. The default Lagrange interpolant
matches every rise sample, so subtraction also removes neural activity in that
window. See the post-epoching guide for the full interpretation of the defaults.

For a quick import check, from `scripts/PythonPlugin` run:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from windowed_detrend import Config, detrend; print(Config())"
```

For Option 2, replace `python` with `.\.venv\Scripts\python.exe`.
The environment file specifies dependency ranges, not an exact lockfile.
It does not install the plugin itself. To record versions used in an experiment:

```bash
conda env export -n windowed-detrend > environment-resolved.yml
```

Or, with Option 2:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-resolved.txt
```

## Published TEP reference checks

Use Python 3.12+ and `python -m pip install ".[validation,benchmark]"` for
`validate_tep_pipeline.py`. The Conda environment includes pandas for these
reference comparisons; pandas and SciPy are not core dependencies. See
[TEP_ALIGNMENT.md](../TEP_ALIGNMENT.md).
