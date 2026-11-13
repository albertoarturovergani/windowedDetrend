# windowedDetrend

Python and MATLAB tools for windowed detrending of residual capacitive artifacts in epoched TMS-EEG recordings.

The algorithm separates the artifact into rise and decay windows, fits each window independently with exponential or polynomial functions, and subtracts the assembled trend. The default uses **Lagrange rise interpolation and degree-3 polynomial decay**. A non-windowed bi-exponential is provided only as an external comparator.

## Conceptual framework

![Conceptual framework of windowed detrending](scripts/images/paper_concept_computed.png)

[Figure PDF](scripts/images/paper_concept_computed.pdf) | [Numerical source](scripts/reconstruct_paper_concept.py) | [Method and interpretation](scripts/POST_EPOCHING_GUIDE.md)

The figure is a computed conceptual example, not a universal ranking of fitting methods. Lagrange interpolation reproduces every observed rise sample, so subtraction leaves zero in that window before the subsequent filling step, including removal of any neural activity. See the guide for model assumptions and comparator limitations.

## Place in the preprocessing pipeline

```mermaid
flowchart LR
    A[Pulse artifact handling] --> B[Bad trial and channel removal]
    B --> C[Epoching]
    C --> D[Windowed detrending]
    D --> E[ICA and component rejection]
    E --> F[Feature extraction]
```

Input to the core functions is an array of **epochs x channels x samples**, with times in **seconds relative to TMS**. Amplitude units and dimensions are preserved. MNE and EEGLAB adapters are also included.

## Python: run without installing the plugin

Download or clone this repository. From its root, create the environment and open the demonstration notebook:

```bash
cd scripts/PythonPlugin
conda env create -f environment.yml
conda activate windowed-detrend
python -m jupyterlab demo_windowed_detrend.ipynb
```

The notebook imports the local source automatically. Edit sections **A: toy signal** and **B: detrending parameters**, then run all cells.

- [Python notebook with toy signal and plots](scripts/PythonPlugin/demo_windowed_detrend.ipynb)
- [Environment setup, including the venv alternative](scripts/PythonPlugin/ENVIRONMENT.md)
- [Python API and tests](scripts/PythonPlugin/README.md)

For an existing pipeline, use a generic path to your downloaded copy:

```python
import sys
from pathlib import Path

plugin_src = Path("/path/to/windowedDetrend/scripts/PythonPlugin/src")
sys.path.insert(0, str(plugin_src))
from windowed_detrend import Config, detrend

cfg = Config(rise_model="lagrange", decay_model="poly", decay_degree=3,
             refinements=5)
# data and times come from your already preprocessed, epoched recording.
corrected, trend, diagnostics = detrend(data, times, cfg)
```

## MATLAB: run directly from the downloaded files

Open the Live Script and select **Run All**. Edit the opening toy-signal and detrending parameter sections to explore the alternatives.

```matlab
plugin_dir = fullfile('/path/to/windowedDetrend', 'scripts', 'MatlabPlugin');
addpath(plugin_dir);
open(fullfile(plugin_dir, 'demo_windowed_detrend.mlx'));
```

- [MATLAB Live Script](scripts/MatlabPlugin/demo_windowed_detrend.mlx)
- [Readable MATLAB demonstration source](scripts/MatlabPlugin/demo_windowed_detrend_source.m)
- [MATLAB API, EEGLAB adapter and tests](scripts/MatlabPlugin/README.md)

For an existing epoched array:

```matlab
cfg = tmspath_config();
cfg.rise_model = 'lagrange';
cfg.decay_model = 'poly';
cfg.decay_degree = 3;
cfg.refinements = 5;
[corrected, trend, diagnostics] = tmspath_detrend(data, times, cfg);
```

The core requires no additional MATLAB toolbox. The demonstration source was tested with MATLAB R2023b. Run the Live Script to generate its inline outputs. Neither PyPI nor MATLAB File Exchange installation is required.

## Outputs and fitted parameters

The functions return the corrected signal, estimated trend, and per-epoch/channel diagnostics. Exponential fits expose fitted taus, amplitudes and optional offsets; polynomial fits expose coefficients in normalized local time; Lagrange returns its interpolation nodes. See [Outputs, parameter interpretation and Python/MATLAB examples](scripts/OUTPUTS.md) for field names, units, saving results, and the distinction between fit residual and neural-recovery MSE.

## Parameters and evaluation

### Shared synthetic dataset: 64 scalp channels

The multichannel Python/MATLAB benchmark contains **214 epochs x 64 EEG
channels x 2,500 samples**, sampled at **10 kHz** over 0-249.9 ms. It combines
a known **50 Hz damped neural oscillator**, a capacitive artifact and optional
Gaussian noise. BioSemi 64 labels and idealized spherical electrode positions
define smooth, distinct neural/artifact gains across channels; noise is
independent per case and channel.

The 214 epochs comprise **200 grid cases** (two artifact families, each with
10 rise constants from 1-50 ms and 10 decay constants from 25-250 ms), **two
noiseless 3/200 ms extreme cases**, and **12 noisy extreme cases** (SD 0.5 and
2 a.u., three replicates per level and family). The current Lagrange/cubic
default plus four comparator methods yield **68,480 reconstructed channel
traces**. Both languages receive exactly the same saved arrays, known artifact
peaks and fit settings. Seed 42 makes the noise reproducible.

This dataset provides a controlled multichannel validation with the channel
count and electrode layout of a 64-channel cap. Its spatial gains are
illustrative; it does not include a biophysical head model or real EEG
recordings. Full construction, file shapes, metrics and reproduction commands
are in [the 64-channel benchmark guide](scripts/SCALP64_BENCHMARK.md).
Figures, per-channel metrics and validation reports are in
`scripts/results/scalp64`. The full 64-channel waveform dataset is generated
locally and excluded from Git: `fixtures/*.mat`, `metadata.mat` and the native
MATLAB `case_*_results.mat` waveform files. The generator, seed, parameters,
CSV/JSON reports and binary `matlab_metrics.mat` summary are versioned.
The historical single-channel benchmark and figures retain their original results.

The complete 64-channel comparison passed on all **68,480 reconstructed
traces**, with a maximum Python/MATLAB absolute difference of **6.16e-12 a.u.**
(tolerance 1e-6). The guide links the full per-channel metrics and validation
reports.

To regenerate the complete dataset after cloning, from `scripts/PythonPlugin`:

```bash
python run_scalp_benchmark.py --output ../results/scalp64 --workers 8
```

Then, from `scripts/MatlabPlugin`:

```matlab
run_scalp_benchmark('../results/scalp64', '../results/scalp64/matlab')
```

### Core parameters and further comparisons

The default onset is 10 ms, the peak is detected automatically within the search interval, and correction extends to the epoch end. Inspect and adjust these settings for your data. Five search refinements apply to optional exponential fits; polynomial fits do not use this setting.

- [Complete defaults, model options, algorithm flowchart and usage examples](scripts/POST_EPOCHING_GUIDE.md)
- [Direct comparison with the original paper's numerical branches](scripts/PAPER_PLUGIN_COMPARISON.md)
- [Supplementary Figure 1, panels A-F, and Python/MATLAB results](scripts/PLUGIN_RESULTS.md)
- [Method alignment and numerical differences](scripts/PAPER_ALIGNMENT.md)
- [Interactive supplementary dashboard](scripts/results/supplement1/dashboard.html): download the repository and open this HTML file locally.

Artifact-fitting MSE and recovery MSE against the known clean oscillator answer different questions. A small fit residual does not alone demonstrate preservation of neural activity. The standalone plugins are adaptations of the original numerical workflow, with documented solver and boundary differences.

## Acknowledgements

This work was developed within the TMSpath clinical research project, ClinicalTrials.gov identifier `NCT06284291`, supported by the Italian Ministry of Health under grant `RF-2021-12372804`.

Principal Investigator of the TMSpath project: **Prof. Simona Balestrini**.

The associated EEG and TMS-EEG preprocessing pipeline is available in the [TMSpathPipeline repository](https://github.com/albertoarturovergani/TMSpathPipeline).

## Citation and license

See the [citation metadata](scripts/MatlabPlugin/CITATION.cff) and [AGPL-3.0 license](LICENSE).

### Paper

If you use windowed detrending in your research, please cite:

Vergani, A. A., Peroni, G., Bruscagli, C., Lionti, A., Brandizzi, F., Lenge, M., Salvestrini, G., Jiménez-Jiménez, D., Casarotto, S., Rosanova, M., Mazzoni, A., Guerrini, R., Grippo, A., & Balestrini, S. (2026). **Effective correction of extreme capacitive artifacts in TMS-EEG via windowed detrending.** *Journal of Neural Engineering*, 23(3), 036009. [https://doi.org/10.1088/1741-2552/ae668e](https://doi.org/10.1088/1741-2552/ae668e).

## Experimental recordings

The Python and MATLAB demos disable filling (`fill_mode='none'`) to show the
detrending result. For experimental recordings, enable random filling of the
zeroed interval with `fill_mode='gaussian'` (the library default), and choose
the peak-search and correction windows for your data. See [filling options](scripts/FILLING.md).
