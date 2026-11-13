# Paper alignment and scientific interpretation

## Sources inspected

- Target repository: https://github.com/albertoarturovergani/windowedDetrend
  commit 83fcec4a1a7d57d6a435fee00576e5eb65e42a00 (AGPL-3.0).
- TMSpathPipeline commit 534e926e659885d97bd873a7a31efbac17ebdd96:
  scripts/tmspath_utils.py, computeDetrend_v6, computeTimeMasks,
  add_exp_artifact and damped-oscillator generation (GPL-3.0).
- Local paper: draftPaperMethods/SUBMISSION/JNE/Revisions/R2/ae668e.pdf,
  Figure 1, sections 2.1 and 2.3.2. This file contains proof-style journal
  headers (volume 0 / xxxxxx); it is not assumed to be the final typeset file.
- Same R2 directory: supporting_material.docx, supporting Figures 1, 3 and 4;
  captions and embedded Figure 1/3 images inspected.
- Local figure notebook:
  draftPaperMethods/windowedDetrend/scripts/old/2025_12_01_FigurePaper.ipynb,
  source cells only (not its saved output as executable instructions).

## Synthetic equations and parameters

Neural signal, u = max(t - 0.010, 0) seconds:

    s(t) = 5 exp(-25 u) sin(2 pi 50 u)

Windowed artifact (the figure-notebook construction):

    a(t) = 0, t < t0
    a(t) = -100 (1 - exp(-(t-t0)/tau_rise)), t0 <= t < tp
    a(t) = -100 (1 - exp(-(tp-t0)/tau_rise)) exp(-(t-tp)/tau_decay), t >= tp

with t0=10 ms, tp=15 ms. Time vector 0..249.9 ms, step 0.1 ms (10 kHz).
Amplitudes are arbitrary units; figures must not claim calibrated microvolts.
Grid: 10 equally spaced rise taus from 1 to 50 ms and 10 decay taus from 25
to 250 ms. Extreme case 3/200 ms is added explicitly because it is not on
that grid. Supplementary Fig.1 confirms these ranges and the extreme peak.

Second artifact family (pipeline add_exp_artifact), an extension of the paper
figure scenario: -100 (1-exp(-u/tau_rise)) exp(-u/tau_decay).
Its known peak is t0 + tau_rise*log(1+tau_decay/tau_rise), not 15 ms.

## Fits and deliberate differences

- Implemented standalone, trial/channel-wise core, with selectable exponential
  rise/decay, polynomial, double decay, global biexponential and alpha bases.
- Paper section 2.1 describes cubic polynomials; section 2.3.2 describes
  Lagrange/optimized rise and fixed cubic decay. The old figure notebook uses
  cubic rise and quintic decay. Both cubic/cubic and cubic/quintic comparators
  are included. Lagrange rise interpolation of degree N-1 is now implemented and is the default, with cubic decay. Automatic optimized order selection remains unported. The interpolant is evaluated exactly at its input nodes, rather than using the original high-degree monomial polyfit; the rise residual is therefore zero.
- Numerical solver is bounded log-tau grid plus local grid refinement,
  with least-squares linear amplitude elimination (variable projection).
  The paper uses scipy.optimize.curve_fit nonlinear least squares.
  Thus this is a new controlled implementation, not bitwise replication.
  More robust fitting can improve the global comparator relative to the old
  notebook. A result favoring the global fit must not be suppressed or treated
  as a reproduction of the old reported ranking.
- Defaults: no exponential offset, zero rise anchor, observed peak decay anchor.
  Offsets and constraints are configurable. The peak anchor contains neural
  activity; early components can therefore be distorted or removed.
- Masks are disjoint: pre t<onset, rise onset<=t<peak, decay t>=peak.
  Supporting Fig.3 instead assigns the peak sample to rise; this one-sample
  convention follows the figure notebook. No sample belongs to both fitting masks; subsequent filling is a separate replacement step.
- Automatic peak: maximum absolute observed amplitude within the search window.
  Main grid uses known simulation peaks (oracle segmentation) to isolate fit
  behavior. Extreme-case automatic-peak tests are separately labeled, so the
  grid cannot be interpreted as an end-to-end peak-detection validation.
- Gaussian gap filling is now enabled by default, with optional Fourier-resampled filling or no fill. Historical benchmark configurations explicitly disable filling. No pulse removal, filtering, ICA, lag correction, whole-epoch resampling or data-driven channel screening is performed.
  These pipeline steps can change recovery and should be evaluated separately.
- Failures raise explicit errors rather than silently substituting a model.

## Metrics and controls

MSE_recovery = mean((recovered - neural_ground_truth)^2), evaluated over
10..249.9 ms, identical interval for all methods. NMSE divides by neural
mean-square power. MSE before correction, correlation, early (10..30 ms)
and late (30..249.9 ms) MSE and residual power are exported separately.
Residual power is NOT the reconstruction MSE. A large percent reduction
relative to a huge artifact does not establish accurate neural recovery.

Core tests include artifact-only exact recovery, preservation of unselected
channels and outside-window samples, amplitude-unit invariance, malformed
input rejection, and model coverage. The batch adds Gaussian noise at SD
0.5 and 2 a.u. (three replicates each) for both extreme artifact families.
Seed 42 and saved inputs enable exact Python/MATLAB comparisons, independent
of the languages' different random generators. Degrees 1-8 are compared on
synthetic extreme cases; this is not a replication of supporting Fig.4's real
TEP sensitivity analysis. No neural truth is supplied to the fitting routine.

The tau search range is 0.5 ms..1 s. Boundary flags and full coefficients are
saved in fit_diagnostics.json. Rank-deficient fits fail rather than appearing
successful. Finite grids approximate the optimum; this is not a guarantee
of an exact continuous nonlinear optimum or of unbiased neural recovery.

## Original-solver diagnostic on the extreme case

The notebook-style global curve_fit call, run with the notebook's milliseconds
and initial parameters [5,20,100] on the extreme windowed case, converged to
an unphysical negative decay time (-2159.82 ms), with a covariance warning.
Its recovery MSE was 257.112 a.u.^2. The new bounded solver gives 0.676874
on the same case. This comparison supports treating solver initialization
and physical bounds as experimental factors, separate from windowing.
Full fitted parameters and clean-only distortion controls are recorded in
results/additional_validation.json. It is a single-case diagnostic, not a
reproduction of the entire published comparison.

## Supporting Figure 1, panel B: isolated artifact sweep

Run `python -m windowed_detrend.artifact_sweep --output ../results` (or the
installed `windowed-detrend-artifact-sweep` entry point). This separately fits
202 isolated artifacts (100 grid pairs plus explicit 3/200 ms extreme for each
family), with no neural waveform and no noise, using two solvers (404 fits).

The **panel B notebook** fits four parameters in milliseconds with
`curve_fit(biexp, t_fit, artifact, p0=[1,50,100,10])`: amplitude, rise tau,
decay tau, and FREE onset. It uses the clipped-exponential model and the
SciPy default evaluation limit. This is different from the three-parameter,
fixed-onset diagnostic for the contaminated signal documented above.

The **bounded plugin** fixes onset at 10 ms and searches positive taus with
variable projection. Both are evaluated on identical samples t>10 ms through
249.9 ms (2399 samples). MSE is between isolated artifact and fitted trend,
not between a recovered waveform and a neural ground truth. The first family
matches the windowed construction of the supporting figure; the multiplicative
biexponential family is an additional control. Optimization results can vary
with SciPy version; the reproduction follows source code, not saved image pixels.

Failures have null MSE and a recorded error, not the notebook's 1e4 sentinel.
Converged fits with nonpositive taus are retained and explicitly flagged.
`artifact_only_metrics.csv`, `artifact_only_summary.json` and full diagnostics
retain the results. The updated dashboard uses a common LINEAR color scale for
the two families (the paper uses a log scale) and separates this experiment
from neural recovery. `run_artifact_sweep` checks the bounded fits in MATLAB;
the original SciPy solver is a Python-only historical comparator.

## Current exponential-search default

The Python and MATLAB cores now use 5 refinements (previously 3), retaining
21 initial tau candidates. This affects optional exponential models only;
Lagrange and polynomial fits are unchanged. The interactive examples and
PAPER_PLUGIN_COMPARISON.md have been recomputed. Older benchmark numbers in
this alignment document describe their recorded historical configurations,
not the new default. The direct comparison retains 3, 5 and 7 refinements as
explicit controls.

See [FILLING.md](../FILLING.md) for current filling defaults and validation.
