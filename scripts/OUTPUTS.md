# Outputs and fitted parameters

Both core functions return three outputs:

```python
corrected, trend, diagnostics = detrend(data, times, cfg)
```

```matlab
[corrected, trend, diagnostics] = tmspath_detrend(data, times, cfg);
```

| Output | Meaning |
|---|---|
| `corrected` | Input minus the estimated trend. This is the array to inspect and pass to subsequent preprocessing. |
| `trend` | Estimated artifact/trend at every input sample, in the same amplitude units as the input. |
| `diagnostics` | Configuration actually used and the fitted parameters for each processed epoch/channel. |

Core arrays have shape epochs x channels x samples. Unselected channels have zero trend and unchanged corrected data. With the default `pre_degree`, samples outside the correction interval also remain unchanged. Outputs are returned in memory; the core does not automatically save files.

`detrend_mne` returns a corrected copy of the MNE Epochs object, an array of trends in core layout, and diagnostics. `tmspath_eeglab` returns a corrected EEG structure, a trend array in channels x samples x trials layout, and diagnostics. Diagnostic indices still refer to epochs and channels.

## Where the fitted values are stored

`diagnostics.config` (MATLAB) or `diagnostics["config"]` (Python) stores the effective configuration. `fits` contains one record per processed epoch/channel, in epoch order and then selected-channel order. It is a Python list of dictionaries or a MATLAB cell array of structs, not a dense epoch/channel matrix.

Each windowed record contains `epoch`, `channel`, `peak_s`, `rise`, and `decay`. Indices are zero-based in Python and one-based in MATLAB. `peak_s` is the split time in seconds relative to TMS, either supplied by the user or detected from the maximum absolute input amplitude. It is not optimized jointly with the time constants. An optional pre-onset fit is stored in `pre`; external-comparator mode stores `global` instead of `rise`, `decay`, and `peak_s`.

| Segment field | Meaning |
|---|---|
| `model` | Model used for this window. |
| `tau_s` | Fitted time constants in seconds; empty for polynomial and Lagrange models. Multiply by 1000 for ms. |
| `coefficients` | Linear coefficients in the model-specific order below; empty for Lagrange. |
| `residual_mse` | Mean squared difference between the observed input and the fitted trend within this segment, in squared input units. |
| `anchor` | Required trend value at local time zero, if constrained; otherwise Python `None` or MATLAB `[]`. |
| `scale_s` | Time normalization used for polynomial coefficients, in seconds. |
| `tau_at_bound` | Whether any fitted tau lies close to a configured search bound. This is a diagnostic flag, not a convergence certificate. False for polynomial and Lagrange models. |

### Exponential fits: what is optimized?

The solver searches time constants on a logarithmic grid within `tau_min_s` and `tau_max_s`. For each candidate, it estimates linear amplitudes (and the optional offset) by least squares, respecting any start anchor. It retains the smallest segment residual MSE and refines the tau grid around the best candidate. Defaults are 21 initial values per tau and 5 refinements, each with 7 values per tau.

This returns the best candidate encountered by this finite search, not a guaranteed continuous global optimum. The fitted taus describe the contaminated input under the selected model and constraints; they need not equal the physical artifact constants or the known toy constants.

Let `x` be time since the start of the segment. The rise origin is `cfg.onset_s`; the decay origin is the record's `peak_s`.

| Model | Function without optional offset | `tau_s` order | `coefficients` order |
|---|---|---|---|
| `singlerise` | `A * (1 - exp(-x/tau))` | `[tau]` | `[A]` |
| `singledecay` | `A * exp(-x/tau)` | `[tau]` | `[A]` |
| `doubledecay` | `A1 * exp(-x/tau1) + A2 * exp(-x/tau2)` | `[tau1, tau2]`, with tau1 < tau2 | `[A1, A2]` |
| `alpha` | `A * (x/tau) * exp(-x/tau)` | `[tau]` | `[A]` |
| `biexp` | `A * (1 - exp(-x/tau_rise)) * exp(-x/tau_decay)` | `[tau_rise, tau_decay]` | `[A]` |

With `fit_offset=True` / `true`, an additive constant `b` is appended as the last coefficient for exponential models. Constraints can determine some amplitudes from the anchor; they are still included in the returned coefficient vector. Amplitudes and offsets use the input amplitude unit. The non-windowed bi-exponential remains an external comparator, not a step in the windowed algorithm.

No confidence intervals, covariance matrix, standard errors, optimizer trajectory, iteration log, or solver exit code are currently returned. Invalid inputs or absence of a full-rank fit cause an error rather than a successful record with a failure flag.

### Polynomial fits: coefficients and degree

For degree `d`, the returned coefficients are **ascending powers**:

```text
[c0, c1, ..., cd]
u = x / scale_s
trend(x) = c0 + c1*u + ... + cd*u^d
```

These coefficients use normalized local time, not absolute epoch time. Since `u` is dimensionless, all coefficients have the input amplitude unit. Coefficients for powers of time in seconds would be `ck / scale_s^k`. The degree is selected by the user, not optimized; for a polynomial record it is `len(coefficients)-1` in Python or `numel(coefficients)-1` in MATLAB. The configuration also records the selected degrees. Polynomial records do not currently contain a separate `degree` field.

For the default cubic decay, the four values are `[c0, c1, c2, c3]`. With `constrain_decay=True` / `true`, `c0` equals the observed signal interpolated at the split (`anchor`); the remaining coefficients are fitted by least squares. No tau is fitted, and tau-search settings are inactive.

To reconstruct the fitted decay from its diagnostics:

```python
import numpy as np
rec = diagnostics["fits"][0]
fit = rec["decay"]  # Default model: poly
end_s = cfg.end_s if cfg.end_s is not None else times[-1]
mask = (times >= rec["peak_s"]) & (times <= end_s)
u = (times[mask] - rec["peak_s"]) / fit["scale_s"]
reconstructed = np.polynomial.polynomial.polyval(u, fit["coefficients"])
np.testing.assert_allclose(reconstructed, trend[rec["epoch"], rec["channel"], mask],
                           rtol=1e-10, atol=1e-10)
```

```matlab
rec = diagnostics.fits{1};
fit = rec.decay; % Default model: poly
end_s = cfg.end_s; if isempty(end_s), end_s = times(end); end
mask = times >= rec.peak_s & times <= end_s;
u = (times(mask) - rec.peak_s) / fit.scale_s;
% MATLAB polyval expects descending powers, so reverse the coefficients.
reconstructed = polyval(flipud(fit.coefficients(:)), u);
reference = reshape(trend(rec.epoch,rec.channel,mask), [], 1);
assert(max(abs(reconstructed(:) - reference)) < 1e-8);
```

### Lagrange rise: interpolation nodes, not optimized taus

The default rise is an interpolant through all observed rise samples. Additional fields are `degree` (N-1), `node_times_s` (relative to the segment origin), and `node_values` (observed amplitudes). `coefficients` and `tau_s` are empty because no monomial coefficient vector or tau is estimated.

At the input nodes the trend equals the observed signal exactly. Consequently `residual_mse=0` and the corrected rise is zero, including any neural component and noise. This is an interpolation property, not evidence of successful neural recovery.

## Inspect the outputs in Python and MATLAB

For the default Lagrange/cubic model:

```python
rec = diagnostics["fits"][0]
print("Epoch/channel:", rec["epoch"], rec["channel"])
print("Split (ms):", 1000 * rec["peak_s"])
print("Rise interpolation degree:", rec["rise"]["degree"])
print("Decay coefficients [c0, c1, c2, c3]:", rec["decay"]["coefficients"])
print("Decay time scale (s):", rec["decay"]["scale_s"])
print("Decay fit residual MSE:", rec["decay"]["residual_mse"])
```

```matlab
rec = diagnostics.fits{1};
disp([rec.epoch, rec.channel]);
disp(1000 * rec.peak_s); % Split in ms
disp(rec.rise.degree);
disp(rec.decay.coefficients); % Ascending powers
disp(rec.decay.scale_s); % Seconds
disp(rec.decay.residual_mse);
```

To inspect separate exponential rise/decay fits, rerun with the alternative configuration:

```python
from dataclasses import replace
exp_cfg = replace(cfg, rise_model="singlerise", decay_model="singledecay")
corrected_exp, trend_exp, diag_exp = detrend(data, times, exp_cfg)
for rec in diag_exp["fits"]:
    for segment in ("rise", "decay"):
        fit = rec[segment]
        print(rec["epoch"], rec["channel"], segment,
              "tau (ms):", np.asarray(fit["tau_s"]) * 1000,
              "coefficients:", fit["coefficients"],
              "fit residual MSE:", fit["residual_mse"],
              "near tau bound:", fit["tau_at_bound"])
```

```matlab
exp_cfg = cfg;
exp_cfg.rise_model = 'singlerise';
exp_cfg.decay_model = 'singledecay';
[corrected_exp, trend_exp, diag_exp] = tmspath_detrend(data,times,exp_cfg);
for k = 1:numel(diag_exp.fits)
    rec = diag_exp.fits{k};
    for name = {'rise','decay'}
        fit = rec.(name{1});
        fprintf('Epoch %d, channel %d, %s\n',rec.epoch,rec.channel,name{1});
        disp(1000 * fit.tau_s); % ms
        disp(fit.coefficients);
        disp(fit.residual_mse);
        disp(fit.tau_at_bound);
    end
end
```

## Fit residual versus recovery MSE

The optimization minimizes the segment residual against the **observed input**. Since `corrected = input - trend`, `residual_mse` is the mean squared corrected amplitude within that segment, not the error against the unknown neural signal.

Recovery MSE requires a known clean reference, available in the toy simulation but normally unavailable in real recordings. It is calculated separately by the demonstrations over their declared evaluation interval. Neither recovery MSE nor error relative to the true artifact taus is a core output. Inspect the fitted curve, corrected waveform, split location, and bounds alongside the numerical diagnostics; a smaller residual alone can reflect removal of neural activity.

## Save the results

```python
import pickle
import numpy as np
np.savez_compressed("detrending_arrays.npz", corrected=corrected, trend=trend, times=times)
with open("detrending_diagnostics.pkl", "wb") as handle:
    pickle.dump(diagnostics, handle)  # Reload only trusted local pickle files.
```

```matlab
save('detrending_results.mat','corrected','trend','diagnostics','times','cfg');
```
