"""Offset filling following apply_offset_correction.

Gaussian noise parameters come from preceding ORIGINAL samples. Fourier
resampling is an alternative. Neither changes the epoch's time axis or
estimates the missing neural response.
"""
import numpy as np


def _fourier_resample(values, count):
    """Real 1-D Fourier resampling, including even-length Nyquist handling.

Equivalent to scipy.signal.resample(values, count) with default window=None;
implemented with NumPy so the optional fill does not add a core dependency.
"""
    values = np.asarray(values, dtype=float)
    source_count = values.size
    spectrum = np.fft.rfft(values)
    output = np.zeros(count // 2 + 1, dtype=complex)
    shared = min(source_count, count)
    stop = shared // 2 + 1
    output[:stop] = spectrum[:stop]
    if shared % 2 == 0:
        if count < source_count:
            output[shared // 2] *= 2
        elif source_count < count:
            output[shared // 2] *= .5
    return np.fft.irfft(output, n=count) * (count / source_count)


def fill_offset(signal, times, onset_s, peak_s, margin_s, mode='resample', seed=42, epoch=0, channel=0):
    """Return replacement values and inclusive zero-based source/target bounds.

    The rise contains onset <= t < peak. Its first and last sampled times
    define the target, expanded by margin_s and clipped to the epoch. The
    source is the original pre-onset interval, trimmed by the same margin;
    if trimming removes all source samples, the full pre-onset interval is
    used, as in the pipeline. Missing source data raises instead of skipping.
    """
    t = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if (t.ndim != 1 or signal.shape != t.shape or len(t) < 2
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(signal))):
        raise ValueError('Fill requires finite one-dimensional signal and times')
    spacing = np.diff(t)
    if np.any(spacing <= 0):
        raise ValueError('Filling requires increasing sample times')
    if mode == 'resample' and not np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-12):
        raise ValueError('Fourier filling requires uniformly increasing sample times')
    if not np.isfinite(margin_s) or margin_s < 0:
        raise ValueError('fill_margin_s must be finite and nonnegative')
    pre = np.flatnonzero(t < onset_s)
    rise = np.flatnonzero((t >= onset_s) & (t < peak_s))
    if not pre.size or not rise.size:
        raise ValueError('Filling requires pre-onset source samples and a nonempty rise window')
    source = np.flatnonzero((t >= t[pre[0]]) & (t <= t[pre[-1]]-margin_s))
    fallback = not source.size
    if fallback:
        source = pre
    target = np.flatnonzero((t >= t[rise[0]]-margin_s) & (t <= t[rise[-1]]+margin_s))
    extra = {}
    if mode == 'gaussian':
        mean = float(np.mean(signal[source]))
        std = max(float(np.std(signal[source], ddof=0)) * .5, 1e-8)
        rng = np.random.default_rng(np.random.SeedSequence([seed, channel, epoch]))
        values = rng.normal(mean, std, target.size)
        extra = dict(mean=mean, std=std, seed=int(seed), rng='NumPy PCG64/SeedSequence(seed,channel,epoch)')
    elif mode == 'resample':
        values = _fourier_resample(signal[source], target.size)
    else:
        raise ValueError('Unsupported fill mode')
    return dict(mode=mode, method='normal' if mode=='gaussian' else 'fourier', margin_s=float(margin_s),
                source_first=int(source[0]), source_last=int(source[-1]),
                source_count=int(source.size), source_fallback=bool(fallback),
                target_first=int(target[0]), target_last=int(target[-1]),
                target_count=int(target.size),
                source_start_s=float(t[source[0]]), source_end_s=float(t[source[-1]]),
                target_start_s=float(t[target[0]]), target_end_s=float(t[target[-1]]),
                values=values.tolist(), **extra)
