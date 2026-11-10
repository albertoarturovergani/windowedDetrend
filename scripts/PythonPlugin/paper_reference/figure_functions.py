import numpy as np
from scipy.interpolate import interp1d

def fit_constrained_poly(t_decay, y_decay, V_peak, degree=5, t_peak=15):
    t_shifted = t_decay - t_peak
    y_shifted = y_decay - V_peak
    X = np.vstack([t_shifted**i for i in range(1, degree + 1)]).T
    coeffs = np.linalg.lstsq(X, y_shifted, rcond=None)[0]
    return V_peak + X @ coeffs

def generate_damped_oscillator(t, A=5.0, f=50.0, lambda_=25, t0=10):
    y = np.zeros_like(t)
    mask = t >= t0
    t_shifted = t[mask] - t0
    y[mask] = A * np.exp(-lambda_ * t_shifted / 1000) * np.sin(2 * np.pi * f * t_shifted / 1000)
    return y

def capacitive_artifact(t, t0, t_peak, tauRise, tauDecay, scale=-100):
    V0 = 1.0
    signal = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t0:
            signal[i] = 0
        elif ti < t_peak:
            signal[i] = V0 * (1 - np.exp(-(ti - t0) / tauRise))
        else:
            V_peak = V0 * (1 - np.exp(-(t_peak - t0) / tauRise))
            signal[i] = V_peak * np.exp(-(ti - t_peak) / tauDecay)
    return scale * signal

def exp_rise(t, A, tau, t0):
    t_adj = np.clip(np.asarray(t) - t0, 0, None)
    return A * (1 - np.exp(-t_adj / tau))

def exp_decay_constrained(t, tau, t_peak, t_full, y_full):
    interp = interp1d(t_full, y_full, kind='linear', bounds_error=False, fill_value="extrapolate")
    V_peak = float(interp(t_peak))
    return V_peak * np.exp(-(t - t_peak) / tau)
