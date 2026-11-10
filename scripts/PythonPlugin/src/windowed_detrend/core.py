"""Standalone TMSpath detrending adaptation. AGPL-3.0; see README.
Input: epochs x channels x samples; times in seconds; amplitudes unchanged.
No epoching, filtering, interpolation, ICA or random replacement is performed.
"""
from dataclasses import dataclass, asdict
from itertools import product
import numpy as np

@dataclass
class Config:
    mode: str = 'windowed'
    onset_s: float = 0.010
    peak_s: float | None = None
    peak_search_end_s: float = 0.100
    end_s: float | None = None
    rise_model: str = 'lagrange'
    decay_model: str = 'poly'
    global_model: str = 'biexp'
    rise_degree: int = 3
    decay_degree: int = 3
    pre_degree: int | None = None
    constrain_rise: bool = False
    constrain_decay: bool = True
    fit_offset: bool = False
    tau_min_s: float = 0.0005
    tau_max_s: float = 1.0
    grid_size: int = 21
    refinements: int = 5
    channels: tuple | None = None


def _basis(x, model, theta, degree, scale):
    if model == 'poly':
        return np.column_stack([(x / scale)**i for i in range(degree + 1)])
    if model == 'singlerise': return (-np.expm1(-x/theta[0]))[:, None]
    if model == 'singledecay': return np.exp(-x/theta[0])[:, None]
    if model == 'biexp': return (-np.expm1(-x/theta[0])*np.exp(-x/theta[1]))[:, None]
    if model == 'doubledecay': return np.column_stack([np.exp(-x/t) for t in theta])
    if model == 'alpha': return (x/theta[0]*np.exp(-x/theta[0]))[:, None]
    raise ValueError(f'Unsupported model: {model}')


def fit_segment(x, y, model, degree, cfg, anchor=None):
    """Variable projection: grid/refined log-tau search, linear LS amplitudes.
    anchor is trend value at x=0; it is not a neural ground truth sample.
    """
    if model == 'lagrange':
        if len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError('Lagrange interpolation requires at least two distinct increasing nodes')
        if anchor is not None:
            raise ValueError('Lagrange interpolation uses observed nodes; disable the start constraint')
        # P(x_i)=y_i exactly. We only evaluate at the original nodes, so no
        # ill-conditioned Vandermonde solve or high-degree coefficients are needed.
        return y.copy(), dict(model=model, degree=len(x)-1, node_times_s=x.tolist(),
                              node_values=y.tolist(), tau_s=[], coefficients=[],
                              residual_mse=0.0, anchor=None,
                              scale_s=max(float(np.max(x)), 1e-12), tau_at_bound=False)
    if len(x) < max(4, degree+2 if model == 'poly' else 4):
        raise ValueError('Too few samples for requested model/degree')
    ntheta = {'poly':0, 'singlerise':1, 'singledecay':1,
              'doubledecay':2, 'biexp':2, 'alpha':1}[model]
    scale = max(float(np.max(x)), 1e-12)
    def solve(theta):
        B = _basis(x, model, theta, degree, scale)
        b0 = _basis(np.zeros(1), model, theta, degree, scale)[0]
        if model != 'poly' and cfg.fit_offset:
            B = np.column_stack([B, np.ones(len(x))]); b0 = np.r_[b0, 1.]
        fixed = np.zeros(B.shape[1])
        if anchor is not None:
            j = int(np.argmax(np.abs(b0)))
            if abs(b0[j]) > 1e-12:
                keep = np.arange(B.shape[1]) != j
                fixed[j] = anchor/b0[j]
                X = B[:, keep] - B[:, [j]] * (b0[keep]/b0[j])
                rhs = y - B@fixed
            else:
                if abs(anchor) > 1e-12: raise ValueError('Incompatible start constraint')
                keep = np.ones(B.shape[1], dtype=bool); X=B; rhs=y
        else:
            keep = np.ones(B.shape[1], dtype=bool); X=B; rhs=y
        if X.shape[1]:
            if np.linalg.matrix_rank(X) < X.shape[1]: return None
            c = np.linalg.lstsq(X, rhs, rcond=None)[0]
            fixed[keep] = c
            if anchor is not None and abs(b0[j]) > 1e-12:
                fixed[j] = (anchor - b0[keep]@c)/b0[j]
        fit = B@fixed
        return float(np.mean((y-fit)**2)), fit, fixed
    best = None
    bounds = np.log([cfg.tau_min_s, cfg.tau_max_s])
    lo = np.repeat(bounds[0], ntheta); hi = np.repeat(bounds[1], ntheta)
    for level in range(1 if ntheta == 0 else cfg.refinements+1):
        axes = [np.linspace(lo[k], hi[k], cfg.grid_size if level == 0 else 7) for k in range(ntheta)]
        for logs in product(*axes):
            theta = np.exp(logs)
            if model == 'doubledecay' and theta[0] >= theta[1]: continue
            result = solve(theta)
            if result is not None and (best is None or result[0] < best[0]):
                best = (*result, theta)
        if best is None: raise ValueError('No full-rank fit for this segment')
        if ntheta:
            step = (hi-lo)/((cfg.grid_size if level == 0 else 7)-1)
            center = np.log(best[3]); lo = np.maximum(bounds[0], center-step); hi=np.minimum(bounds[1],center+step)
    mse, fit, coef, theta = best
    return fit, dict(model=model, tau_s=theta.tolist(), coefficients=coef.tolist(),
                     residual_mse=mse, anchor=anchor, scale_s=scale,
                     tau_at_bound=bool(np.any(np.isclose(theta,cfg.tau_min_s)) or np.any(np.isclose(theta,cfg.tau_max_s))))


def detrend(data, times, config=None):
    """Return corrected array, estimated trend, and per-trace diagnostics.
    Raises on invalid/nonfinite input or undersampled windows; never silently
    substitutes a different model. Unselected channels/samples remain intact.
    """
    cfg = config or Config()
    y=np.asarray(data, dtype=float); t=np.asarray(times, dtype=float)
    if y.ndim != 3 or t.ndim != 1 or y.shape[-1] != t.size or not y.size:
        raise ValueError('Expected data[epochs, channels, samples] and times[samples]')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError('Finite data and strictly increasing times required')
    end=t[-1] if cfg.end_s is None else cfg.end_s
    if not t[0] <= cfg.onset_s < end <= t[-1]+1e-12: raise ValueError('Invalid correction interval')
    if cfg.mode not in ('windowed','global'): raise ValueError('Invalid mode')
    if not 0<cfg.tau_min_s<cfg.tau_max_s or cfg.grid_size<3 or cfg.refinements<0:
        raise ValueError('Invalid tau search configuration')
    if cfg.grid_size != int(cfg.grid_size) or cfg.refinements != int(cfg.refinements): raise ValueError('Grid counts must be integers')
    for degree in (cfg.rise_degree,cfg.decay_degree,cfg.pre_degree):
        if degree is not None and (degree<0 or degree!=int(degree)): raise ValueError('Invalid degree')
    channels=list(range(y.shape[1])) if cfg.channels is None else list(cfg.channels)
    if len(set(channels))!=len(channels) or any(not isinstance(c,(int,np.integer)) or c<0 or c>=y.shape[1] for c in channels):
        raise ValueError('Invalid channel indices')
    trend=np.zeros_like(y); records=[]
    for e in range(y.shape[0]):
        for c in channels:
            sig=y[e,c]; rec=dict(epoch=e,channel=c)
            if cfg.pre_degree is not None:
                pre=t<cfg.onset_s
                trend[e,c,pre],rec['pre']=fit_segment(t[pre]-t[pre][0],sig[pre],'poly',cfg.pre_degree,cfg)
            if cfg.mode=='global':
                mask=(t>=cfg.onset_s)&(t<=end)
                trend[e,c,mask],rec['global']=fit_segment(t[mask]-cfg.onset_s,sig[mask],cfg.global_model,cfg.decay_degree,cfg,0. if cfg.constrain_rise else None)
            else:
                if cfg.peak_s is None:
                    search=np.flatnonzero((t>=cfg.onset_s)&(t<=min(cfg.peak_search_end_s,end)))
                    if not len(search): raise ValueError('Empty peak search interval')
                    peak=t[search[np.argmax(np.abs(sig[search]))]]
                else: peak=cfg.peak_s
                if not cfg.onset_s<peak<end: raise ValueError('Peak must be strictly inside correction interval')
                rise=(t>=cfg.onset_s)&(t<peak); decay=(t>=peak)&(t<=end)
                trend[e,c,rise],rec['rise']=fit_segment(t[rise]-cfg.onset_s,sig[rise],cfg.rise_model,cfg.rise_degree,cfg,0. if cfg.constrain_rise else None)
                anchor=float(np.interp(peak,t,sig)) if cfg.constrain_decay else None
                trend[e,c,decay],rec['decay']=fit_segment(t[decay]-peak,sig[decay],cfg.decay_model,cfg.decay_degree,cfg,anchor)
                rec['peak_s']=float(peak)
            records.append(rec)
    return y-trend,trend,dict(config=asdict(cfg),fits=records)


def detrend_mne(epochs, config=None, picks=None):
    """Copy an MNE Epochs object; select EEG channels by default (exclude bads)."""
    import mne
    from dataclasses import replace
    if picks is None: picks=mne.pick_types(epochs.info,eeg=True,exclude='bads')
    cfg=replace(config or Config(),channels=tuple(int(i) for i in picks))
    cleaned,trend,info=detrend(epochs.get_data(),epochs.times,cfg)
    out=epochs.copy().load_data()
    out._data=cleaned
    return out,trend,info
