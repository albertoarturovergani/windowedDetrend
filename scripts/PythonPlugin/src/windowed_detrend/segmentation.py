"""Window conventions and channel screening from the published TEP workflow."""
import numpy as np


def time_masks(times, signal, cfg, peak=None):
    """Return pre/rise/decay masks and the actual split time.

    Pipeline masks include onset in pre and rise, and peak in rise only.
    A supplied off-grid peak selects the last sample at or before that time.
    The explicit figure convention preserves historical benchmark masks.
    """
    t = np.asarray(times)
    end = t[-1] if cfg.end_s is None else cfg.end_s
    requested = cfg.peak_s if peak is None else peak
    if requested is None:
        search = np.flatnonzero((t >= cfg.onset_s) & (t <= min(cfg.peak_search_end_s, end)))
        if not search.size:
            raise ValueError('Empty peak search interval')
        requested = t[search[np.argmax(np.abs(signal[search]))]]
    if not cfg.onset_s <= requested < end:
        raise ValueError('Peak must lie after onset and before correction end')
    if cfg.mask_convention == 'pipeline':
        ids = np.flatnonzero((t >= cfg.onset_s) & (t <= requested))
        if not ids.size:
            raise ValueError('Rise window contains no samples')
        split = float(t[ids[-1]])
        pre = t <= cfg.onset_s
        rise = (t >= cfg.onset_s) & (t <= split)
        decay = (t > split) & (t <= end)
    else:
        split = float(requested)
        pre = t < cfg.onset_s
        rise = (t >= cfg.onset_s) & (t < split)
        decay = (t >= split) & (t <= end)
    return pre, rise, decay, split


def select_offset_channels(data, times, channels, cfg):
    """Match computeSlopes_v4 + check_detrend_need on the selected channels.

    Z-scores use population SD over every epoch/channel within each window.
    The mean z-score across epochs selects a channel if any window reaches
    abs(mean_z) >= slope_threshold. Electrode distances in the source are
    reporting-only and do not participate in this decision.
    """
    result = dict(policy=cfg.channel_policy, channels=list(channels),
                  threshold=cfg.slope_threshold, window_labels=['pre', 'rise', 'decay'])
    if cfg.channel_policy == 'all':
        offsets = list(channels)
    elif cfg.channel_policy == 'manual':
        offsets = [c for c in channels if c in cfg.offset_channels]
    else:
        slopes = np.empty((data.shape[0], len(channels), 3))
        peaks = np.empty((data.shape[0], len(channels)))
        for j, c in enumerate(channels):
            for e in range(data.shape[0]):
                *masks, peaks[e, j] = time_masks(times, data[e, c], cfg)
                for k, mask in enumerate(masks):
                    x = times[mask]
                    y = data[e, c, mask]
                    if x.size < 2:
                        raise ValueError('Automatic channel screening needs two samples in every window')
                    x = x - x.mean()
                    slopes[e, j, k] = np.dot(x, y-y.mean()) / np.dot(x, x)
        zscores = np.zeros_like(slopes)
        for k in range(3):
            # Source traversal is channel first, then epoch.
            values = slopes[:, :, k].T.ravel()
            sd = np.std(values, ddof=0)
            if sd > 0:
                zscores[:, :, k] = (slopes[:, :, k] - np.mean(values)) / sd
        mean_z = zscores.mean(axis=0)
        offsets = [c for j, c in enumerate(channels) if np.any(np.abs(mean_z[j]) >= cfg.slope_threshold)]
        result.update(slopes=slopes.tolist(), zscores=zscores.tolist(),
                      mean_z=mean_z.tolist(), peaks_s=peaks.tolist())
    result['offset_channels'] = offsets
    result['found_offsets'] = bool(offsets)
    return result
