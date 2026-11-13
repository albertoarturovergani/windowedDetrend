"""Reproducible synthetic epoched EEG for the MATLAB/Python demonstrations.

This is illustrative data, not a physiological head model. The small portable
random generator is demo-only and gives both languages the same input arrays.
"""
import numpy as np


def make_demo_eeg(*, n_epochs=12, n_channels=16, fs_hz=2000, seed=42,
                  stimulus_jitter_s=0.0007, channel_jitter_s=0.00035,
                  amplitude_jitter=0.10, noise_sd=0.05):
    if n_epochs < 2 or n_channels < 8:
        raise ValueError('Use at least two epochs and eight channels for this demo')
    if not 0 < seed < 2147483647 or int(seed) != seed:
        raise ValueError('Demo seed must be an integer in 1..2147483646')
    if min(stimulus_jitter_s,channel_jitter_s,amplitude_jitter,noise_sd) < 0:
        raise ValueError('Jitter and noise parameters must be nonnegative')
    if stimulus_jitter_s+channel_jitter_s >= .003 or amplitude_jitter >= 1:
        raise ValueError('Keep artifact onset after zero and amplitude jitter below one')
    state=int(seed)
    def uniform(n):
        nonlocal state
        values=np.empty(n)
        for i in range(n):
            state=(16807*state)%2147483647
            values[i]=state/2147483647
        return values

    times=np.arange(round(-.1*fs_hz),round(.25*fs_hz))/fs_hz
    nt=len(times)
    channel_latency=channel_jitter_s*(2*uniform(n_channels)-1)
    stimulus_latency=stimulus_jitter_s*(2*uniform(n_epochs)-1)
    channel_gain=4+2*uniform(n_channels)
    channel_gain[:2]=[80,65]  # Two strong artifact channels; weaker contamination elsewhere.
    phase=2*np.pi*uniform(n_channels)
    neural_gain=.9+.2*uniform(n_channels)
    data=np.empty((n_epochs,n_channels,nt));truth=np.empty_like(data)
    artifact=np.empty_like(data);onsets=np.empty((n_epochs,n_channels))
    peaks=np.empty_like(onsets);gains=np.empty_like(onsets)
    for e in range(n_epochs):
        for c in range(n_channels):
            random=uniform(2)
            onset=.003+stimulus_latency[e]+channel_latency[c]
            peak=onset+.006
            gain=channel_gain[c]*(1+amplitude_jitter*(2*random[0]-1))
            u=np.maximum(times-onset,0)
            pulse=-gain*(-np.expm1(-u/.0025))
            tail=times>=peak
            pulse[tail]=-gain*(-np.expm1(-.006/.0025))*np.exp(-(times[tail]-peak)/.12)
            neural_u=np.maximum(times-(.003+stimulus_latency[e]+.5*channel_latency[c]),0)
            response=3*neural_gain[c]*(1+.1*(2*random[1]-1))*np.exp(-25*neural_u)*np.sin(2*np.pi*40*neural_u)
            background=.3*np.sin(2*np.pi*11*times+phase[c]+2*np.pi*e/n_epochs)
            # Box-Muller noise; draw order is mirrored in tmspath_demo_data.m.
            pairs=(nt+1)//2;u1=uniform(pairs);u2=uniform(pairs)
            radius=np.sqrt(-2*np.log(u1));angle=2*np.pi*u2
            noise=np.column_stack((radius*np.cos(angle),radius*np.sin(angle))).ravel()[:nt]
            truth[e,c]=response+background+noise_sd*noise
            artifact[e,c]=pulse;data[e,c]=truth[e,c]+pulse
            onsets[e,c]=onset;peaks[e,c]=peak;gains[e,c]=gain
    metadata=dict(channel_labels=[f'EEG{c+1:02d}' for c in range(n_channels)],
                  stimulus_jitter_s=stimulus_latency,channel_jitter_s=channel_latency,
                  artifact_onsets_s=onsets,artifact_peaks_s=peaks,artifact_gains=gains,
                  seed=seed,fs_hz=fs_hz)
    return data,times,truth,artifact,metadata
