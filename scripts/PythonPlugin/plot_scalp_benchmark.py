"""Plot the 64-channel geometry, spatial gains and extreme default recovery."""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from windowed_detrend import Config, detrend
from windowed_detrend.scalp_benchmark import scalp_layout, scalp_signal


def main():
    out = Path(__file__).resolve().parent.parent/'results'/'scalp64'
    out.mkdir(parents=True, exist_ok=True)
    labels, xyz, neural_gain, artifact_gain = scalp_layout()
    t, y, truth, _ = scalp_signal(('windowed', .003, .2, 0., 0, 'extreme'), 100)
    cleaned, _, _ = detrend(y[None], t, Config(fill_mode='none', peak_s=.015))
    # Azimuthal projection keeps the below-equator electrodes distinct.
    unit = xyz/.095
    angle = np.arccos(np.clip(unit[:, 2], -1, 1))
    azimuth = np.arctan2(unit[:, 1], unit[:, 0])
    xy = np.column_stack((angle*np.cos(azimuth), angle*np.sin(azimuth)))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    for ax, gain, title, cmap in zip(axes[0], [neural_gain, artifact_gain],
            ['Neural amplitude by channel', 'Artifact amplitude by channel (maximum at C3)'],
            ['coolwarm', 'YlOrRd']):
        points = ax.scatter(xy[:, 0], xy[:, 1], c=gain, cmap=cmap, s=260,
                            edgecolors='#555555', linewidths=.4)
        for ch, label in enumerate(labels):
            ax.text(*xy[ch], label, ha='center', va='center', fontsize=6.5)
        phi = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.pi/2*np.cos(phi), np.pi/2*np.sin(phi), color='.5', lw=.7)
        ax.text(0, 2.18, 'Anterior', ha='center', color='.35')
        ax.set(aspect='equal', title=title, xlim=(-2.2, 2.2), ylim=(-2.3, 2.4))
        ax.axis('off')
        fig.colorbar(points, ax=ax, label='Gain (dimensionless)', shrink=.8)
    axes[1, 0].plot(t*1000, y.T, lw=.6, alpha=.45)
    axes[1, 0].set(title='All 64 contaminated channels: noiseless 3/200 ms case',
                  xlabel='Time (ms)', ylabel='Amplitude (a.u.)', xlim=(0, 250))
    ch = labels.index('C3')
    axes[1, 1].plot(t*1000, truth[ch], color='black', lw=1.6, label='Known clean C3 signal')
    axes[1, 1].plot(t*1000, cleaned[0, ch], color='#1866a8', lw=1.4,
                    label='Lagrange rise / cubic decay')
    axes[1, 1].axvspan(10, 15, color='#d66b19', alpha=.15, label='Rise interval: residual is zero')
    axes[1, 1].set(title='Default recovery example at C3', xlabel='Time (ms)',
                  ylabel='Amplitude (a.u.)', xlim=(0, 250))
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('64-channel synthetic scalp benchmark\nBioSemi 64 layout; illustrative gain fields, no biophysical head model', fontsize=15)
    fig.savefig(out/'scalp64_overview.png', dpi=170)
    fig.savefig(out/'scalp64_overview.pdf')
    plt.close(fig)
    print(out/'scalp64_overview.png')


if __name__ == '__main__':
    main()
