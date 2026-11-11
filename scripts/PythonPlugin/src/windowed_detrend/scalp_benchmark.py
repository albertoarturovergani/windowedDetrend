"""Shared Python/MATLAB benchmark with 64 distinct synthetic scalp channels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .benchmark import configs, metrics, synthetic
from .core import Config, detrend


def scalp_layout():
    """BioSemi 64 spherical layout; coordinates are idealized, not digitized."""
    source = Path(__file__).with_name('data') / 'biosemi64.txt'
    table = np.genfromtxt(source, skip_header=1, dtype=None, encoding='utf-8')
    table = [row for row in table if row[0] not in ('Nz', 'LPA', 'RPA')]
    labels = [row[0] for row in table]
    theta = np.deg2rad([row[1] for row in table])
    phi = np.deg2rad([row[2] for row in table])
    xyz = np.column_stack((np.sin(theta)*np.cos(phi),
                           np.sin(theta)*np.sin(phi), np.cos(theta)))
    # Smooth illustrative gain fields, not a volume-conductor/forward solution.
    neural_gain = 0.4 + 0.8*xyz[:, 0] + 0.4*xyz[:, 1]
    center = xyz[labels.index('C3')]
    artifact_gain = 0.25 + 1.25*np.exp(-np.sum((xyz-center)**2, axis=1)/0.5)
    return labels, xyz*0.095, neural_gain, artifact_gain


def case_definitions(size=10):
    if size < 2:
        raise ValueError('Grid size must be at least 2')
    cases = []
    for family in ('windowed', 'biexp'):
        cases.extend((family, float(r), float(d), 0., 0, 'grid')
                     for r in np.linspace(.001, .050, size)
                     for d in np.linspace(.025, .250, size))
        cases.extend((family, .003, .200, noise, rep, 'extreme')
                     for noise in (0., .5, 2.)
                     for rep in range(1 if noise == 0 else 3))
    return cases


def scalp_signal(case, case_id, seed=42):
    family, rise, decay, noise_sd, _, _ = case
    times = np.arange(2500)/10000
    _, _, neural_gain, artifact_gain = scalp_layout()
    oscillator, artifact = synthetic(times, rise, decay, family)
    truth = neural_gain[:, None]*oscillator
    artifact = artifact_gain[:, None]*artifact
    # A separate reproducible stream for every case/channel, independent of workers.
    noise = np.stack([np.random.default_rng(np.random.SeedSequence([seed, case_id, ch]))
                      .normal(0, noise_sd, times.size) for ch in range(64)])
    return times, truth+artifact+noise, truth, artifact


def scalp_configs(peak):
    return {'default_lagrange3': Config(peak_s=peak), **configs(peak)}


def _run_case(job):
    from scipy.io import savemat
    case_id, case, seed, output = job
    family, rise, decay, noise_sd, replicate, group = case
    times, observed, truth, artifact = scalp_signal(case, case_id, seed)
    peak = .015 if family == 'windowed' else .010+rise*np.log1p(decay/rise)
    method_cfgs = scalp_configs(peak)
    recovered, rows = [], []
    mask = times >= .010
    early = (times >= .010) & (times < .030)
    late = times >= .030
    for method, cfg in method_cfgs.items():
        clean, _, _ = detrend(observed[None, :, :], times, cfg)
        recovered.append(clean[0])
        for channel in range(64):
            rows.append(dict(case_id=case_id, channel_id=channel, family=family,
                             group=group, tau_rise_ms=rise*1000, tau_decay_ms=decay*1000,
                             noise_sd=noise_sd, replicate=replicate, method=method,
                             **metrics(truth[channel], observed[channel], clean[0, channel], mask),
                             mse_early=float(np.mean((clean[0, channel, early]-truth[channel, early])**2)),
                             mse_late=float(np.mean((clean[0, channel, late]-truth[channel, late])**2))))
    path = Path(output)/'fixtures'/f'case_{case_id:03d}.mat'
    savemat(path, dict(data=observed[None, :, :], python_recovered=np.array(recovered),
                      parameters=np.array([rise, decay, noise_sd, replicate, int(family == 'biexp')]),
                      peak_s=peak, case_id=case_id), do_compression=True)
    return rows, dict(case_id=case_id, family=family, group=group,
                      tau_rise_ms=rise*1000, tau_decay_ms=decay*1000,
                      noise_sd=noise_sd, replicate=replicate, peak_s=float(peak),
                      file=f'fixtures/{path.name}', sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(output, size=10, seed=42, workers=1):
    from scipy.io import savemat
    if workers < 1 or seed < 0:
        raise ValueError('Workers must be positive; seed must be nonnegative')
    cases = case_definitions(size)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'fixtures').mkdir(exist_ok=True)
    # Avoid mixing a new partial run with an old successful report.
    summary_path = out/'summary.json'
    if summary_path.exists():
        summary_path.unlink()
    labels, xyz, neural_gain, artifact_gain = scalp_layout()
    times, _, truth, _ = scalp_signal(cases[0], 0, seed)
    savemat(out/'metadata.mat', dict(times=times, truth=truth,
            channel_names=np.array(labels, dtype=object), channel_positions_m=xyz,
            neural_gain=neural_gain, artifact_gain=artifact_gain,
            method_names=np.array(list(scalp_configs(.015)), dtype=object)), do_compression=True)
    with (out/'channels.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['channel_id', 'label', 'x_m', 'y_m', 'z_m', 'neural_gain', 'artifact_gain'])
        writer.writerows((ch, label, *xyz[ch], neural_gain[ch], artifact_gain[ch])
                         for ch, label in enumerate(labels))
    (out/'method_configs.json').write_text(json.dumps(
        {name: asdict(cfg) for name, cfg in scalp_configs(.015).items()}, indent=2), encoding='utf-8')
    jobs = [(idx, case, seed, str(out)) for idx, case in enumerate(cases)]
    start = time.perf_counter()
    all_rows, manifest = [], []
    def collect(results):
        for idx, (rows, entry) in enumerate(results):
            all_rows.extend(rows)
            manifest.append(entry)
            print(f'{idx+1}/{len(cases)} cases x 64 channels x 5 methods', flush=True)
    if workers == 1:
        collect(map(_run_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_run_case, jobs))
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = dict(seed=seed, grid_size=size, cases=len(cases), channels=64, samples=2500,
                   dataset_shape=[len(cases), 64, 2500], sampling_hz=10000,
                   epoch_s=[0., float(times[-1])], evaluation_window_s=[.010, float(times[-1])],
                   traces=len(cases)*64, reconstructed_traces=len(all_rows),
                   methods=list(scalp_configs(.015)), amplitude_unit='a.u.',
                   montage='BioSemi 64, idealized sphere with radius 0.095 m; not digitized',
                   spatial_model='Fixed smooth neural/artifact gain fields; no head forward model',
                   peak_mode='Known artifact peak for each case, shared across channels',
                   noise='Independent Gaussian case/channel streams; SeedSequence([seed, case_id, channel_id])',
                   fixture_shape=[1, 64, 2500], python_recovered_shape=[5, 64, 2500],
                   truth_location='metadata.mat: truth[64,2500], common to all cases',
                   seconds=time.perf_counter()-start, numpy_version=np.__version__)
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='../results/scalp64')
    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.grid_size < 2 or args.workers < 1 or args.seed < 0:
        parser.error('grid-size >= 2, workers >= 1 and seed >= 0 required')
    run(args.output, args.grid_size, args.seed, args.workers)


if __name__ == '__main__':
    main()
