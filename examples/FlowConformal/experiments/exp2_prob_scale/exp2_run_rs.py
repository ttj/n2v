"""Exp 2 — Cohen randomized-smoothing (RS) runner.

Mirrors the per-(experiment, tool) runner pattern used elsewhere
(``--benchmark X``, ``--smoke``, ``--instance-idx <N>``,
``--list-instances``). Internally calls
``baselines.run_rs._process_factory`` for the actual per-instance
``Smooth.certify`` work — that's the canonical RS implementation
(scipy/statsmodels shims + classification-spec parsing).

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_rs \\
        --benchmark cifar10_resnet110 --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments._runner_utils import (
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
    EXP2_RS_APPLICABLE,
    list_instances,
)

_SEED = 47
_DEFAULT_N_INSTANCES = 100
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'instance', 'verdict', 'wall_s', 'timeout_s',
    'sigma', 'n0', 'n_certify', 'alpha',
    'pred_class', 'true_class', 'l2_radius', 'eps_linf_threshold_l2',
    'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _make_process_one(args):
    """Construct the per-instance ``process_one`` closure from
    :mod:`baselines.run_rs._process_factory`. The factory needs an
    args-like object with ``sigma``/``n0``/``n_certify``/``alpha``/
    ``batch_size``/``eps`` fields; we mirror that into a Namespace.
    """
    from examples.FlowConformal.experiments.baselines.run_rs import (
        _process_factory,
    )
    rs_args = argparse.Namespace(
        sigma=args.sigma, n0=args.n0, n_certify=args.n_certify,
        alpha=args.alpha, batch_size=args.batch_size, eps=args.eps,
    )
    return _process_factory(rs_args)


def _run_one_idx(idx: int, instances, process_one, benchmark: str):
    """Run a single instance and return the CSV row dict."""
    name, loader, timeout_s = instances[idx]
    t0 = time.time()
    try:
        result = process_one(loader, name)
    except Exception as e:
        result = {'verdict': 'ERROR',
                  'error': f'process {type(e).__name__}: {e}'}
    wall_s = time.time() - t0

    row = {f: '' for f in _FIELDS}
    row.update({
        'benchmark': benchmark,
        'instance': name,
        'verdict': result.get('verdict', 'ERROR'),
        'wall_s': f'{wall_s:.2f}',
        'timeout_s': timeout_s,
        'sigma': result.get('sigma', ''),
        'n0': result.get('n0', ''),
        'n_certify': result.get('n_certify', ''),
        'alpha': result.get('alpha', ''),
        'pred_class': result.get('pred_class', ''),
        'true_class': result.get('true_class', ''),
        'l2_radius': result.get('l2_radius', ''),
        'eps_linf_threshold_l2': result.get('eps_linf_threshold_l2', ''),
        'error': result.get('error', ''),
        'timestamp': _now_iso(),
    })
    return row


def _write_timeout_row(out_csv, benchmark, name, timeout_s):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'instance': name,
        'verdict': 'TIMEOUT', 'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP2_RS_APPLICABLE,
                   help=f'RS applies to image classification only: '
                        f'{EXP2_RS_APPLICABLE}.')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-instances', type=int, default=_DEFAULT_N_INSTANCES)
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only this 0-based index; APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <timeout_s>" per instance, exit.')
    p.add_argument('--sigma', type=float, default=0.25,
                   help='RS noise level σ (default 0.25 — Cohen et al. '
                        'default for CIFAR ResNet-110).')
    p.add_argument('--alpha', type=float, default=0.001,
                   help='RS failure probability (default 1e-3).')
    p.add_argument('--n0', type=int, default=100,
                   help='RS selection samples (default 100).')
    p.add_argument('--n-certify', type=int, default=10_000,
                   help='RS certification samples (default 10000).')
    p.add_argument('--batch-size', type=int, default=400,
                   help='RS evaluation batch size (default 400).')
    p.add_argument('--eps', type=float, default=8.0 / 255.0,
                   help='L∞ verification budget (default 8/255).')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    instances = list_instances(benchmark, n=args.n_instances)

    if args.list_instances:
        for idx, (_name, _loader, t) in enumerate(instances):
            print(f'{idx} {t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp2_{benchmark}_rs.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx', file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        name, _loader, t = instances[args.instance_idx]
        _write_timeout_row(out_csv, benchmark, name, t)
        return

    process_one = _make_process_one(args)

    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print(f'--instance-idx {args.instance_idx} out of range '
                  f'[0, {len(instances)})', file=sys.stderr)
            sys.exit(2)
        target_indices = [args.instance_idx]
        append_mode = True
        print(f'[{benchmark}] running only idx={args.instance_idx}; '
              f'appending to {out_csv}', flush=True)
    else:
        if args.smoke:
            target_indices = [0]
            print(f'[smoke] {benchmark}: running only the first instance',
                  flush=True)
        else:
            target_indices = list(range(len(instances)))
        append_mode = False

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'NOT_APPLICABLE': 0, 'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader()
            f.flush()

        for k, idx in enumerate(target_indices, start=1):
            elapsed = time.time() - t_start
            name, _loader, t = instances[idx]
            print(f'[{benchmark} {k}/{len(target_indices)} t={elapsed:.0f}s '
                  f'budget={t}s] idx={idx} {name}', flush=True)
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            row = _run_one_idx(idx, instances, process_one, benchmark)
            writer.writerow(row)
            f.flush()
            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  wall={row["wall_s"]}s',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        if counts.get('ERROR', 0) > 0:
            print(f'[smoke] FAIL on {benchmark}: ERROR observed, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next((v for v, c in counts.items() if c > 0), 'NONE')
        print(f'[smoke] PASS on {benchmark}: RS ran (verdict={actual}).')


if __name__ == '__main__':
    main()
