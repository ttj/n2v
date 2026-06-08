"""Exp 2 — Hashemi-clipping with PCA (m=8000) runner.

Identical to ``exp2_run_hashemi_clipping`` except it threads a
``pca_components`` argument into ``conformal_reach(...)``. PCA-augmented
clipping is the configuration the published Hashemi-clipping paper
uses on high-output-dim networks (semantic segmentation, multi-class
classifiers); raw clipping_block solves m=8000 LPs per instance with
``2*output_dim`` constraints each, which TIMEOUTs on cifar100 (out=100)
and tinyimagenet (out=200) in our sweep. PCA reduces the LP constraint
count from ``2*output_dim`` to ``2*K`` while retaining most of the
output variance, restoring tractability.

Verdict semantics, falsification, parity with ``exp2_run_ours`` are
unchanged from the no-PCA runner.

Output CSV: ``exp2_<benchmark>_hashemi_clipping_pca.csv`` (sibling of
``exp2_<benchmark>_hashemi_clipping.csv``). The ``pca_components``
column carries K so the aggregator can distinguish the two configs.

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_hashemi_clipping_pca \\
        --benchmark cifar100_2024 --pca-components 32 --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from examples.FlowConformal.experiments._runner_utils import (
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments.baselines._common import (
    halfspace_disjoint_from_box,
    torch_callable,
)
from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
    EXP2_BENCHMARKS,
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
from n2v.nn import NeuralNetwork
from n2v.nn.reach import ConformalReachConfig
from n2v.sets import Box
from n2v.utils.falsify import falsify

_SEED = 47
_DEFAULT_M = 8000
_EPSILON = 0.001
_DEFAULT_N_INSTANCES = 100
_DEFAULT_PCA_COMPONENTS = 32
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'instance', 'verdict', 'wall_s', 'timeout_s',
    'm', 'ell', 'epsilon', 'pca_components', 'training_samples',
    'coverage', 'confidence',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _write_timeout_row(out_csv: Path, benchmark: str, name: str,
                       timeout_s: int, pca_components: int,
                       m: int, training_samples: int) -> None:
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'instance': name,
        'verdict': 'TIMEOUT', 'timeout_s': timeout_s,
        'm': m, 'ell': m - 1,
        'pca_components': pca_components,
        'training_samples': training_samples,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def _run_one_instance(benchmark: str, loader, *, seed: int,
                      pca_components: int, m: int,
                      training_samples: int | None = None) -> Dict[str, Any]:
    try:
        network, boxes, spec, _name = load_one_instance(benchmark, loader)
    except FileNotFoundError as e:
        return {'verdict': 'ERROR', 'error': f'load_missing: {e}'}
    except NotImplementedError as e:
        return {'verdict': 'SKIPPED',
                'error': f'unsupported_spec {type(e).__name__}: {e}'}
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'load {type(e).__name__}: {e}'}

    try:
        network = network.cuda()
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'gpu_move {type(e).__name__}: {e}'}

    cfg = PER_BENCHMARK_CONFIG[benchmark]
    use_falsifier = cfg.get('use_falsifier', False)
    falsifier_method = cfg.get('falsifier_method', 'apgd')
    falsifier_kwargs = {
        'n_restarts': cfg.get('falsifier_n_restarts', 10),
        'n_steps': cfg.get('falsifier_n_steps', 100),
    }

    net = NeuralNetwork(network)
    model_fn = torch_callable(network)  # still needed for empirical_coverage_for_box below
    any_unknown = False
    last_pbox = None
    cex_x_str = ''
    cex_y_str = ''
    sat = False

    for box_idx, (lb, ub) in enumerate(boxes):
        input_set = Box(np.asarray(lb).flatten(),
                        np.asarray(ub).flatten())

        if use_falsifier:
            try:
                fals_result, fals_cex = falsify(
                    model=network,
                    lb=np.asarray(lb), ub=np.asarray(ub),
                    property=spec,
                    method=falsifier_method,
                    seed=seed,
                    **falsifier_kwargs,
                )
                if fals_result == 0 and fals_cex is not None:
                    cex_x_arr, cex_y_arr = fals_cex
                    sat = True
                    cex_x_str = json.dumps(np.asarray(cex_x_arr).flatten().tolist())
                    cex_y_str = json.dumps(np.asarray(cex_y_arr).flatten().tolist())
                    break
            except Exception:
                pass

        try:
            config_kwargs = dict(
                m=m, ell=m - 1, epsilon=_EPSILON,
                surrogate='clipping_block',
                pca_components=pca_components,
                seed=seed, verbose=False,
            )
            if training_samples is not None:
                config_kwargs['training_samples'] = training_samples
            pbox = net.reach(
                input_set, method='conformal',
                config=ConformalReachConfig(**config_kwargs),
            )
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'verify box={box_idx} {type(e).__name__}: {e}'}
        last_pbox = pbox

        disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
        if disjoint is True:
            continue
        any_unknown = True

    if sat:
        verdict = 'SAT'
    elif any_unknown:
        verdict = 'UNKNOWN'
    else:
        verdict = 'UNSAT'

    return {
        'verdict': verdict,
        'm': m, 'ell': m - 1, 'epsilon': _EPSILON,
        'pca_components': pca_components,
        'training_samples': (training_samples
                              if training_samples is not None
                              else m // 2),
        'coverage': (
            f'{last_pbox.coverage:.4f}' if last_pbox is not None else ''),
        'confidence': (
            f'{last_pbox.confidence:.4f}' if last_pbox is not None else ''),
        'cex_x': cex_x_str, 'cex_y': cex_y_str, 'error': '',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP2_BENCHMARKS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-instances', type=int, default=_DEFAULT_N_INSTANCES)
    p.add_argument('--pca-components', type=int,
                   default=_DEFAULT_PCA_COMPONENTS,
                   help=(f'PCA components K for the deflation-PCA stage '
                         f'before the convex-hull projection. Default '
                         f'K={_DEFAULT_PCA_COMPONENTS}; matches the '
                         f'paper-style scaling for high-output-dim '
                         f'classifiers.'))
    p.add_argument('--m', type=int, default=_DEFAULT_M,
                   help=(f'Calibration set size. Default {_DEFAULT_M}. '
                         f'Reduce (e.g. 750) to fit tight per-instance '
                         f'budgets at the cost of a slightly looser '
                         f'conformal coverage interval.'))
    p.add_argument('--training-samples', type=int, default=None,
                   help=(f'Training-set size for the convex-hull surrogate. '
                         f'Default = m // 2. Reducing this lowers the LP '
                         f'variable count.'))
    p.add_argument('--instance-idx', type=int, default=None)
    p.add_argument('--list-instances', action='store_true')
    p.add_argument('--write-timeout-row', action='store_true')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    pca_K = args.pca_components
    m_val = int(args.m)
    train_val = (int(args.training_samples)
                 if args.training_samples is not None else None)

    if args.list_instances:
        rows = list_instances(benchmark, n=args.n_instances)
        for idx, (_name, _loader, t) in enumerate(rows):
            print(f'{idx} {t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp2_{benchmark}_hashemi_clipping_pca.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        all_inst = list_instances(benchmark, n=args.n_instances)
        if not (0 <= args.instance_idx < len(all_inst)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        name, _loader, t = all_inst[args.instance_idx]
        _write_timeout_row(out_csv, benchmark, name, t, pca_K, m_val,
                           train_val if train_val is not None
                           else m_val // 2)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        all_instances = list_instances(benchmark, n=args.n_instances)
        if not (0 <= args.instance_idx < len(all_instances)):
            print(f'--instance-idx {args.instance_idx} out of range '
                  f'[0, {len(all_instances)})', file=sys.stderr)
            sys.exit(2)
        instances = [all_instances[args.instance_idx]]
        append_mode = True
        print(f'[{benchmark}] running only idx={args.instance_idx}; '
              f'appending to {out_csv}', flush=True)
    else:
        n = 1 if args.smoke else args.n_instances
        instances = list_instances(benchmark, n=n)
    if args.smoke:
        instances = instances[:1]
        print(f'[smoke] {benchmark}: running only the first instance',
              flush=True)
    print(f'[{benchmark}] Loaded {len(instances)} instances; '
          f'writing to {out_csv}', flush=True)
    train_print = train_val if train_val is not None else m_val // 2
    print(f'[{benchmark}] Hashemi-clipping+PCA config: m={m_val} '
          f'ell={m_val-1} epsilon={_EPSILON} pca_components={pca_K} '
          f'training_samples={train_print} SEED={_SEED}',
          flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'SKIPPED': 0, 'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader()
            f.flush()

        for k, (name, loader, timeout_s) in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            print(f'[{benchmark} {k}/{len(instances)} t={elapsed:.0f}s '
                  f'budget={timeout_s}s] {name}', flush=True)
            t0 = time.time()
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            try:
                row = _run_one_instance(
                    benchmark, loader, seed=_SEED,
                    pca_components=pca_K, m=m_val,
                    training_samples=train_val)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'instance': name,
                'wall_s': f'{wall_s:.1f}',
                'timeout_s': timeout_s,
                'timestamp': _now_iso(),
            })
            out_row.update(row)
            writer.writerow(out_row)
            f.flush()
            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  wall={wall_s:.1f}s',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        bad = {'ERROR', 'TIMEOUT', 'SKIPPED'}
        if any(counts.get(v, 0) > 0 for v in bad):
            print(f'[smoke] FAIL on {benchmark}: counts={counts}',
                  file=sys.stderr)
            sys.exit(1)
        actual = next(v for v, c in counts.items() if c > 0)
        from examples.FlowConformal.experiments._ground_truth_lookup import (
            lookup_ground_truth,
        )
        first_inst = instances[0][0]
        gt = lookup_ground_truth('exp2', benchmark, first_inst)
        print(f'[smoke] PASS on {benchmark}: pipeline ran end-to-end '
              f'(verdict={actual}, VNN-COMP ground truth {gt}).')


if __name__ == '__main__':
    main()
