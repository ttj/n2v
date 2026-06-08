"""Exp 2 — Hashemi-clipping (m=8000) runner.

Mirrors Exp 1's Hashemi-clipping runner but consumes Exp 2's deferred
loaders (which give a Python-side network for image classification
benchmarks). Iterates the first N (default 100) instances per
benchmark and writes one CSV row per instance.

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_hashemi_clipping \\
        --benchmark cifar10_resnet110 --smoke
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
_M = 8000
_EPSILON = 0.001
_ELL = _M - 1
_DEFAULT_N_INSTANCES = 100
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'instance', 'verdict', 'wall_s', 'timeout_s',
    'm', 'ell', 'epsilon', 'coverage', 'confidence',
    'cex_x', 'cex_y', 'error', 'timestamp',
]



def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _write_timeout_row(out_csv: Path, benchmark: str, name: str,
                       timeout_s: int) -> None:
    """Append a TIMEOUT row when killed by outer shell timeout."""
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'instance': name,
        'verdict': 'TIMEOUT', 'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def _run_one_instance(benchmark: str, loader, *, seed: int) -> Dict[str, Any]:
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

    # Move network to GPU (mirrors exp2_run_ours). Fall back to ERROR
    # rather than silently using CPU on networks that don't fit on
    # GPU (or have CPU-only ops).
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

        # Stage-1 falsifier (parity with ours): runs BEFORE the
        # Hashemi calibration ``conformal_reach(...)`` so a CEX short-circuits
        # to SAT without paying calibration cost — matching the
        # falsify-first ordering in ``run_verification_pipeline``.
        # Uses the same APGD method and per-benchmark
        # ``(n_restarts, n_steps)`` budget as ``exp2_run_ours``.
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
            pbox = net.reach(
                input_set, method='conformal',
                config=ConformalReachConfig(
                    m=_M, ell=_ELL, epsilon=_EPSILON,
                    surrogate='clipping_block',
                    seed=seed, verbose=False,
                ),
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
        'm': _M, 'ell': _ELL, 'epsilon': _EPSILON,
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
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only the instance at this 0-based index '
                        'and APPEND to CSV (used by run_cell.sh).')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <timeout_s>" per instance, exit.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    cfg = PER_BENCHMARK_CONFIG[benchmark]

    if args.list_instances:
        rows = list_instances(benchmark, n=args.n_instances)
        for idx, (_name, _loader, t) in enumerate(rows):
            print(f'{idx} {t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp2_{benchmark}_hashemi_clipping.csv')
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
        _write_timeout_row(out_csv, benchmark, name, t)
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
    print(f'[{benchmark}] Hashemi-clipping config: m={_M} ell={_ELL} '
          f'epsilon={_EPSILON} SEED={_SEED}', flush=True)

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
                row = _run_one_instance(benchmark, loader, seed=_SEED)
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
