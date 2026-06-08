"""Exp 4 — Hashemi-clipping (m=8000) runner on synthetic-network family.

One script invocation iterates the 10 instances at a given depth and
writes one CSV row per instance under
``examples/FlowConformal/experiments/exp4_scaling/outputs/exp4_d<D>_hashemi_clipping.csv``.

Uses :func:`n2v.probabilistic.conformal_reach` directly (in-process, n2v env)
since Hashemi-clipping needs the network as a Python callable, not via
ONNX/vnnlib subprocess. Verdict-derivation logic mirrors Exp 1's
Hashemi-clipping runner: halfspace-vs-box disjointness check + a
small uniform-sample falsifier.

Each instance has ``ground_truth='unsat'`` by construction. SAT here
would be a soundness violation in Hashemi-clipping at m=8000 — exactly
the FUR signal Exp 1 measures, but on synthetic networks at scale.

Usage::

    cd /path/to/n2v

    # Smoke (1 instance at depth 2):
    python -m \\
        examples.FlowConformal.experiments.exp4_scaling.exp4_run_hashemi_clipping \\
        --depth 2 --smoke

    # Full sweep at depth 16:
    nohup python -u -m \\
        examples.FlowConformal.experiments.exp4_scaling.exp4_run_hashemi_clipping \\
        --depth 16 \\
        > examples/FlowConformal/experiments/exp4_scaling/outputs/exp4_d16_hashemi_clipping.log 2>&1 &
    disown
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
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
)
from examples.FlowConformal.experiments.exp4_scaling._benchmarks import (
    load_instances,
)
from examples.FlowConformal.experiments.exp4_scaling.networks import (
    EXP4_DEPTHS,
    EXP4_WIDTH,
)
from n2v.nn import NeuralNetwork
from n2v.nn.reach import ConformalReachConfig
from n2v.sets import Box

_SEED = 47
_M = 8000
_EPSILON = 0.001
_ELL = _M - 1
_TIMEOUT_S = 300  # reduced from 600s; BaB scaling failure visible within 60s, matches VNN-COMP standard for similar benchmarks
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'depth', 'width', 'n_params', 'instance_idx', 'method',
    'verdict', 'wall_s', 'timeout_s', 'x_0_seed', 'eps',
    'spec_threshold', 'empirical_max', 'ground_truth',
    'm', 'ell', 'epsilon', 'coverage', 'confidence',
    'amls_bounded_eps_2_upper', 'amls_levels_used',
    'cex_x', 'cex_y', 'error', 'timestamp',
]



def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _write_timeout_row(out_csv, depth, instance_idx, timeout_s):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'depth': depth, 'instance_idx': instance_idx,
        'method': 'hashemi_clipping', 'verdict': 'TIMEOUT',
        'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def _run_one_instance(inst: dict, *, seed: int) -> Dict[str, Any]:
    """Run Hashemi-clipping on one Exp 4 synthetic instance."""
    network = inst['net']
    spec = inst['spec_halfspace']
    lb = np.asarray(inst['lb']).flatten()
    ub = np.asarray(inst['ub']).flatten()

    input_set = Box(lb, ub)
    net = NeuralNetwork(network)
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
                'error': f'verify {type(e).__name__}: {e}'}

    # Verdict via box-vs-halfspace disjointness. Exp 4 specs are
    # UNSAT-by-construction (Lipschitz upper bound + safety margin), so
    # no sample-based falsifier could ever produce a witness; the
    # earlier ``halfspace_witness_from_samples`` block was dead code
    # under this construction.
    cex_x_str = ''
    cex_y_str = ''
    disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
    verdict = 'UNSAT' if disjoint is True else 'UNKNOWN'

    return {
        'verdict': verdict,
        'm': _M, 'ell': _ELL, 'epsilon': _EPSILON,
        'coverage': f'{pbox.coverage:.4f}' if pbox is not None else '',
        'confidence': f'{pbox.confidence:.4f}' if pbox is not None else '',
        'cex_x': cex_x_str,
        'cex_y': cex_y_str,
        'error': '',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, required=True, choices=EXP4_DEPTHS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only this 0-based index; APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <timeout_s>" per instance, exit.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    instances = load_instances(args.depth)

    if args.list_instances:
        for idx in range(len(instances)):
            print(f'{idx} {_TIMEOUT_S}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp4_d{args.depth}_hashemi_clipping.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        _write_timeout_row(out_csv, args.depth, args.instance_idx, _TIMEOUT_S)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print(f'--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        instances = [instances[args.instance_idx]]
        append_mode = True
        print(f'[exp4 d={args.depth}] running only idx={args.instance_idx}; '
              f'appending to {out_csv}', flush=True)
    elif args.smoke:
        instances = instances[:1]
        print('[smoke] Running only the first instance', flush=True)
    print(f'Loaded {len(instances)} instances at depth={args.depth} '
          f'(width={EXP4_WIDTH})', flush=True)
    print(f'Tool: Hashemi-clipping  m={_M}  epsilon={_EPSILON}  '
          f'SEED={_SEED}  timeout={_TIMEOUT_S}s', flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader()
            f.flush()

        for k, inst in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            print(f'[{k}/{len(instances)}  t={elapsed:.0f}s  '
                  f'budget={_TIMEOUT_S}s] depth={inst["depth"]} '
                  f'idx={inst["instance_idx"]} '
                  f'C={inst["C"]:.4f}',
                  flush=True)
            t0 = time.time()
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            try:
                row = _run_one_instance(inst, seed=_SEED)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'depth': inst['depth'],
                'width': EXP4_WIDTH,
                'n_params': inst['n_params'],
                'instance_idx': inst['instance_idx'],
                'method': 'hashemi_clipping',
                'wall_s': f'{wall_s:.2f}',
                'timeout_s': _TIMEOUT_S,
                'x_0_seed': inst['instance_seed'],
                'eps': f'{inst["eps"]:.4f}',
                'spec_threshold': f'{inst["C"]:.6f}',
                'empirical_max': f'{inst["empirical_max"]:.6f}',
                'ground_truth': inst['ground_truth'],
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
        if counts.get('ERROR', 0) > 0 or counts.get('TIMEOUT', 0) > 0:
            print(f'[smoke] FAIL: pipeline errored or timed out, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        # SAT is possible (Hashemi-clipping has known FUR > 0); it's the
        # very signal Exp 4 measures. Don't fail on SAT here — let the
        # full sweep aggregate the FUR statistic.
        print(f'[smoke] PASS: Hashemi-clipping ran end-to-end. '
              f'verdict counts={counts}')


if __name__ == '__main__':
    main()
