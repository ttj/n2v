"""Exp 4 — bounded-AMLS (ours) runner.

One script call iterates the 10 instances at a given depth and writes
one CSV row per instance under
``examples/FlowConformal/experiments/exp4_scaling/outputs/exp4_d<D>_ours.csv``.

Usage::

    cd /path/to/n2v

    # Smoke (1 instance at depth 2, expects UNSAT):
    python -m \\
        examples.FlowConformal.experiments.exp4_scaling.exp4_run_ours \\
        --depth 2 --smoke

    # Full sweep at depth 16:
    nohup python -u -m \\
        examples.FlowConformal.experiments.exp4_scaling.exp4_run_ours \\
        --depth 16 \\
        > examples/FlowConformal/experiments/exp4_scaling/outputs/exp4_d16_ours.log 2>&1 &
    disown

Per the FlowConformal seeding convention, the verification RNG is reset
to ``SEED=47`` at the top of every instance. Instance generation uses a
separate RNG seeded from ``hash((depth, instance_idx))`` so the 10
instances per depth differ in their starting sample ``x_0``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments._runner_utils import (
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments._shared_flow_runner import (
    run_flow_pipeline,
)
from examples.FlowConformal.experiments.exp4_scaling._benchmarks import (
    load_instances,
)
from examples.FlowConformal.experiments.exp4_scaling.networks import (
    EXP4_DEPTHS,
    EXP4_WIDTH,
)

_SEED = 47
_TIMEOUT_S = 300  # reduced from 600s; BaB scaling failure visible within 60s, matches VNN-COMP standard for similar benchmarks
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'depth', 'width', 'n_params', 'instance_idx', 'method',
    'verdict', 'wall_s', 'train_s', 'verify_s',
    'timeout_s', 'x_0_seed', 'eps',
    'spec_threshold', 'empirical_max', 'ground_truth',
    'coverage', 'q', 'epsilon_total', 'delta_total',
    'amls_bounded_eps_2_upper', 'amls_bounded_detected_unsafe',
    'amls_levels_used',
    'cex_x', 'cex_y', 'error', 'timestamp',
]

# Single config across depths. The synthetic networks are 1-Lipschitz so
# bounded-AMLS rejection geometry is well-behaved. We use the ``mega``
# training budget (n_train=10K, flow_epochs=2K, scenario_n=2K) — the
# locked config from Exp 2 cifar10_resnet110 / vit_2023 — paired with
# the standard ``base`` flow architecture (h128/L4) from _common.py.
_CONFIG = dict(
    flow_config='base',
    n_train=10_000,
    flow_epochs=2_000,
    scenario_n_samples=2_000,
    alpha=0.001,
    verification_method='amls_bounded',
    amls_max_levels=30,
)



def _run_one_instance(inst: dict, *, seed: int) -> dict:
    """Run the bounded-AMLS pipeline on a single Exp 4 instance.

    Returns a row dict; never raises (errors land in the ``error`` field).
    """
    # Move network to GPU for the sample-generation forward passes
    # (n_train + m + n_test = 20K passes per instance). The network is
    # the dominant CPU cost — moving to CUDA collapses "other" wall
    # from O(d) seconds per depth to O(1).
    net = inst['net']
    if torch.cuda.is_available():
        net = net.cuda()
    # Exp 4 instances are UNSAT-by-construction (1-Lipschitz network +
    # tight halfspace spec). Falsifier can't find a CEX and would burn
    # ~115-190 s of PGD/APGD per instance. Skipping it here mirrors
    # αβ-CROWN's ``attack: pgd_order: skip`` and NeuralSAT's
    # ``--disable_attack`` for an apples-to-apples scaling comparison.
    cfg = dict(
        alpha=_CONFIG['alpha'],
        n_train=_CONFIG['n_train'],
        flow_epochs=_CONFIG['flow_epochs'],
        flow_config=_CONFIG['flow_config'],
        scenario_n_samples=_CONFIG['scenario_n_samples'],
        verification_method=_CONFIG['verification_method'],
        amls_max_levels=_CONFIG['amls_max_levels'],
        amls_bounded_eps_2_target=_CONFIG.get('amls_bounded_eps_2_target'),
        use_falsifier=False,
    )
    try:
        r = run_flow_pipeline(
            net,
            np.asarray(inst['lb']), np.asarray(inst['ub']),
            inst['spec_halfspace'], cfg, seed=seed,
        )
    except NotImplementedError as e:
        return {'verdict': 'SKIPPED', 'error': f'{type(e).__name__}: {e}'}
    except Exception as e:
        return {'verdict': 'ERROR', 'error': f'runfailed {type(e).__name__}: {e}'}

    def _fmt(v, spec):
        return f'{v:{spec}}' if v is not None else ''

    cex_x, cex_y = '', ''
    if r.get('counterexample') is not None:
        ce = r['counterexample']
        cex_x = json.dumps(ce['x'].tolist())
        cex_y = json.dumps(ce['y'].tolist())
    amls_lvls = r.get('amls_levels_used')
    return {
        'verdict': r['verdict'],
        'wall_s': _fmt(r.get('total_time_s'), '.1f'),
        'train_s': _fmt(r.get('flow_train_time_s'), '.1f'),
        'verify_s': _fmt(r.get('verification_time_s'), '.1f'),
        'coverage': _fmt(r.get('coverage_empirical'), '.4f'),
        'q': _fmt(r.get('q'), '.4f'),
        'epsilon_total': _fmt(r.get('epsilon_total'), '.4f'),
        'delta_total': _fmt(r.get('delta_total'), '.4f'),
        'amls_bounded_eps_2_upper': _fmt(r.get('amls_bounded_eps_2_upper'), '.4e'),
        'amls_bounded_detected_unsafe': str(
            r.get('amls_bounded_detected_unsafe', '')),
        'amls_levels_used': str(amls_lvls) if amls_lvls is not None else '',
        'cex_x': cex_x,
        'cex_y': cex_y,
        'error': '',
    }


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _write_timeout_row(out_csv, depth, instance_idx, timeout_s):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'depth': depth, 'instance_idx': instance_idx,
        'method': 'ours', 'verdict': 'TIMEOUT',
        'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, required=True, choices=EXP4_DEPTHS)
    p.add_argument('--smoke', action='store_true',
                   help='Run only the first instance and assert UNSAT.')
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

    instances = load_instances(args.depth)

    if args.list_instances:
        for idx in range(len(instances)):
            print(f'{idx} {_TIMEOUT_S}')
        return

    out_csv = (
        args.output_csv if args.output_csv is not None
        else _OUT_DIR / f'exp4_d{args.depth}_ours.csv'
    )
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
            print(f'--instance-idx {args.instance_idx} out of range',
                  file=sys.stderr)
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
    print(f'Config: {_CONFIG}  SEED={_SEED}', flush=True)
    print(f'Timeout policy: fixed {_TIMEOUT_S}s per instance', flush=True)

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

        for k, inst in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            print(f'[{k}/{len(instances)}  t={elapsed:.0f}s  '
                  f'budget={_TIMEOUT_S}s] depth={inst["depth"]} '
                  f'idx={inst["instance_idx"]} '
                  f'C={inst["C"]:.4f} '
                  f'(emax={inst["empirical_max"]:.4f})',
                  flush=True)
            t0 = time.time()
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            try:
                row = _run_one_instance(inst, seed=_SEED)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'depth': inst['depth'],
                'width': EXP4_WIDTH,
                'n_params': inst['n_params'],
                'instance_idx': inst['instance_idx'],
                'method': 'ours',
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
            print(f'    verdict={row["verdict"]}  wall={time.time()-t0:.1f}s',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        if counts.get('UNSAT', 0) != 1:
            print(f'[smoke] FAIL: expected UNSAT on first instance, '
                  f'got counts={counts}', file=sys.stderr)
            sys.exit(1)
        print('[smoke] PASS: first instance UNSAT as expected.')


if __name__ == '__main__':
    main()
