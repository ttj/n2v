"""Exp 2 — ProbStar (StarV) runner.

Subprocess-dispatches each instance to the ``starv`` conda env via
:func:`examples.FlowConformal.experiments._external_verifiers.run_probstar`
(mirrors how αβ-CROWN and NeuralSAT are dispatched). The actual
verification work runs in :mod:`baselines.run_probstar_standalone`,
which has zero n2v dependencies — only StarV + numpy + onnx.

ProbStar is fundamentally restricted by what StarV's ``load_onnx_network``
can parse: feedforward Conv2D / FullyConnected (via MatMul+Add, NOT
Gemm) / ReLU / BatchNorm / MaxPool / AvgPool. Networks that include
transformer attention, residual ``Add`` skip-connections, ``Gemm``
nodes, softmax, sigmoid, or GeLU emit ``NOT_APPLICABLE`` from the
standalone — empirically, ALL Exp 2 benchmarks hit one of these
limitations (vit_2023 = transformer; tinyimagenet/cifar100 ResNets =
residual Add; cifar10_resnet110 = same plus 110 layers). The runner
records the NOT_APPLICABLE outcome for the aggregator either way.

A Gurobi license is required for non-trivial runs (StarV's reach LP
defaults to gurobi). The runner falls back to glpk via ``--lp-solver
glpk`` if needed.

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_probstar \\
        --benchmark cifar100_2024 --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

from examples.FlowConformal.experiments._external_verifiers import (
    run_probstar,
)
from examples.FlowConformal.experiments._runner_utils import (
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
    EXP2_VNNCOMP_FORMAT,
    list_vnncomp_format_instances,
)

_DEFAULT_N_INSTANCES = 100
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'instance', 'verdict', 'wall_s', 'timeout_s',
    'p_filter', 'lp_solver', 'p_min', 'p_max', 'threshold',
    'n_disjuncts',
    'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _write_timeout_row(out_csv, benchmark, name, timeout_s):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'instance': name,
        'verdict': 'TIMEOUT', 'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP2_VNNCOMP_FORMAT,
                   help=f'ProbStar applies natively only to PWL networks '
                        f'StarV\'s loader can ingest. On non-PWL or '
                        f'unsupported-op benchmarks the standalone emits '
                        f'NOT_APPLICABLE per instance. Choices: '
                        f'{EXP2_VNNCOMP_FORMAT}.')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-instances', type=int, default=_DEFAULT_N_INSTANCES)
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only this 0-based index; APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <timeout_s>" per instance, exit.')
    p.add_argument('--p-filter', type=float, default=0.0,
                   help='Probability filter for ProbStar reach '
                        '(0.0 = exact). Default 0.0.')
    p.add_argument('--lp-solver', type=str, default='gurobi',
                   choices=['gurobi', 'glpk'],
                   help='LP solver for star ranging. Gurobi requires a '
                        'license; glpk is free but slower. Default gurobi.')
    p.add_argument('--gauss-alpha', type=float, default=2.5,
                   help='Coefficient adjusting the truncation: '
                        'sigma = (mu - lb) / alpha. Default 2.5.')
    p.add_argument('--unsafe-threshold', type=float, default=1e-3,
                   help='Pr[unsafe] threshold above which we declare SAT. '
                        'Default 1e-3.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    instances = list_vnncomp_format_instances(benchmark, n=args.n_instances)

    if args.list_instances:
        for idx, (_o, _v, t) in enumerate(instances):
            print(f'{idx} {t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp2_{benchmark}_probstar.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        onnx_path, vnnlib_path, t = instances[args.instance_idx]
        name = f'{onnx_path.name}+{vnnlib_path.name}'
        _write_timeout_row(out_csv, benchmark, name, t)
        return

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
            onnx_path, vnnlib_path, t = instances[idx]
            name = f'{onnx_path.name}+{vnnlib_path.name}'
            print(f'[{benchmark} {k}/{len(target_indices)} t={elapsed:.0f}s '
                  f'budget={t}s] idx={idx} {name}', flush=True)
            tag = f'exp2_{benchmark}_idx{idx}'
            verdict, wall_s, err, extras = run_probstar(
                onnx_path=onnx_path,
                vnnlib_path=vnnlib_path,
                timeout_s=t,
                instance_tag=tag,
                p_filter=args.p_filter,
                lp_solver=args.lp_solver,
                gauss_alpha=args.gauss_alpha,
                unsafe_threshold=args.unsafe_threshold,
            )
            row = {fld: '' for fld in _FIELDS}
            row.update({
                'benchmark': benchmark,
                'instance': name,
                'verdict': verdict,
                'wall_s': f'{wall_s:.2f}' if wall_s is not None else '',
                'timeout_s': t,
                'p_filter': extras.get('p_filter', args.p_filter),
                'lp_solver': extras.get('lp_solver', args.lp_solver),
                'p_min': extras.get('p_min', ''),
                'p_max': extras.get('p_max', ''),
                'threshold': extras.get('threshold', args.unsafe_threshold),
                'n_disjuncts': extras.get('n_disjuncts', ''),
                'error': extras.get('error', err),
                'timestamp': _now_iso(),
            })
            writer.writerow(row)
            f.flush()
            counts[verdict] = counts.get(verdict, 0) + 1
            print(f'    verdict={verdict}  wall={row["wall_s"]}s',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        # Smoke pass: ProbStar emits NOT_APPLICABLE when StarV can't
        # parse the network, which is still a successful subprocess
        # invocation (the runner ran; the tool just doesn't apply).
        # Only ERROR (env / dispatch failure) is a smoke failure.
        if counts.get('ERROR', 0) > 0:
            print(f'[smoke] FAIL on {benchmark}: ERROR observed, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next((v for v, c in counts.items() if c > 0), 'NONE')
        print(f'[smoke] PASS on {benchmark}: ProbStar ran '
              f'(verdict={actual}).')


if __name__ == '__main__':
    main()
