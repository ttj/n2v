"""Exp 1 — ProbStar (StarV) runner.

Mirrors :mod:`exp2_prob_scale.exp2_run_probstar` but consumes Exp 1's
benchmark roots (which use the same VNN-COMP-format ONNX + vnnlib
layout). Subprocess-dispatches each instance to the ``starv`` conda
env via :func:`examples.FlowConformal.experiments._external_verifiers.run_probstar`.

ProbStar's StarV loader is restrictive (feedforward Conv2D /
MatMul+Add / ReLU / BatchNorm / MaxPool / AvgPool only); empirically
the Exp 1 networks may not all parse, in which case the runner records
NOT_APPLICABLE per the standalone's contract. The user explicitly
opted to try ProbStar on Exp 1 anyway.

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_probstar \\
        --benchmark acasxu_2023 --smoke
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
from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    BENCHMARK_ROOTS,
    EXP1_BENCHMARKS,
    list_instances,
)

_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'onnx_file', 'vnnlib_file', 'verdict',
    'wall_s', 'vnncomp_timeout_s',
    'p_filter', 'lp_solver', 'p_min', 'p_max', 'threshold',
    'n_disjuncts',
    'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _resolve_paths(benchmark: str, onnx_rel: str, vnn_rel: str
                   ) -> tuple[Path, Path]:
    """Resolve ``instances.csv`` relative paths to absolute on-disk paths.

    For ACAS Xu the ONNX lives under ``examples/ACASXu/`` (sibling of
    the FlowConformal experiments), not under the VNN-COMP root; the
    Exp 1 ours-runner replicates that dispatch. We mirror it here so
    StarV can read the ONNX file.
    """
    if benchmark == 'acasxu_2023':
        acasxu_root = (Path(__file__).resolve().parents[3] / 'ACASXu')
        onnx_path = acasxu_root / onnx_rel.removeprefix('./')
    else:
        root = BENCHMARK_ROOTS[benchmark]
        onnx_path = root / onnx_rel.removeprefix('./')
    root = BENCHMARK_ROOTS[benchmark]
    vnn_path = root / vnn_rel.removeprefix('./')
    return onnx_path, vnn_path


def _write_timeout_row(out_csv: Path, benchmark: str,
                       onnx_rel: str, vnn_rel: str,
                       timeout_s: int) -> None:
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark,
        'onnx_file': Path(onnx_rel).name,
        'vnnlib_file': Path(vnn_rel).name,
        'verdict': 'TIMEOUT', 'vnncomp_timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP1_BENCHMARKS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--instance-idx', type=int, default=None)
    p.add_argument('--list-instances', action='store_true')
    p.add_argument('--p-filter', type=float, default=0.0)
    p.add_argument('--lp-solver', type=str, default='gurobi',
                   choices=['gurobi', 'glpk'])
    p.add_argument('--gauss-alpha', type=float, default=2.5)
    p.add_argument('--unsafe-threshold', type=float, default=1e-3)
    p.add_argument('--write-timeout-row', action='store_true')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    instances = list_instances(benchmark)

    if args.list_instances:
        for idx, (_o, _v, t) in enumerate(instances):
            print(f'{idx} {t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp1_{benchmark}_probstar.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        onnx_rel, vnn_rel, t = instances[args.instance_idx]
        _write_timeout_row(out_csv, benchmark, onnx_rel, vnn_rel, t)
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
    elif args.smoke:
        target_indices = [0]
        append_mode = False
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
            onnx_rel, vnn_rel, t = instances[idx]
            try:
                onnx_path, vnn_path = _resolve_paths(
                    benchmark, onnx_rel, vnn_rel)
            except Exception as e:
                row = {fld: '' for fld in _FIELDS}
                row.update({
                    'benchmark': benchmark,
                    'onnx_file': Path(onnx_rel).name,
                    'vnnlib_file': Path(vnn_rel).name,
                    'verdict': 'ERROR',
                    'vnncomp_timeout_s': t,
                    'error': f'path_resolve {type(e).__name__}: {e}',
                    'timestamp': _now_iso(),
                })
                writer.writerow(row)
                f.flush()
                counts['ERROR'] = counts.get('ERROR', 0) + 1
                continue

            tag = f'exp1_{benchmark}_idx{idx}'
            print(f'[{benchmark} {k}/{len(target_indices)} '
                  f't={elapsed:.0f}s budget={t}s] idx={idx} '
                  f'{onnx_path.name}+{vnn_path.name}', flush=True)
            verdict, wall_s, err, extras = run_probstar(
                onnx_path=onnx_path,
                vnnlib_path=vnn_path,
                timeout_s=t if t > 0 else 600,
                instance_tag=tag,
                p_filter=args.p_filter,
                lp_solver=args.lp_solver,
                gauss_alpha=args.gauss_alpha,
                unsafe_threshold=args.unsafe_threshold,
            )
            row = {fld: '' for fld in _FIELDS}
            row.update({
                'benchmark': benchmark,
                'onnx_file': onnx_path.name,
                'vnnlib_file': vnn_path.name,
                'verdict': verdict,
                'wall_s': f'{wall_s:.2f}' if wall_s is not None else '',
                'vnncomp_timeout_s': t,
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
        if counts.get('ERROR', 0) > 0:
            print(f'[smoke] FAIL on {benchmark}: ERROR observed, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next((v for v, c in counts.items() if c > 0), 'NONE')
        print(f'[smoke] PASS on {benchmark}: ProbStar ran '
              f'(verdict={actual}).')


if __name__ == '__main__':
    main()
