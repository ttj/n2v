"""Exp 1 — bounded-AMLS (ours) runner.

One script invocation iterates the entire ``instances.csv`` for a
benchmark and writes one CSV row per instance under
``examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/exp1_<benchmark>_ours.csv``.

Usage::

    cd /path/to/n2v

    # Smoke (1 instance, hand-checked verdict; <2 min):
    python -m \\
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_ours \\
        --benchmark acasxu_2023 --smoke

    # Full sweep at VNN-COMP per-row timeout:
    nohup python -u -m \\
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_ours \\
        --benchmark dist_shift_2023 \\
        > examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/exp1_dist_shift_2023_ours.log 2>&1 &
    disown

Per the FlowConformal seeding convention, the verification RNG is
reset to ``SEED=47`` at the top of every instance. The Exp 1 sweep
honours the VNN-COMP per-row timeout (``instances.csv`` column 3) so
ours runs at the same budget the sound verifiers got.

For OR-of-input-regions instances (e.g. ACAS Xu prop_6 has 4 disjoint
input boxes), each box is verified independently and the verdicts are
Bonferroni-aggregated: any SAT short-circuits to SAT, all UNSAT
aggregates to UNSAT (eps_total summed, delta_total intersected),
otherwise UNKNOWN.
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
    aggregate_box_verdicts,
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments._shared_flow_runner import (
    run_flow_pipeline,
)
from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    EXP1_BENCHMARKS,
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)

_SEED = 47
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'onnx_file', 'vnnlib_file', 'verdict',
    'wall_s', 'train_s', 'verify_s',
    'vnncomp_timeout_s', 'coverage', 'q', 'epsilon_total', 'delta_total',
    'amls_bounded_eps_2_upper', 'amls_bounded_detected_unsafe',
    'amls_levels_used',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


def _run_one_instance(benchmark: str, onnx_rel: str, vnn_rel: str,
                      cfg: dict, *, seed: int) -> dict:
    """Run our pipeline on one VNN-COMP instance.

    Returns a row dict; never raises (load/run failures become
    ``ERROR`` rows so the outer loop logs them and continues).
    """
    try:
        network, boxes, spec = load_one_instance(benchmark, onnx_rel, vnn_rel)
    except NotImplementedError as e:
        return {'verdict': 'SKIPPED',
                'error': f'{type(e).__name__}: {e}'}
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'loadfailed {type(e).__name__}: {e}'}

    # Move network to GPU for sample-generation forward passes
    # (n_train + m + n_test = 20K passes per instance). The flow
    # pipeline's patched ``_forward`` pushes inputs to the network's
    # device. If a network has ops that fail on CUDA, the pipeline's
    # per-instance try/except surfaces it as an ERROR row.
    if torch.cuda.is_available():
        try:
            network = network.cuda()
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'gpu_move {type(e).__name__}: {e}'}

    box_results = []
    for box_idx, (lb, ub) in enumerate(boxes):
        try:
            r = run_flow_pipeline(network, lb, ub, spec, cfg, seed=seed)
        except NotImplementedError as e:
            return {'verdict': 'SKIPPED',
                    'error': f'{type(e).__name__}: {e}'}
        except TimeoutError:
            raise
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'runfailed box={box_idx} {type(e).__name__}: {e}'}
        box_results.append(r)
        if r['verdict'] == 'SAT':
            break

    # Aggregate verdicts across boxes (OR-of-input-regions union).
    result = aggregate_box_verdicts(box_results)

    cex_x, cex_y = '', ''
    if result.get('counterexample') is not None:
        ce = result['counterexample']
        cex_x = json.dumps(np.asarray(ce['x']).tolist())
        cex_y = json.dumps(np.asarray(ce['y']).tolist())
    amls_lvls = result.get('amls_levels_used')
    return {
        'verdict': result['verdict'],
        'wall_s': _fmt(result.get('total_time_s'), '.1f'),
        'train_s': _fmt(result.get('flow_train_time_s'), '.1f'),
        'verify_s': _fmt(result.get('verification_time_s'), '.1f'),
        'coverage': _fmt(result.get('coverage_empirical'), '.4f'),
        'q': _fmt(result.get('q'), '.4f'),
        'epsilon_total': _fmt(result.get('epsilon_total'), '.4f'),
        'delta_total': _fmt(result.get('delta_total'), '.4f'),
        'amls_bounded_eps_2_upper': _fmt(
            result.get('amls_bounded_eps_2_upper'), '.4e'),
        'amls_bounded_detected_unsafe': str(
            result.get('amls_bounded_detected_unsafe', '')),
        'amls_levels_used': str(amls_lvls) if amls_lvls is not None else '',
        'cex_x': cex_x,
        'cex_y': cex_y,
        'error': '',
    }


def _open_for_append(out_csv: Path):
    """Open ``out_csv`` for append, writing the header only when the
    file is empty/new. Returns the open file handle and a DictWriter
    bound to ``_FIELDS``.

    Caller is responsible for closing.
    """
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    f = open(out_csv, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=_FIELDS)
    if not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer


def _write_timeout_row(out_csv: Path, benchmark: str,
                       onnx_rel: str, vnn_rel: str,
                       vnncomp_t: int) -> None:
    """Append a single TIMEOUT row for an instance that was killed by
    the outer shell timeout (run_cell.sh exit 124). Mirrors VNN-COMP's
    pattern of having the bash wrapper write the timeout row when the
    Python process can't.
    """
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark,
        'onnx_file': Path(onnx_rel).name,
        'vnnlib_file': Path(vnn_rel).name,
        'verdict': 'TIMEOUT',
        'vnncomp_timeout_s': vnncomp_t,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def _run_and_write(out_csv: Path, benchmark: str, cfg: dict,
                   instances: list, *, append: bool) -> dict:
    """Loop the given (subset of) instances, write/append rows to ``out_csv``.

    Per-instance timeouts are handled by the OUTER shell timeout in
    ``run_cell.sh`` (single timeout layer; mirrors VNN-COMP's
    ``run_single_instance.sh`` pattern). This function trusts that
    each call processes within the shell budget; if a hang occurs,
    the shell SIGKILLs our Python process and ``run_cell.sh``
    re-invokes us with ``--write-timeout-row`` to record the row.
    """
    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'SKIPPED': 0, 'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    mode = 'a' if append and file_exists else 'w'
    with open(out_csv, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or mode == 'w':
            writer.writeheader()
            f.flush()

        for k, (onnx_rel, vnn_rel, vnncomp_t) in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            timeout_s = vnncomp_t if vnncomp_t > 0 else 600
            print(f'[{benchmark} {k}/{len(instances)} t={elapsed:.0f}s '
                  f'budget={timeout_s}s] {onnx_rel} + {vnn_rel}',
                  flush=True)
            t0 = time.time()
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            try:
                row = _run_one_instance(
                    benchmark, onnx_rel, vnn_rel, cfg, seed=_SEED)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'onnx_file': Path(onnx_rel).name,
                'vnnlib_file': Path(vnn_rel).name,
                'vnncomp_timeout_s': vnncomp_t,
                'timestamp': _now_iso(),
            })
            out_row.update(row)
            writer.writerow(out_row)
            f.flush()

            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  '
                  f'wall={time.time()-t0:.1f}s', flush=True)
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP1_BENCHMARKS)
    p.add_argument('--smoke', action='store_true',
                   help='Run only the first instance and assert the '
                        'pipeline ran end-to-end (no ERROR/TIMEOUT/'
                        'SKIPPED). Mutually exclusive with '
                        '--instance-idx / --list-instances.')
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only the instance at this 0-based index '
                        '(skipping all others) and APPEND its row to '
                        'the output CSV. Used by run_cell.sh to enforce '
                        'per-instance shell timeouts. Mutually exclusive '
                        'with --smoke.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <vnncomp_timeout_s>" lines for '
                        'each instance and exit. Used by run_cell.sh '
                        'to know what to loop over.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for the instance at '
                        '--instance-idx and exit 0. Called by run_cell.sh '
                        'on outer-timeout (exit 124) so the CSV always has '
                        'a row for every instance.')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    cfg = PER_BENCHMARK_CONFIG[benchmark]
    instances = list_instances(benchmark)

    # --list-instances: print idx + per-instance timeout, exit.
    if args.list_instances:
        for idx, (_o, _v, vnncomp_t) in enumerate(instances):
            print(f'{idx} {vnncomp_t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp1_{benchmark}_ours.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --write-timeout-row: append a TIMEOUT row for an instance whose
    # actual run was killed by the outer shell timeout. Called by
    # run_cell.sh on exit 124. Requires --instance-idx.
    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print(f'--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
        _write_timeout_row(out_csv, benchmark, onnx_rel, vnn_rel, vnncomp_t)
        return

    # --instance-idx: run only instances[idx], append to CSV.
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print(f'--instance-idx {args.instance_idx} out of range '
                  f'[0, {len(instances)})', file=sys.stderr)
            sys.exit(2)
        only = [instances[args.instance_idx]]
        print(f'[{benchmark}] running only instance '
              f'idx={args.instance_idx}; appending to {out_csv}',
              flush=True)
        _run_and_write(out_csv, benchmark, cfg, only, append=True)
        return

    # Default: whole-cell mode (or --smoke = 1 instance, fresh CSV).
    if args.smoke:
        instances = instances[:1]
        print(f'[smoke] {benchmark}: running only the first instance',
              flush=True)
    print(f'[{benchmark}] Loaded {len(instances)} instances; '
          f'writing to {out_csv}', flush=True)
    print(f'[{benchmark}] Config: flow_config={cfg["flow_config"]} '
          f'n_train={cfg["n_train"]} flow_epochs={cfg["flow_epochs"]} '
          f'alpha={cfg["alpha"]} method={cfg["verification_method"]} '
          f'max_levels={cfg["amls_max_levels"]} SEED={_SEED}', flush=True)

    t_start = time.time()
    counts = _run_and_write(out_csv, benchmark, cfg, instances, append=False)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        # The smoke validates that the pipeline ran end-to-end without
        # ERROR/TIMEOUT/SKIPPED — those signal genuine problems
        # (broken loader, oversized config, OOM, unsupported spec).
        # UNKNOWN is a legitimate honest abstention and SAT is a
        # legitimate falsification; both indicate a working pipeline.
        # Per-instance verdict-vs-ground-truth comparison is the job
        # of the full sweep, not the smoke.
        bad = {'ERROR', 'TIMEOUT', 'SKIPPED'}
        if any(counts.get(v, 0) > 0 for v in bad):
            print(f'[smoke] FAIL on {benchmark}: expected pipeline to '
                  f'complete without ERROR/TIMEOUT/SKIPPED, got '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next(v for v, c in counts.items() if c > 0)
        from examples.FlowConformal.experiments._ground_truth_lookup import (
            lookup_ground_truth,
        )
        onnx_rel, vnn_rel, _ = instances[0]
        first_inst = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'
        gt = lookup_ground_truth('exp1', benchmark, first_inst)
        match = '✓' if actual == gt else '≠'
        print(f'[smoke] PASS on {benchmark}: pipeline ran end-to-end '
              f'(verdict={actual} {match} VNN-COMP ground truth '
              f'{gt}).')


if __name__ == '__main__':
    main()
