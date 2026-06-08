"""Exp 2 — bounded-AMLS (ours) runner.

One script invocation iterates the first N (default 100) instances for
a benchmark and writes one CSV row per instance under
``examples/FlowConformal/experiments/exp2_prob_scale/outputs/exp2_<benchmark>_ours.csv``.

Per the README, Exp 2 timeouts vary by benchmark:
    * vit_2023, tinyimagenet_2024 — VNN-COMP per-row
    * cifar10_resnet110 — fixed 300s
    * cifar100_2024 — fixed 100s

cifar100_2024 uses ``verification_method='amls_bounded_union'``
(single chain on ``phi_min`` over the 99 other-class halfspaces);
all others use the standard per-halfspace ``amls_bounded`` chain.

Falsifier is OFF (per Exp 2 design — "probabilistic vs sound at scale";
SAT comes from each method's intrinsic mechanism, not an added
PGD/APGD ensemble).

Usage::

    cd /path/to/n2v

    # Smoke (1 instance):
    python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_ours \\
        --benchmark vit_2023 --smoke

    # Full sweep (N=100):
    nohup python -u -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_ours \\
        --benchmark cifar100_2024 \\
        > examples/FlowConformal/experiments/exp2_prob_scale/outputs/exp2_cifar100_2024_ours.log 2>&1 &
    disown
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
from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
    EXP2_BENCHMARKS,
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
from examples.FlowConformal.experiments._shared_flow_runner import (
    run_flow_pipeline,
)

_SEED = 47
_DEFAULT_N_INSTANCES = 100
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'instance', 'verdict',
    'wall_s', 'train_s', 'verify_s',
    'timeout_s', 'coverage', 'q', 'epsilon_total', 'delta_total',
    'amls_bounded_eps_2_upper', 'amls_bounded_detected_unsafe',
    'amls_levels_used',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _write_timeout_row(out_csv: Path, benchmark: str, name: str,
                       timeout_s: int) -> None:
    """Append a single TIMEOUT row when killed by outer shell timeout."""
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark,
        'instance': name,
        'verdict': 'TIMEOUT',
        'wall_s': '',
        'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


def _run_one_instance(benchmark: str, loader, cfg: dict, *,
                      seed: int) -> dict:
    """Run our pipeline on one Exp 2 instance.

    Returns a row dict; never raises.
    """
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
            r = run_flow_pipeline(
                network,
                np.asarray(lb).flatten(),
                np.asarray(ub).flatten(),
                spec, cfg, seed=seed,
            )
        except NotImplementedError as e:
            return {'verdict': 'SKIPPED',
                    'error': f'unsupported_spec {type(e).__name__}: {e}'}
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'runfailed box={box_idx} {type(e).__name__}: {e}'}
        box_results.append(r)
        if r['verdict'] == 'SAT':
            break

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
        'cex_x': cex_x, 'cex_y': cex_y, 'error': '',
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
               else _OUT_DIR / f'exp2_{benchmark}_ours.csv')
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
    print(f'[{benchmark}] Config: flow_config={cfg["flow_config"]} '
          f'n_train={cfg["n_train"]} flow_epochs={cfg["flow_epochs"]} '
          f'method={cfg["verification_method"]} '
          f'max_levels={cfg["amls_max_levels"]} SEED={_SEED}', flush=True)

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
                row = _run_one_instance(benchmark, loader, cfg, seed=_SEED)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'instance': name,
                'timeout_s': timeout_s,
                'timestamp': _now_iso(),
            })
            out_row.update(row)
            if 'wall_s' not in row or not row.get('wall_s'):
                out_row['wall_s'] = f'{wall_s:.1f}'
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
        match = '✓' if actual == gt else '≠'
        print(f'[smoke] PASS on {benchmark}: pipeline ran end-to-end '
              f'(verdict={actual} {match} VNN-COMP ground truth {gt}).')


if __name__ == '__main__':
    main()
