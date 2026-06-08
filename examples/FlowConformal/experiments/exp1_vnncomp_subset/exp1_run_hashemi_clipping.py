"""Exp 1 — Hashemi-clipping (m=8000) runner.

Drops a uniform-sample-based conformal calibration with the
``clipping_block`` surrogate (Hashemi 2025 paper) over each Exp 1
benchmark. One script invocation iterates the entire ``instances.csv``
and writes one CSV row per instance under
``examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/exp1_<benchmark>_hashemi_clipping.csv``.

Per the Exp 1 design, this is the comparison baseline that ours
needs to beat on false-UNSAT-rate (Hashemi-clipping is known to
produce ~25-28% FUR at small m). Verdicts are derived as follows:

* The ``conformal_reach(...)`` call returns a ``ProbabilisticBox`` ``[lb_y, ub_y]``
  with conformal guarantee ``Pr[Pr[f(x) ∈ box] >= 1-ε] >= δ_2``.
* A halfspace check ``halfspace_disjoint_from_box`` certifies UNSAT
  when the box is fully outside the unsafe region in some group.
* A uniform-sample falsifier (~2K samples) flips to SAT if any sample
  lands in the unsafe region — Hashemi has no PGD/APGD ensemble, so
  SAT verdicts come purely from random sampling.

Usage::

    cd /path/to/n2v

    # Smoke (1 instance):
    python -m \\
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_hashemi_clipping \\
        --benchmark acasxu_2023 --smoke

    # Full sweep:
    nohup python -u -m \\
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_hashemi_clipping \\
        --benchmark dist_shift_2023 \\
        > examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/exp1_dist_shift_2023_hashemi_clipping.log 2>&1 &
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
from typing import Any, Dict

import numpy as np
import torch

from examples.FlowConformal.experiments._runner_utils import (
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments.baselines._common import (
    empirical_coverage_for_box,
    halfspace_disjoint_from_box,
    torch_callable,
)
from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    EXP1_BENCHMARKS,
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
_N_TEST_COVERAGE = 1000
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'onnx_file', 'vnnlib_file', 'verdict', 'wall_s',
    'vnncomp_timeout_s', 'm', 'ell', 'epsilon', 'coverage',
    'coverage_empirical', 'coverage_n_test', 'confidence',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _write_timeout_row(out_csv: Path, benchmark: str,
                       onnx_rel: str, vnn_rel: str,
                       vnncomp_t: int) -> None:
    """Append a single TIMEOUT row when killed by outer shell timeout.
    Mirrors VNN-COMP's run_single_instance.sh pattern.
    """
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark,
        'onnx_file': Path(onnx_rel).name,
        'vnnlib_file': Path(vnn_rel).name,
        'verdict': 'TIMEOUT',
        'wall_s': '',
        'vnncomp_timeout_s': vnncomp_t,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _run_one_instance(benchmark: str, onnx_rel: str, vnn_rel: str,
                      *, seed: int) -> Dict[str, Any]:
    """Run Hashemi-clipping on one VNN-COMP instance.

    Falsifier behaviour mirrors :mod:`exp1_run_ours`: when
    ``cfg['use_falsifier']`` is True, calls
    :func:`n2v.utils.falsify.falsify` with the same
    ``(method, n_restarts, n_steps)`` as ours so the only difference
    between the two methods is the score function and the calibrated
    set's geometry. When False, no falsification is performed.

    Returns a row dict; never raises (load/run failures land in the
    ``error`` field with verdict ``ERROR``).
    """
    try:
        network, boxes, spec = load_one_instance(benchmark, onnx_rel, vnn_rel)
    except NotImplementedError as e:
        return {'verdict': 'SKIPPED',
                'error': f'unsupported_spec {type(e).__name__}: {e}'}
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'load {type(e).__name__}: {e}'}

    # Move network to GPU so the m=8000 calibration sampling and
    # APGD falsifier run on GPU (mirrors exp1_run_ours). On networks
    # that don't fit on GPU (or have CPU-only ops), fall back to ERROR
    # rather than silently using CPU — same convention as ours.
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
    # Per-benchmark Hashemi calibration size. Defaults to 8000 (the
    # canonical clipping_block setting); benchmarks with sub-30s
    # VNN-COMP per-instance budgets (lsnc_relu, relusplitter) override
    # to a smaller m so the run fits the budget.
    m = int(cfg.get('hashemi_m', _DEFAULT_M))
    ell = m - 1

    net = NeuralNetwork(network)
    model_fn = torch_callable(network)  # still needed for empirical_coverage_for_box below
    any_unknown = False
    cov_vals: list = []
    cov_n_total = 0
    last_pbox = None
    cex_x_str = ''
    cex_y_str = ''
    sat = False

    for box_idx, (lb, ub) in enumerate(boxes):
        input_set = Box(np.asarray(lb).flatten(),
                        np.asarray(ub).flatten())

        # Stage-1 falsifier (parity with ours): runs BEFORE the
        # Hashemi calibration ``conformal_reach(...)`` so a CEX short-circuits
        # to SAT without paying the calibration cost — matching the
        # falsify-first ordering in ``run_verification_pipeline``. Uses the
        # same APGD method and per-benchmark ``(n_restarts, n_steps)``
        # budget as ``exp1_run_ours``, so the only difference between the
        # two methods is the score function used for set construction.
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
                    m=m, ell=ell, epsilon=_EPSILON,
                    surrogate='clipping_block',
                    seed=seed, verbose=False,
                ),
            )
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'verify box={box_idx} {type(e).__name__}: {e}'}
        last_pbox = pbox

        # Empirical coverage on N_TEST samples (sanity diagnostic; not
        # used in verdict).
        try:
            cov, _sigma, n_eff = empirical_coverage_for_box(
                model_fn=model_fn,
                input_lb=input_set.lb, input_ub=input_set.ub,
                box_lb=pbox.lb, box_ub=pbox.ub,
                n_test=_N_TEST_COVERAGE,
                seed=seed,
            )
            if not np.isnan(cov):
                cov_vals.append(cov)
                cov_n_total += n_eff
        except Exception:
            pass

        disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
        if disjoint is True:
            continue
        any_unknown = True

    cov_emp = float(np.mean(cov_vals)) if cov_vals else float('nan')
    if sat:
        verdict = 'SAT'
    elif any_unknown:
        verdict = 'UNKNOWN'
    else:
        verdict = 'UNSAT'

    return {
        'verdict': verdict,
        'm': m, 'ell': ell, 'epsilon': _EPSILON,
        'coverage': (
            f'{last_pbox.coverage:.4f}' if last_pbox is not None else ''),
        'coverage_empirical': (
            f'{cov_emp:.4f}' if not np.isnan(cov_emp) else ''),
        'coverage_n_test': cov_n_total,
        'confidence': (
            f'{last_pbox.confidence:.4f}' if last_pbox is not None else ''),
        'cex_x': cex_x_str,
        'cex_y': cex_y_str,
        'error': '',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP1_BENCHMARKS)
    p.add_argument('--smoke', action='store_true',
                   help='Run only the first instance; assert the pipeline '
                        'completed without ERROR/TIMEOUT/SKIPPED. Reports '
                        'verdict against ground_truth.csv at end.')
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only the instance at this 0-based index '
                        'and APPEND to CSV (used by run_cell.sh).')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <vnncomp_timeout_s>" per instance, exit.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    instances = list_instances(benchmark)

    if args.list_instances:
        for idx, (_o, _v, vnncomp_t) in enumerate(instances):
            print(f'{idx} {vnncomp_t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp1_{benchmark}_hashemi_clipping.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --write-timeout-row: append a TIMEOUT row for instance at
    # --instance-idx whose actual run was killed by the outer shell
    # timeout (run_cell.sh exit 124). Required combo: --write-timeout-row
    # AND --instance-idx.
    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
        _write_timeout_row(out_csv, benchmark, onnx_rel, vnn_rel, vnncomp_t)
        return

    # Per-instance mode: append to CSV and run only the requested idx.
    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print(f'--instance-idx {args.instance_idx} out of range '
                  f'[0, {len(instances)})', file=sys.stderr)
            sys.exit(2)
        instances = [instances[args.instance_idx]]
        append_mode = True
        print(f'[{benchmark}] running only idx={args.instance_idx}; '
              f'appending to {out_csv}', flush=True)
    elif args.smoke:
        instances = instances[:1]
        print(f'[smoke] {benchmark}: running only the first instance',
              flush=True)

    # ---- Open CSV in correct mode (write fresh, or append if per-instance) ----
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    print(f'[{benchmark}] Loaded {len(instances)} instances; '
          f'writing to {out_csv}', flush=True)
    cfg_for_print = PER_BENCHMARK_CONFIG[benchmark]
    print_m = int(cfg_for_print.get('hashemi_m', _DEFAULT_M))
    print(f'[{benchmark}] Hashemi-clipping config: m={print_m} ell={print_m - 1} '
          f'epsilon={_EPSILON} SEED={_SEED}', flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'SKIPPED': 0, 'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
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
                    benchmark, onnx_rel, vnn_rel, seed=_SEED)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'onnx_file': Path(onnx_rel).name,
                'vnnlib_file': Path(vnn_rel).name,
                'wall_s': f'{wall_s:.1f}',
                'vnncomp_timeout_s': vnncomp_t,
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
        # Hashemi-clipping is *known* to be unsound at this m on some
        # benchmarks (FUR ~25-28%); a single-instance smoke isn't a
        # soundness validator. We just assert it ran without ERROR/TIMEOUT.
        bad = (counts.get('ERROR', 0) > 0 or counts.get('TIMEOUT', 0) > 0
               or counts.get('SKIPPED', 0) == 1)
        if bad:
            print(f'[smoke] FAIL on {benchmark}: '
                  f'expected a non-error verdict, got counts={counts}',
                  file=sys.stderr)
            sys.exit(1)
        # Note: we don't assert the verdict matches ground truth because
        # the FUR is part of what Exp 1 measures. Just report.
        from examples.FlowConformal.experiments._ground_truth_lookup import (
            lookup_ground_truth,
        )
        onnx_rel, vnn_rel, _ = instances[0]
        first_inst = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'
        gt = lookup_ground_truth('exp1', benchmark, first_inst)
        print(f'[smoke] PASS on {benchmark}: pipeline ran end-to-end '
              f'(VNN-COMP ground truth = {gt}; Hashemi verdict = '
              f'{[k for k,v in counts.items() if v]}).')


if __name__ == '__main__':
    main()
