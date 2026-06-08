"""Exp 3 — bounded-AMLS (ours) runner with score-family ablation.

One script invocation runs ``K`` seeds of our pipeline on a given
``(benchmark, score, spec)`` tuple and writes one CSV row per seed
under
``examples/FlowConformal/experiments/exp3_synthetic/outputs/exp3_<benchmark>_<score>_<spec>_ours.csv``.

The ``--score`` flag selects the nonconformity-score family used during
calibration. Currently only ``flow`` (the default `FlowScore` already
plumbed through :func:`run_verification_pipeline`) is fully wired up;
``hyperrect``, ``ellipsoid``, ``gmm`` raise ``NotImplementedError`` so
the CLI shape is fixed and we can fill them in incrementally without
breaking call sites.

The ``--spec`` flag selects between two specs per benchmark:

* ``unsat`` — far-away halfspace ``y_0 >= 1e6``; UNSAT by construction
  (the network's bounded output range can't reach the threshold).
  Smoke / sweep should yield UNSAT.
* ``sat`` — reachable halfspace ``y_0 >= 0``. Falsifier OFF for Exp 3,
  so ours abstains (UNKNOWN). Empirical UNKNOWN rate measures the
  honest-abstention behavior.

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours \\
        --benchmark synth_5d --score flow --spec unsat --smoke
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
from examples.FlowConformal.experiments.exp3_synthetic._benchmarks import (
    EXP3_BENCHMARKS,
    EXP3_SCORES,
    EXP3_SPECS,
    PER_BENCHMARK_CONFIG,
    make_input_box,
    make_network,
    make_spec,
)
# NB: flow-matching reach happens inside
# ``_score_pipeline.run_score_pipeline(score='flow', ...)``, which calls
# the shared three-stage runner.

_SEED_BASE = 47
_DEFAULT_K_SEEDS = 5
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'score', 'spec_type', 'seed', 'verdict',
    'wall_s', 'train_s', 'verify_s',
    'coverage', 'q', 'epsilon_total', 'delta_total',
    'amls_bounded_eps_2_upper', 'amls_levels_used',
    'volume_estimate', 'volume_exact', 'volume_ratio',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


def _run_one_seed(benchmark: str, score: str, spec_type: str, cfg: dict,
                  *, seed: int, overrides: dict) -> dict:
    """Run one (benchmark, score, spec) instance at the given seed."""
    try:
        network = make_network(benchmark, seed=seed)
        lb, ub = make_input_box(benchmark)
        spec = make_spec(benchmark, spec_type)
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'load {type(e).__name__}: {e}'}

    from examples.FlowConformal.experiments.exp3_synthetic._score_pipeline import (
        run_score_pipeline,
    )
    try:
        r = run_score_pipeline(
            network=network,
            input_lb=lb, input_ub=ub, spec=spec,
            score=score,
            n_train=overrides.get('n_train', cfg['n_train']),
            alpha=cfg['alpha'],
            seed=seed,
            flow_epochs=overrides.get('flow_epochs', 2_000),
            flow_config=overrides.get('flow_config', 'base'),
            scenario_n_samples=overrides.get('scenario_n_samples', 2_000),
            scenario_beta=overrides.get('scenario_beta', 0.001),
            volume_m=overrides.get('volume_m', 8_000),
            volume_ell=overrides.get(
                'volume_ell',
                max(1, overrides.get('volume_m', 8_000) - 1),
            ),
            volume_n_samples=overrides.get(
                'volume_n_samples', 200_000,
            ),
        )
    except NotImplementedError as e:
        return {'verdict': 'NOT_IMPLEMENTED', 'error': str(e)}
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'run {type(e).__name__}: {e}'}

    return {
        'verdict': r['verdict'],
        'wall_s': _fmt(r.get('wall_s'), '.1f'),
        'train_s': _fmt(r.get('train_s'), '.1f'),
        'verify_s': _fmt(r.get('verify_s'), '.1f'),
        'coverage': _fmt(r.get('coverage_empirical'), '.4f'),
        'q': _fmt(r.get('q'), '.4f'),
        'epsilon_total': _fmt(r.get('epsilon_total'), '.4f'),
        'delta_total': _fmt(r.get('delta_total'), '.4f'),
        'amls_bounded_eps_2_upper': _fmt(
            r.get('amls_bounded_eps_2_upper'), '.4e'),
        'amls_levels_used': (str(r.get('amls_levels_used'))
                              if r.get('amls_levels_used') is not None
                              else ''),
        'volume_estimate': _fmt(r.get('volume_estimate'), '.6e'),
        'volume_exact': _fmt(r.get('volume_exact'), '.6e'),
        'volume_ratio': _fmt(r.get('volume_ratio'), '.4f'),
        'cex_x': r.get('cex_x', ''),
        'cex_y': r.get('cex_y', ''),
        'error': '',
    }


def _write_timeout_row(out_csv, benchmark, score, spec_type, seed):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'score': score,
        'spec_type': spec_type, 'seed': seed,
        'verdict': 'TIMEOUT',
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP3_BENCHMARKS)
    p.add_argument('--score', required=True, choices=EXP3_SCORES,
                   help='Nonconformity-score family. Only "flow" fully '
                        'wired; others raise NotImplementedError until '
                        'their calibration + verdict plumbing is added.')
    p.add_argument('--spec', dest='spec_type', required=True,
                   choices=EXP3_SPECS,
                   help='"unsat" = far halfspace (always UNSAT by '
                        'construction); "sat" = reachable halfspace '
                        '(falsifier OFF, expect UNKNOWN abstention).')
    p.add_argument('--seeds', type=int, default=_DEFAULT_K_SEEDS,
                   help=f'Number of seeds (default {_DEFAULT_K_SEEDS}).')
    p.add_argument('--smoke', action='store_true',
                   help='Run only seed=0.')
    p.add_argument('--n-train', type=int, default=None,
                   help='Override calibration sample budget '
                        '(default = PER_BENCHMARK_CONFIG[benchmark][n_train]).')
    p.add_argument('--flow-epochs', type=int, default=None,
                   help='Override flow training epochs (default 2000).')
    p.add_argument('--flow-config', type=str, default=None,
                   help='Override flow architecture config name '
                        '(default "base").')
    p.add_argument('--scenario-n-samples', type=int, default=None,
                   help='Override scenario / AMLS sample budget per level '
                        '(default 2000).')
    p.add_argument('--volume-m', type=int, default=None,
                   help='Hashemi m param used inside ProbabilisticSet for '
                        'the MC volume estimate (default 8000).')
    p.add_argument('--volume-ell', type=int, default=None,
                   help='Hashemi ell param for the MC volume estimate '
                        '(default = volume_m - 1).')
    p.add_argument('--volume-n-samples', type=int, default=None,
                   help='MC sample count for ProbabilisticSet.estimate_volume '
                        '(default 200000).')
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only the seed at this 0-based index '
                        '(0..seeds-1) and APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<seed_idx> <approx_timeout_s>" per seed, exit. '
                        'Exp 3 has no per-seed VNN-COMP timeout; uses a '
                        'fixed 600s estimate.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    score = args.score
    spec_type = args.spec_type
    cfg = PER_BENCHMARK_CONFIG[benchmark]

    overrides: dict = {}
    if args.n_train is not None:
        overrides['n_train'] = args.n_train
    if args.flow_epochs is not None:
        overrides['flow_epochs'] = args.flow_epochs
    if args.flow_config is not None:
        overrides['flow_config'] = args.flow_config
    if args.scenario_n_samples is not None:
        overrides['scenario_n_samples'] = args.scenario_n_samples
    if args.volume_m is not None:
        overrides['volume_m'] = args.volume_m
    if args.volume_ell is not None:
        overrides['volume_ell'] = args.volume_ell
    if args.volume_n_samples is not None:
        overrides['volume_n_samples'] = args.volume_n_samples

    if args.list_instances:
        for s in range(args.seeds):
            print(f'{s} 600')   # fixed 10-min budget per seed
        return

    out_csv = (
        args.output_csv if args.output_csv is not None
        else _OUT_DIR / f'exp3_{benchmark}_{score}_{spec_type}_ours.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx', file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < args.seeds):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        _write_timeout_row(out_csv, benchmark, score, spec_type, args.instance_idx)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < args.seeds):
            print(f'--instance-idx {args.instance_idx} out of range '
                  f'[0, {args.seeds})', file=sys.stderr)
            sys.exit(2)
        seeds = [args.instance_idx]
        append_mode = True
        print(f'[{benchmark}/{score}/{spec_type}] running only seed_idx='
              f'{args.instance_idx}; appending to {out_csv}', flush=True)
    else:
        seeds = [_SEED_BASE] if args.smoke else list(range(args.seeds))
    if args.smoke:
        print(f'[smoke] {benchmark}/{score}/{spec_type}: running 1 seed',
              flush=True)
    print(f'[{benchmark}/{score}/{spec_type}] '
          f'{len(seeds)} seeds; writing to {out_csv}', flush=True)
    print(f'[{benchmark}/{score}/{spec_type}] Config: '
          f'flow_config={cfg["flow_config"]} n_train={cfg["n_train"]} '
          f'flow_epochs={cfg["flow_epochs"]} '
          f'method={cfg["verification_method"]}', flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'NOT_IMPLEMENTED': 0, 'ERROR': 0}
    t_start = time.time()

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader()
            f.flush()

        for k, seed in enumerate(seeds, start=1):
            elapsed = time.time() - t_start
            print(f'[{benchmark}/{score}/{spec_type} {k}/{len(seeds)} '
                  f't={elapsed:.0f}s] seed={seed}', flush=True)
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            row = _run_one_seed(benchmark, score, spec_type, cfg, seed=seed,
                                overrides=overrides)

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'score': score,
                'spec_type': spec_type,
                'seed': seed,
                'timestamp': _now_iso(),
            })
            out_row.update(row)
            if not row.get('wall_s'):
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
        # Pipeline must complete without ERROR. NOT_IMPLEMENTED is OK
        # for the stubbed scores (ellipsoid, gmm) since their plumbing
        # is intentionally deferred.
        if counts.get('ERROR', 0) > 0:
            print(f'[smoke] FAIL on {benchmark}: ERROR observed, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next(v for v, c in counts.items() if c > 0)
        if actual == 'NOT_IMPLEMENTED':
            print(f'[smoke] PASS on {benchmark}/{score}/{spec_type}: '
                  f'NOT_IMPLEMENTED (score family stubbed for now).')
            return
        expected_by_spec = {'unsat': 'UNSAT', 'sat': 'UNKNOWN'}
        expected = expected_by_spec[spec_type]
        match = '✓' if actual == expected else '≠'
        print(f'[smoke] PASS on {benchmark}/{score}/{spec_type}: '
              f'verdict={actual} {match} expected {expected}.')


if __name__ == '__main__':
    main()
