"""Exp 3 — Hashemi-clipping runner on synthetic benchmarks.

Mirrors Exp 1/2/4 Hashemi-clipping runners. Synthetic networks are
constructed in-Python (no ONNX/vnnlib) so we use the
:func:`n2v.probabilistic.conformal_reach` API directly.

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_run_hashemi_clipping \\
        --benchmark synth_5d --spec unsat --smoke
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
from examples.FlowConformal.experiments.exp3_synthetic._benchmarks import (
    EXP3_BENCHMARKS,
    EXP3_SPECS,
    make_input_box,
    make_network,
    make_spec,
)
from examples.FlowConformal.experiments.exp3_synthetic.exact_volumes import (
    exact_volume_linear_net,
    exact_volume_three_blob_3d,
    exact_volume_two_banana,
)
from n2v.nn import NeuralNetwork
from n2v.nn.reach import ConformalReachConfig
from n2v.sets import Box

_SEED_BASE = 47
_M = 8000
_EPSILON = 0.001
_ELL = _M - 1
_DEFAULT_K_SEEDS = 5
# MC sample budget for the volume sanity check. 200K matches the
# ours-side ProbabilisticSet.estimate_volume call.
_VOL_MC_SAMPLES = 200_000
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'spec_type', 'seed', 'verdict', 'wall_s',
    'm', 'ell', 'epsilon', 'coverage', 'confidence',
    # Closed-form pbox volume = product of (ub - lb). Exact for axis-
    # aligned boxes; the MC column is a sanity check (sample uniformly
    # from a 1.1x expanded bbox, count fraction inside pbox, multiply
    # by expanded-bbox volume — must match the closed form modulo MC
    # noise).
    'volume_estimate_closedform', 'volume_estimate_mc',
    'volume_exact',
    'volume_ratio_closedform', 'volume_ratio_mc',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _mc_pbox_volume(
    pbox_lb: np.ndarray, pbox_ub: np.ndarray, *, seed: int,
    n_samples: int = _VOL_MC_SAMPLES,
) -> float:
    """MC sanity-check volume of an axis-aligned pbox.

    Sample uniformly from a 1.1× padded bounding box around pbox,
    count fraction inside pbox, multiply by the padded-bbox volume.
    Trivially must match ``prod(pbox_ub - pbox_lb)`` modulo MC noise.
    Useful only as a smoke check that the box accounting is right.
    """
    side = pbox_ub - pbox_lb
    if not np.all(side > 0):
        return float('nan')
    pad = 0.05 * side
    bbox_lb = pbox_lb - pad
    bbox_ub = pbox_ub + pad
    rng = np.random.default_rng(int(seed) + 9_973)
    x = rng.uniform(low=bbox_lb, high=bbox_ub,
                    size=(n_samples, pbox_lb.size))
    inside = np.all((x >= pbox_lb) & (x <= pbox_ub), axis=1)
    bbox_vol = float(np.prod(bbox_ub - bbox_lb))
    return float(inside.mean() * bbox_vol)


def _fmt_e(v) -> str:
    if v is None:
        return ''
    try:
        if not np.isfinite(v):
            return ''
    except TypeError:
        return ''
    return f'{v:.6e}'


def _fmt_f(v) -> str:
    if v is None:
        return ''
    try:
        if not np.isfinite(v):
            return ''
    except TypeError:
        return ''
    return f'{v:.4f}'


def _exact_reach_volume(benchmark: str, network, lb, ub) -> float:
    """Per-benchmark closed-form / cached reach-set volume V_R.

    Returns the FULL reach-set volume (no ``(1 - alpha)`` shrinkage)
    so volume_ratio = (predicted reach volume) / V_R has the natural
    "how much over-approximation" reading. Mirrors ours' convention in
    :func:`_score_pipeline._exact_volume_lipschitz`.

    Returns nan when no ground truth is wired up (an unfamiliar
    nonlinear classifier with no MC cache).
    """
    if benchmark == '3d_banana':
        return exact_volume_three_blob_3d(alpha=0.0)
    if benchmark == '2d_banana':
        return exact_volume_two_banana(alpha=0.0)
    if (hasattr(network, 'total_weight')
            and getattr(network, 'activation_name', None) == 'identity'):
        try:
            return exact_volume_linear_net(
                network,
                np.asarray(lb).flatten(), np.asarray(ub).flatten(),
                alpha=0.0,
            )
        except (ValueError, TypeError, AttributeError):
            return float('nan')
    return float('nan')


def _run_one_seed(benchmark: str, spec_type: str, *,
                  seed: int, m: int, ell: int) -> Dict[str, Any]:
    try:
        network = make_network(benchmark, seed=seed)
        lb, ub = make_input_box(benchmark)
        spec = make_spec(benchmark, spec_type)
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'load {type(e).__name__}: {e}'}

    input_set = Box(np.asarray(lb).flatten(), np.asarray(ub).flatten())
    net = NeuralNetwork(network)
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
                'error': f'verify {type(e).__name__}: {e}'}

    # Closed-form hashemi-clipping verdict via box-vs-halfspace
    # disjointness. Per Exp 3's "no falsifier" design we do NOT run a
    # sample-based unsafe-witness check here — that's handled by the
    # production Exp 1/2 Hashemi runners (which gate it on use_falsifier)
    # and by the gate-FUR study runner.
    cex_x_str = ''
    cex_y_str = ''
    disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
    verdict = 'UNSAT' if disjoint is True else 'UNKNOWN'

    pbox_lb = np.asarray(pbox.lb, dtype=np.float64).flatten()
    pbox_ub = np.asarray(pbox.ub, dtype=np.float64).flatten()
    vol_closed = float(np.prod(pbox_ub - pbox_lb))
    vol_mc = _mc_pbox_volume(pbox_lb, pbox_ub, seed=seed)
    vol_exact = _exact_reach_volume(benchmark, network, lb, ub)
    if np.isfinite(vol_exact) and vol_exact > 0:
        vol_ratio_closed = vol_closed / vol_exact
        vol_ratio_mc = (
            vol_mc / vol_exact if np.isfinite(vol_mc) else float('nan')
        )
    else:
        vol_ratio_closed = float('nan')
        vol_ratio_mc = float('nan')

    return {
        'verdict': verdict,
        'm': m, 'ell': ell, 'epsilon': _EPSILON,
        'coverage': f'{pbox.coverage:.4f}' if pbox is not None else '',
        'confidence': f'{pbox.confidence:.4f}' if pbox is not None else '',
        'volume_estimate_closedform': _fmt_e(vol_closed),
        'volume_estimate_mc': _fmt_e(vol_mc),
        'volume_exact': _fmt_e(vol_exact),
        'volume_ratio_closedform': _fmt_f(vol_ratio_closed),
        'volume_ratio_mc': _fmt_f(vol_ratio_mc),
        'cex_x': cex_x_str, 'cex_y': cex_y_str, 'error': '',
    }


def _write_timeout_row(out_csv, benchmark, spec_type, seed):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'spec_type': spec_type,
        'seed': seed, 'verdict': 'TIMEOUT',
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP3_BENCHMARKS)
    p.add_argument('--spec', dest='spec_type', required=True,
                   choices=EXP3_SPECS)
    p.add_argument('--seeds', type=int, default=_DEFAULT_K_SEEDS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--m', type=int, default=_M,
                   help=f'Total Hashemi sample budget (default {_M}).')
    p.add_argument('--ell', type=int, default=None,
                   help='Order-statistic index. Default = m - 1 (the '
                        'maximum), matching the production setting.')
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only seed at this 0-based index; APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<seed_idx> <timeout_s>" per seed, exit.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    spec_type = args.spec_type
    m = args.m
    ell = args.ell if args.ell is not None else max(1, m - 1)
    if not (1 <= ell <= m):
        print(f'--ell {ell} out of range [1, {m}]', file=sys.stderr)
        sys.exit(2)

    if args.list_instances:
        for s in range(args.seeds):
            print(f'{s} 600')
        return

    out_csv = (
        args.output_csv if args.output_csv is not None
        else _OUT_DIR / f'exp3_{benchmark}_{spec_type}_hashemi_clipping.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx', file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < args.seeds):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        _write_timeout_row(out_csv, benchmark, spec_type, args.instance_idx)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < args.seeds):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        seeds = [args.instance_idx]
        append_mode = True
        print(f'[{benchmark}/{spec_type}] running only seed_idx='
              f'{args.instance_idx}; appending to {out_csv}', flush=True)
    else:
        seeds = [_SEED_BASE] if args.smoke else list(range(args.seeds))
    if args.smoke:
        print(f'[smoke] {benchmark}/{spec_type}: running 1 seed', flush=True)
    print(f'[{benchmark}/{spec_type}] {len(seeds)} seeds; '
          f'writing to {out_csv}', flush=True)
    print(f'[{benchmark}/{spec_type}] Hashemi-clipping config: m={m} '
          f'ell={ell} epsilon={_EPSILON}', flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0, 'ERROR': 0}
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
            print(f'[{benchmark}/{spec_type} {k}/{len(seeds)} '
                  f't={elapsed:.0f}s] seed={seed}', flush=True)
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            row = _run_one_seed(benchmark, spec_type, seed=seed,
                                m=m, ell=ell)

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'spec_type': spec_type,
                'seed': seed,
                'wall_s': f'{wall_s:.1f}',
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
        if counts.get('ERROR', 0) > 0:
            print(f'[smoke] FAIL on {benchmark}/{spec_type}: ERROR observed, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next(v for v, c in counts.items() if c > 0)
        expected_by_spec = {'unsat': 'UNSAT', 'sat': 'UNKNOWN'}
        print(f'[smoke] PASS on {benchmark}/{spec_type}: '
              f'verdict={actual} (expected {expected_by_spec[spec_type]}).')


if __name__ == '__main__':
    main()
