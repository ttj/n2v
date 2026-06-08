"""Exp 3 — sound star-set approx-reach baseline.

Mirrors :mod:`exp3_run_hashemi_clipping` but uses the n2v API
``reach_pytorch_model(model, Star, method='approx')`` to obtain a
sound over-approximate output Star, then reports its axis-aligned
bounding box volume as the predicted reach-set volume.

This baseline is *deterministic* and *sound* — its predicted volume
is a guaranteed upper bound on the true reach-set volume. Comparison
against ours / hashemi clipping shows the gap between (sound, fast,
no calibration) and (probabilistic, calibration-tuned) reach-set
prediction.

Per-instance the runner records the closed-form / cached exact
reach-set volume (when available) and the resulting ratio
``starset_bbox_vol / volume_exact``.

Usage::

    python -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_run_starset_approx \\
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
from examples.FlowConformal.experiments.baselines._common import (
    halfspace_disjoint_from_box,
)
from n2v.nn.reach import reach_pytorch_model
from n2v.sets.box import Box
from n2v.sets.star import Star

_SEED_BASE = 47
_DEFAULT_K_SEEDS = 5
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'spec_type', 'seed', 'verdict', 'wall_s',
    'n_output_sets',
    'starset_bbox_vol', 'volume_exact', 'volume_ratio',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


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


def _star_from_box(lb: np.ndarray, ub: np.ndarray) -> Star:
    """Build a unit-cube-parameterised :class:`Star` from a bbox.

    Star representation: ``x = V[:, 0] + V[:, 1:] @ alpha`` with
    ``C alpha <= d`` and ``-1 <= alpha <= 1``. Parameterising with a
    unit-cube alpha lets approx reachability propagate the polytope
    structure through linear + ReLU layers cleanly.
    """
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    n = lb.size
    center = ((lb + ub) / 2.0).reshape(-1, 1)
    V_basis = np.diag((ub - lb) / 2.0)
    V = np.hstack([center, V_basis])
    C = np.vstack([np.eye(n), -np.eye(n)])
    d = np.ones(2 * n)
    return Star(V=V, C=C, d=d)


def _exact_reach_volume(benchmark: str, network, lb, ub) -> float:
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


def _output_set_bbox(output_set):
    """Return ``(lb, ub)`` bbox arrays for one output Star/Box/Zono."""
    if hasattr(output_set, 'get_box'):
        b = output_set.get_box()
        return (np.asarray(b.lb).flatten(),
                np.asarray(b.ub).flatten())
    return (np.asarray(output_set.lb).flatten(),
            np.asarray(output_set.ub).flatten())


def _bbox_union_volume(out_sets) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume of the smallest bbox enclosing *all* output sets.

    For ``approx`` reachability of identity-activation nets, ``len(out_sets)
    == 1``. For ReLU-style nets the splitter may return multiple Stars;
    take the bbox enclosing every Star and report ``prod(ub - lb)``.
    """
    bbox_lbs, bbox_ubs = [], []
    for s in out_sets:
        lb, ub = _output_set_bbox(s)
        bbox_lbs.append(lb); bbox_ubs.append(ub)
    bbox_lbs = np.stack(bbox_lbs, axis=0)
    bbox_ubs = np.stack(bbox_ubs, axis=0)
    union_lb = bbox_lbs.min(axis=0)
    union_ub = bbox_ubs.max(axis=0)
    return float(np.prod(union_ub - union_lb)), union_lb, union_ub


def _run_one_seed(benchmark: str, spec_type: str, *,
                  seed: int) -> Dict[str, Any]:
    try:
        network = make_network(benchmark, seed=seed)
        lb, ub = make_input_box(benchmark)
        spec = make_spec(benchmark, spec_type)
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'load {type(e).__name__}: {e}'}

    star = _star_from_box(np.asarray(lb).flatten(),
                          np.asarray(ub).flatten())
    try:
        out_sets = reach_pytorch_model(network, star, method='approx')
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'reach {type(e).__name__}: {e}'}

    if not out_sets:
        return {'verdict': 'ERROR',
                'error': 'reach returned empty set list'}

    try:
        starset_vol, union_lb, union_ub = _bbox_union_volume(out_sets)
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'bbox {type(e).__name__}: {e}'}

    # Verdict via the same bbox-vs-halfspace disjointness check the
    # hashemi runner uses. The starset bbox is a sound over-approx so
    # disjoint => the spec is genuinely UNSAT (no probabilistic gap).
    cex_x_str = ''
    cex_y_str = ''
    disjoint = halfspace_disjoint_from_box(spec, union_lb, union_ub)
    verdict = 'UNSAT' if disjoint is True else 'UNKNOWN'

    vol_exact = _exact_reach_volume(benchmark, network, lb, ub)
    if np.isfinite(vol_exact) and vol_exact > 0:
        vol_ratio = starset_vol / vol_exact
    else:
        vol_ratio = float('nan')

    return {
        'verdict': verdict,
        'n_output_sets': len(out_sets),
        'starset_bbox_vol': _fmt_e(starset_vol),
        'volume_exact': _fmt_e(vol_exact),
        'volume_ratio': _fmt_f(vol_ratio),
        'cex_x': cex_x_str, 'cex_y': cex_y_str,
        'error': '',
    }


def _write_timeout_row(out_csv, benchmark, spec_type, seed):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'spec_type': spec_type,
        'seed': seed, 'verdict': 'TIMEOUT',
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True, choices=EXP3_BENCHMARKS)
    p.add_argument('--spec', dest='spec_type', required=True,
                   choices=EXP3_SPECS)
    p.add_argument('--seeds', type=int, default=_DEFAULT_K_SEEDS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--instance-idx', type=int, default=None)
    p.add_argument('--list-instances', action='store_true')
    p.add_argument('--write-timeout-row', action='store_true')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    spec_type = args.spec_type

    if args.list_instances:
        for s in range(args.seeds):
            print(f'{s} 600')
        return

    out_csv = (
        args.output_csv if args.output_csv is not None
        else _OUT_DIR / f'exp3_{benchmark}_{spec_type}_starset_approx.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx', file=sys.stderr)
            sys.exit(2)
        _write_timeout_row(out_csv, benchmark, spec_type, args.instance_idx)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < args.seeds):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        seeds = [args.instance_idx]
        append_mode = True
    else:
        seeds = [_SEED_BASE] if args.smoke else list(range(args.seeds))

    print(f'[starset-approx {benchmark}/{spec_type}] {len(seeds)} seed(s); '
          f'writing to {out_csv}')

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
            row = _run_one_seed(benchmark, spec_type, seed=seed)

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'spec_type': spec_type,
                'seed': seed,
                'wall_s': f'{wall_s:.2f}',
                'timestamp': _now_iso(),
            })
            out_row.update(row)
            writer.writerow(out_row); f.flush()
            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  wall={wall_s:.2f}s  '
                  f'starset_vol={row.get("starset_bbox_vol","")}  '
                  f'ratio={row.get("volume_ratio","")}', flush=True)

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
        match = '✓' if actual == expected_by_spec[spec_type] else '≠'
        print(f'[smoke] PASS on {benchmark}/{spec_type}: '
              f'verdict={actual} {match} expected '
              f'{expected_by_spec[spec_type]}.')


if __name__ == '__main__':
    main()
