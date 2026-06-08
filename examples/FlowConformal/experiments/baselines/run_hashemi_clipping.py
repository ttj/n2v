"""Hashemi-clipping-block probabilistic-verifier runner.

Same structure as ``run_hashemi_naive.py`` but uses the
``surrogate='clipping_block'`` mode, which fits a convex-hull surrogate
to the training outputs. Tighter bounds at the cost of more LP solves
during calibration.

Usage:
    cd /path/to/n2v
    python -u -m \\
        examples.FlowConformal.experiments.baselines.run_hashemi_clipping \\
        --benchmark <name> --smoke
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from examples.FlowConformal.experiments.baselines._common import (
    add_common_args, empirical_coverage_for_box, halfspace_disjoint_from_box,
    halfspace_witness_from_samples, load_benchmark_instances,
    resolve_n_instances, resolve_output_csv, run_baseline_sweep, torch_callable,
)


_BASELINE = 'hashemi_clipping'
_M = 8000
_ELL = 7999
_EPSILON = 0.001
_N_TEST_COVERAGE = 1000


def _process_factory(seed: int, *, m: int = _M, ell: int | None = None,
                     epsilon: float = _EPSILON):
    if ell is None:
        ell = m - 1
    _m = m
    _ell = ell
    _epsilon = epsilon
    from n2v.probabilistic import conformal_reach
    from n2v.sets import Box

    def process_one(loader, name):
        try:
            net, boxes, spec, _ = loader()
        except FileNotFoundError as e:
            return {'verdict': 'ERROR', 'error': f'load_missing: {e}'}
        except NotImplementedError as e:
            return {'verdict': 'ERROR', 'error': f'unsupported_spec: {e}'}
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'load {type(e).__name__}: {e}'}

        any_unknown = False
        cov_vals: list[float] = []
        cov_n_total = 0
        for (lb, ub) in boxes:
            input_set = Box(np.asarray(lb).flatten(),
                            np.asarray(ub).flatten())
            model_fn = torch_callable(net)
            try:
                pbox = conformal_reach(
                    model=model_fn,
                    input_box=input_set,
                    m=_m, ell=_ell,
                    epsilon=_epsilon,
                    surrogate='clipping_block',
                    seed=seed,
                    verbose=False,
                )
            except Exception as e:
                return {'verdict': 'ERROR',
                        'error': f'verify {type(e).__name__}: {e}'}

            # Empirical coverage on N_TEST_COVERAGE held-out samples
            # drawn uniformly from THIS input box.
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

            try:
                lb_samp = input_set.lb.flatten()
                ub_samp = input_set.ub.flatten()
                rng = np.random.default_rng(seed)
                xs = rng.uniform(lb_samp, ub_samp,
                                 size=(min(2048, max(1, _m // 4)),
                                       lb_samp.size)).astype(np.float32)
                ys = model_fn(xs)
                cex_idx = halfspace_witness_from_samples(spec, ys)
                if cex_idx is not None:
                    cov_emp = float(np.mean(cov_vals)) if cov_vals else float('nan')
                    return {
                        'verdict': 'SAT',
                        'm': _m, 'ell': _ell, 'epsilon': _epsilon,
                        'coverage': pbox.coverage,
                        'coverage_empirical': cov_emp,
                        'coverage_n_test': cov_n_total,
                        'confidence': pbox.confidence,
                        'error': '',
                    }
            except Exception:
                pass

            disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
            if disjoint is True:
                continue
            any_unknown = True

        cov_emp = float(np.mean(cov_vals)) if cov_vals else float('nan')
        if any_unknown:
            return {
                'verdict': 'UNKNOWN',
                'm': _m, 'ell': _ell, 'epsilon': _epsilon,
                'coverage': pbox.coverage,
                'coverage_empirical': cov_emp,
                'coverage_n_test': cov_n_total,
                'confidence': pbox.confidence,
                'error': '',
            }
        return {
            'verdict': 'UNSAT',
            'm': _M, 'ell': _ELL, 'epsilon': _EPSILON,
            'coverage': pbox.coverage,
            'coverage_empirical': cov_emp,
            'coverage_n_test': cov_n_total,
            'confidence': pbox.confidence,
            'error': '',
        }

    return process_one


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--m', type=int, default=_M,
                        help='Calibration set size m (default 8000).')
    parser.add_argument('--epsilon', type=float, default=_EPSILON,
                        help='Miscoverage level (default 1e-3).')
    args = parser.parse_args()

    n = resolve_n_instances(args)
    try:
        instances = load_benchmark_instances(args.benchmark, n)
    except FileNotFoundError as e:
        print(f'[{_BASELINE}] TODO/load failed: {e}', file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f'[{_BASELINE}] load error: {type(e).__name__}: {e}',
              file=sys.stderr)
        sys.exit(2)
    if not instances:
        print(f'[{_BASELINE}] no instances', file=sys.stderr)
        sys.exit(0)

    out_csv = resolve_output_csv(args, _BASELINE)
    extra_fields = ['m', 'ell', 'epsilon', 'coverage',
                    'coverage_empirical', 'coverage_n_test', 'confidence']
    run_baseline_sweep(
        baseline=_BASELINE, benchmark=args.benchmark,
        instances=instances, out_csv=out_csv,
        extra_fields=extra_fields,
        process_one=_process_factory(args.seed, m=args.m, epsilon=args.epsilon),
    )


if __name__ == '__main__':
    main()
