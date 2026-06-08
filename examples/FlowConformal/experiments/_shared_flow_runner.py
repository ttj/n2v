"""Shared three-stage flow-reach helper for Exp 1–4 + ablation runners.

Thin wrapper over :func:`n2v.probabilistic.flow_reach` plus
:func:`n2v.utils.verify_specification.verify_specification` that returns
a dict with the key shape the existing ``exp{N}_run_ours.py`` aggregation
+ CSV-row construction code expects.

This consolidation keeps the 4+1 runners' algorithmic surface in one
place. Per-experiment tweaks (e.g. Exp 4 uses a different ``m``/``ell``
than the paper defaults) come through the ``cfg`` dict at call time.

Stages:
  1. Optional falsifier — short-circuits to SAT on a confirmed
     counterexample.
  2. Flow-matching reach via the model-agnostic ``flow_reach`` free
     function (bypasses ``NeuralNetwork``'s eager torch.fx trace, which
     would fail on models with data-dependent control flow such as the
     ACAS Xu wrapper).
  3. Spec verification via ``verify_specification(prob_set, spec, ...)``.

The held-out coverage_empirical diagnostic is computed via
:meth:`ProbabilisticSet.estimate_coverage` with the seed structure
``cal_seed + 2_000_000`` so it matches byte-for-byte for paper
reproducibility.
"""
from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from n2v.probabilistic import FlowReachConfig, flow_reach
from n2v.sets import Box
from n2v.utils.falsify import falsify
from n2v.utils.verify_specification import (
    ProbVerifyConfig,
    verify_specification,
)


def run_flow_pipeline(network, lb: np.ndarray, ub: np.ndarray, spec,
                      cfg: dict, *, seed: int) -> dict:
    """Three-stage pipeline against the new public API.

    Args:
        network: PyTorch ``nn.Module`` to verify.
        lb, ub: input region bounds (1-D numpy arrays).
        spec: VNN-LIB-shaped property (HalfSpace / list / dict / list[dict]).
        cfg: per-instance config dict. Required keys: ``alpha``,
            ``flow_config``, ``n_train``, ``flow_epochs``,
            ``scenario_n_samples``, ``verification_method``,
            ``amls_max_levels``. Optional keys with defaults:
            ``use_falsifier`` (False), ``falsifier_method`` ('apgd'),
            ``falsifier_n_restarts`` (10), ``falsifier_n_steps`` (100),
            ``m`` (8000), ``ell`` (m - 1), ``amls_bounded_eps_2_target``
            (None — required for amls_bounded / raw_mc_uniform methods).
        seed: master RNG seed; flow / calibration / verification
            stages all derive from it (matches legacy pipeline).

    Returns:
        Result dict whose keys match the legacy
        ``run_verification_pipeline`` output (the helper itself is
        gone; see module docstring), so existing runner-side
        aggregation + CSV-row construction keeps working unchanged.
    """
    t_pipeline_start = time.time()

    # ---- Stage 1: optional falsifier ----
    sat_backend_time = 0.0
    if cfg.get('use_falsifier'):
        falsifier_method = cfg.get('falsifier_method', 'apgd')
        falsifier_kwargs = {
            'n_restarts': cfg.get('falsifier_n_restarts', 10),
            'n_steps': cfg.get('falsifier_n_steps', 100),
        }
        t_sat = time.time()
        try:
            fals_int, fals_cex = falsify(
                model=network, lb=np.asarray(lb), ub=np.asarray(ub),
                property=spec, method=falsifier_method, seed=seed,
                **falsifier_kwargs,
            )
        except Exception as e:
            fals_int, fals_cex = 2, None
            print(f'[_shared_flow_runner] falsify({falsifier_method}) raised '
                  f'{type(e).__name__}: {e}', file=sys.stderr)
        sat_backend_time = time.time() - t_sat
        if fals_int == 0 and fals_cex is not None:
            cex_x, cex_y = fals_cex
            return {
                'verdict': 'SAT',
                'counterexample': {
                    'x': np.asarray(cex_x).flatten(),
                    'y': np.asarray(cex_y).flatten(),
                },
                'total_time_s': sat_backend_time,
                'flow_train_time_s': 0.0,
                'verification_time_s': 0.0,
                'epsilon_total': None, 'delta_total': None,
                'q': None, 'coverage_empirical': None,
                'amls_bounded_eps_2_target': None,
                'amls_bounded_eps_2_upper': None,
                'amls_bounded_detected_unsafe': None,
                'amls_levels_used': None,
                'prob_set': None,
            }

    # ---- Stage 2: flow reach ----
    # Use the model-agnostic free function directly. The OO surface
    # (``NeuralNetwork(network).reach(method='flow_matching')``) eagerly
    # torch.fx-traces the model in ``NeuralNetwork.__init__`` to inventory
    # its layers, which fails for models with data-dependent control
    # flow (e.g. the ACAS Xu wrapper's ``if x.dim() == 2:`` reshape
    # branch). ``flow_reach`` accepts any callable / nn.Module without
    # tracing, so this path supports the full benchmark suite.
    input_box = Box(np.asarray(lb), np.asarray(ub))
    m = int(cfg.get('m', 8000))
    ell = int(cfg.get('ell', m - 1))
    t_train = time.time()
    prob_set = flow_reach(
        network, input_box,
        FlowReachConfig(
            epsilon=cfg['alpha'],
            m=m, ell=ell,
            n_train=cfg['n_train'],
            flow_epochs=cfg['flow_epochs'],
            flow_config=cfg['flow_config'],
            seed=seed,
        ),
    )
    train_time = time.time() - t_train

    # Empirical coverage diagnostic on a fresh 2 000-sample test set.
    # Seeded with the same offset the legacy ``_calibrate_flow_for_spec``
    # uses (``cal_seed + 2_000_000``) for byte-exact parity.
    coverage = prob_set.estimate_coverage(
        network, input_box, n_test=2_000, seed=seed + 2_000_000,
    )

    # ---- Stage 3: verify_specification ----
    # AMLS-bounded / raw-MC methods require an eps_2 target; legacy
    # pipeline defaults this to ``alpha`` (the conformal miscoverage)
    # when the caller didn't set it explicitly. Preserve that for parity.
    method = cfg['verification_method']
    eps_2_target = cfg.get('amls_bounded_eps_2_target')
    if eps_2_target is None and method in (
            'amls_bounded', 'amls_bounded_union', 'raw_mc_uniform'):
        eps_2_target = cfg['alpha']
        print(
            f'[_shared_flow_runner] amls_bounded_eps_2_target unset for '
            f'method={method!r}; defaulting to alpha={eps_2_target}. '
            f'Set cfg["amls_bounded_eps_2_target"] explicitly to silence.',
            file=sys.stderr,
        )

    t_verify = time.time()
    result = verify_specification(
        prob_set, spec,
        config=ProbVerifyConfig(
            method=method,
            n_samples=cfg['scenario_n_samples'],
            beta=0.001,
            seed=seed,
            amls_max_levels=cfg['amls_max_levels'],
            amls_bounded_eps_2_target=eps_2_target,
        ),
    )
    verify_time = time.time() - t_verify
    # ``t_pipeline_start`` was captured BEFORE the optional Stage-1
    # falsifier, so ``time.time() - t_pipeline_start`` already covers the
    # falsifier window on the non-SAT-exit path. Do NOT add
    # ``sat_backend_time`` here — that double-counted falsifier wall and
    # inflated wall_s on every UNSAT-after-falsifier instance.
    total_time = time.time() - t_pipeline_start

    return {
        'verdict': result.verdict,
        'counterexample': None,
        'total_time_s': total_time,
        'flow_train_time_s': train_time,
        'verification_time_s': verify_time,
        'epsilon_total': result.epsilon_total,
        'delta_total': result.delta_total,
        'q': result.q,
        'coverage_empirical': coverage,
        'amls_bounded_eps_2_target': eps_2_target,
        'amls_bounded_eps_2_upper': result.amls_bounded_eps_2_upper,
        'amls_bounded_detected_unsafe': result.amls_bounded_detected_unsafe,
        'amls_levels_used': result.amls_levels_used,
        # Exposed for callers (Exp 3) that want the actual prob_set
        # downstream (volume estimation etc.). Other runners ignore it.
        'prob_set': prob_set,
    }
