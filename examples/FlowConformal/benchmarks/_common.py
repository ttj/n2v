"""Shared helpers for the FlowConformal benchmark scripts.

This module exposes two helpers used outside this file:

* :func:`exact_star_union_volume` — Star-union ground-truth volume of a
  network's reach set on an L-inf input ball. Used by
  ``exp3_synthetic/exact_volumes.py`` to compute exact volume references.

* :func:`run_verification_pipeline` — backward-compat wrapper around the
  new three-stage flow-conformal verification API. Used by
  ``demo_acasxu_single.py`` and the ``exp1_vnncomp_subset`` /
  ``exp2_prob_scale`` per-experiment helpers. New code should call the
  library API directly (``NeuralNetwork.reach(method='flow_matching')``
  + ``verify_specification(...)``).

For analytical-ground-truth benchmarks (identity, rotated linear) see
the sibling ``_common_analytical.py``.
"""
from __future__ import annotations

import numpy as np

from examples.FlowConformal.utils import compute_exact_reach
from n2v.sets.volume import (
    compute_mc_bbox, exact_volume_2d, star_union_volume_mc,
)


def exact_star_union_volume(net, x_center: np.ndarray, radius: float,
                            output_dim: int, n_mc: int = 500_000,
                            seed: int = 42) -> tuple[float, list]:
    """Star-union ground-truth volume (an exact deterministic over-approx of
    f_#P_X's support; the 1-alpha reachset is smaller by (1-alpha)).

    Returns (volume_mean, stars). The MC estimate is used because the Star
    union can have thousands of overlapping polytopes whose exact volume
    requires inclusion-exclusion.
    """
    reach = compute_exact_reach(net, x_center, radius, output_dim=output_dim)
    stars = reach['stars']
    if output_dim == 2:
        # 2D has a cheap rasterization method, which we use as the ground-
        # truth reference rather than MC on a box (the 2D Star union is a
        # measure-zero manifold in some cases, so MC-on-a-box would give 0).
        y_bbox = compute_mc_bbox(net, x_center, radius, output_dim=output_dim,
                                 n_samples=5000, pad=1.0)
        vol = exact_volume_2d(stars, (y_bbox[0].numpy(), y_bbox[1].numpy()),
                              resolution=500)
        return float(vol), stars
    ve = star_union_volume_mc(
        stars, n_samples=n_mc, batch_size=25_000, seed=seed,
        contains_method='algebraic',
    )
    return float(ve.mean), stars


# --- Verification pipeline shim ---------------------------------------
#
# Thin backward-compat wrapper around the three-stage flow-conformal +
# AMLS pipeline (``net.reach(method='flow_matching')`` +
# ``verify_specification(...)``). Defaults ``use_falsifier=True`` so the
# legacy example scripts keep producing the same falsifier-on-by-default
# behavior they were built against. New code should prefer the n2v
# library API directly.


def run_verification_pipeline(
    network,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec,
    *,
    use_falsifier: bool = True,  # legacy default (Stage-1 falsifier ON)
    alpha: float = 0.001,
    n_train: int = 10_000,
    flow_epochs: int = 5000,
    flow_config: str = 'tight',
    scenario_n_samples: int = 10_000,
    verification_method: str = 'scenario',
    amls_max_levels: int = 30,
    amls_bounded_eps_2_target: float | None = None,
    seed: int = 0,
    falsifier_method: str = 'apgd',
    falsifier_n_restarts: int = 10,
    falsifier_n_steps: int = 100,
    **_unused_kwargs,  # absorb any legacy-only kwargs gracefully
) -> dict:
    """Backward-compat shim that delegates to the new three-stage API.

    Bridges existing demo scripts (e.g. ``demo_acasxu_single.py``) to
    the post-refactor public API: ``NeuralNetwork.reach(method='flow_matching')``
    + ``verify_specification(...)``. Re-attaches the legacy-only result
    keys (``spec_summary``, ``epsilon_1``/``delta_1``,
    ``epsilon_2``/``delta_2``) by reading from the returned
    ``ProbabilisticSet``, so callers that printed those fields keep
    working unchanged.

    Defaults ``use_falsifier=True`` to preserve the falsifier-on-by-
    default behavior these scripts were built against. (The new public
    API ``net.reach(method='flow_matching')`` does NOT run a falsifier;
    callers wire it explicitly via :func:`n2v.utils.falsify.falsify`
    before the reach call.)
    """
    from examples.FlowConformal.experiments._shared_flow_runner import (
        run_flow_pipeline,
    )
    from n2v.utils.verify_specification import spec_summary

    cfg = dict(
        alpha=alpha,
        n_train=n_train,
        flow_epochs=flow_epochs,
        flow_config=flow_config,
        scenario_n_samples=scenario_n_samples,
        verification_method=verification_method,
        amls_max_levels=amls_max_levels,
        amls_bounded_eps_2_target=amls_bounded_eps_2_target,
        use_falsifier=use_falsifier,
        falsifier_method=falsifier_method,
        falsifier_n_restarts=falsifier_n_restarts,
        falsifier_n_steps=falsifier_n_steps,
    )
    result = run_flow_pipeline(
        network,
        np.asarray(input_lb).flatten(),
        np.asarray(input_ub).flatten(),
        spec, cfg, seed=seed,
    )

    # Legacy-compat: re-attach fields the old result dict carried.
    result['spec_summary'] = spec_summary(spec)
    pset = result.get('prob_set')
    if pset is not None:
        # Conformal layer guarantee.
        result['epsilon_1'] = pset.epsilon
        result['delta_1'] = 1.0 - pset.confidence
        # Verification layer: scenario_beta default in the shared helper
        # is 0.001 (matches paper config); recover delta_2.
        result['delta_2'] = 0.999
        # epsilon_2 is method-dependent and not always populated; derive
        # from joint if both layers' epsilon are known.
        if result.get('epsilon_total') is not None:
            eps_total = result['epsilon_total']
            eps_1 = pset.epsilon
            # epsilon_total = 1 - (1 - eps_1)(1 - eps_2) → solve for eps_2.
            try:
                result['epsilon_2'] = 1.0 - (1.0 - eps_total) / (1.0 - eps_1)
            except ZeroDivisionError:
                result['epsilon_2'] = None
        else:
            result['epsilon_2'] = None
    else:
        # SAT short-circuit path: no flow trained, no per-layer guarantees.
        result['epsilon_1'] = None
        result['delta_1'] = None
        result['epsilon_2'] = None
        result['delta_2'] = None

    return result
