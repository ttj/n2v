"""Per-(experiment, tool) runner shim for Exp 3.

Exp 3 covers synthetic benchmarks where the *exact* (or near-exact)
reach-set volume is known, so we can compare each method's predicted
reach set to ground truth on geometry, not just on verdict.

Supported benchmarks:

* ``3d_banana`` — :class:`examples.FlowConformal.networks.ThreeBlobClassifier3D`
  (3D input ``[-1, 1]^3``, 3D output logits with three multimodal blobs
  + curved separators). Exact reach-set volume from the cached Star-
  union (~213.72; (1-α)·V is the tightness floor).
* ``synth_5d`` / ``synth_10d`` / ``synth_20d`` — identity-activation
  1-Lipschitz nets where the composed map is purely linear and the
  reach set is the closed-form parallelotope
  ``|det(W_total)| · prod(ub - lb)``.

Each benchmark has two spec variants:

* ``unsat`` — unsafe far from the data (always UNSAT by construction).
  Tests that the calibrated reach set tightens around the true support.
* ``sat`` — unsafe inside the reach support. Falsifier is OFF in Exp 3,
  so ours abstains (UNKNOWN); the empirical UNKNOWN-rate measures the
  honest-abstention behavior on a hard but reachable spec.

The Exp 3 ``geo_transforms`` benchmark from the README (axis-aligned /
rotated / translated / nonlinear input transforms) is deferred — it
requires an additional input-set transform layer that the existing
Exp 3 networks don't expose. To be wired in once the multi-score-
family comparison is fleshed out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

EXP3_BENCHMARKS: Tuple[str, ...] = (
    '2d_banana', '3d_banana',
    'synth_2d', 'synth_3d', 'synth_5d', 'synth_10d', 'synth_20d',
)
EXP3_SCORES: Tuple[str, ...] = ('flow', 'hyperrect', 'ellipsoid', 'gmm')
EXP3_SPECS: Tuple[str, ...] = ('unsat', 'sat')


# Per-benchmark hparam overrides — same Phase-5d-locked config as
# Exp 3's existing per-benchmark scripts.
PER_BENCHMARK_CONFIG: Dict[str, Dict[str, Any]] = {
    '2d_banana': dict(
        n_train=2_000, flow_epochs=2_000, scenario_n_samples=2_000,
        alpha=0.001, verification_method='amls_bounded', amls_max_levels=30,
        flow_config='base',
    ),
    '3d_banana': dict(
        n_train=2_000, flow_epochs=2_000, scenario_n_samples=2_000,
        alpha=0.001, verification_method='amls_bounded', amls_max_levels=30,
        flow_config='base',
    ),
    'synth_2d': dict(
        n_train=5_000, flow_epochs=2_000, scenario_n_samples=2_000,
        alpha=0.001, verification_method='amls_bounded', amls_max_levels=30,
        flow_config='base',
    ),
    'synth_3d': dict(
        n_train=5_000, flow_epochs=2_000, scenario_n_samples=2_000,
        alpha=0.001, verification_method='amls_bounded', amls_max_levels=30,
        flow_config='base',
    ),
    'synth_5d': dict(
        n_train=5_000, flow_epochs=2_000, scenario_n_samples=2_000,
        alpha=0.001, verification_method='amls_bounded', amls_max_levels=30,
        flow_config='base',
    ),
    'synth_10d': dict(
        n_train=5_000, flow_epochs=2_000, scenario_n_samples=2_000,
        alpha=0.001, verification_method='amls_bounded', amls_max_levels=30,
        flow_config='base',
    ),
    'synth_20d': dict(
        n_train=5_000, flow_epochs=2_000, scenario_n_samples=2_000,
        alpha=0.001, verification_method='amls_bounded', amls_max_levels=30,
        flow_config='base',
    ),
}


def make_network(benchmark: str, *, seed: int = 0):
    """Instantiate the synthetic network for the named benchmark."""
    if benchmark == '2d_banana':
        from examples.FlowConformal.networks import RotatedBananaNet
        return RotatedBananaNet().eval()
    if benchmark == '3d_banana':
        from examples.FlowConformal.networks import ThreeBlobClassifier3D
        return ThreeBlobClassifier3D()
    if benchmark in ('synth_2d', 'synth_3d',
                     'synth_5d', 'synth_10d', 'synth_20d'):
        from examples.FlowConformal.experiments.exp3_synthetic.networks import (
            make_synthetic_2d, make_synthetic_3d,
            make_synthetic_5d, make_synthetic_10d, make_synthetic_20d,
        )
        return {
            'synth_2d': make_synthetic_2d,
            'synth_3d': make_synthetic_3d,
            'synth_5d': make_synthetic_5d,
            'synth_10d': make_synthetic_10d,
            'synth_20d': make_synthetic_20d,
        }[benchmark](seed=seed)
    raise KeyError(
        f'unknown Exp 3 benchmark: {benchmark}; '
        f'expected one of {EXP3_BENCHMARKS}')


def make_input_box(benchmark: str):
    """Default input box ``[lb, ub]`` per benchmark."""
    import numpy as np
    if benchmark == '2d_banana':
        # RotatedBananaNet was trained on x ~ U([0, 1]^2).
        return (np.zeros(2, dtype=np.float32),
                np.ones(2, dtype=np.float32))
    if benchmark == '3d_banana':
        return np.full(3, -1.0, dtype=np.float32), np.full(3, 1.0, dtype=np.float32)
    dim = {
        'synth_2d': 2, 'synth_3d': 3,
        'synth_5d': 5, 'synth_10d': 10, 'synth_20d': 20,
    }.get(benchmark)
    if dim is None:
        raise KeyError(f'unknown benchmark: {benchmark}')
    return (np.full(dim, -0.5, dtype=np.float32),
            np.full(dim,  0.5, dtype=np.float32))


def make_spec(benchmark: str, spec_type: str):
    """Return the :class:`HalfSpace` for the named (benchmark, spec_type) pair.

    * ``spec_type='unsat'`` — unsafe far from the reach set
      (e.g. ``y[0] >= 1e6`` for synthetic-Lipschitz nets, where the
      bounded output range guarantees no point reaches the threshold).
    * ``spec_type='sat'`` — unsafe inside the reach support
      (e.g. ``y[0] >= 0`` so the upper-half of the output distribution
      is "unsafe"; falsifier OFF → ours should return UNKNOWN).
    """
    import numpy as np
    from n2v.sets.halfspace import HalfSpace
    if spec_type not in EXP3_SPECS:
        raise KeyError(
            f'unknown spec type: {spec_type}; expected {EXP3_SPECS}')

    # OneLipschitzNet is square (dim → dim), so synth_<N>d outputs <N>D.
    # ThreeBlobClassifier3D outputs 3D, RotatedBananaNet outputs 2D.
    out_dim = {
        '2d_banana': 2, '3d_banana': 3,
        'synth_2d': 2, 'synth_3d': 3,
        'synth_5d': 5, 'synth_10d': 10, 'synth_20d': 20,
    }[benchmark]
    # Halfspace ``-y_0 <= -threshold``  ⇔  ``y_0 >= threshold``.
    G = np.zeros((1, out_dim), dtype=np.float64)
    G[0, 0] = -1.0
    threshold = 1e6 if spec_type == 'unsat' else 0.0
    g = np.array([[-threshold]], dtype=np.float64)
    return HalfSpace(G, g)
