"""Per-instance verification problem generator for Exp 4.

Each instance is parameterised by ``(depth, instance_idx)``:

  1. Sample a "starting sample" ``x_0 ~ Uniform([-1, 1]^5)`` with an RNG
     seeded by ``hash((depth, instance_idx))``.
  2. The input region is the L∞ ball ``[x_0 - eps, x_0 + eps]`` with
     ``eps=0.1``.
  3. Compute an empirical max ``y_max`` of the network over a few
     thousand uniform samples in the box.
  4. Threshold ``C = y_max + 0.1`` so the spec ``y > C`` is UNSAT by
     construction (the network is 1-Lipschitz, so the true max over
     the box is bounded above by ``f(x_0) + eps·sqrt(in_dim)`` —
     well below ``y_max + 0.1`` for the depths we run).

The unsafe region (in the n2v halfspace convention) is ``G y <= g``
with ``G = [[-1]]`` and ``g = [[-C]]``, i.e. ``y >= C``. ``y > C`` is
open while ``y >= C`` is closed; the difference is measure-zero and
the closed form is what our HalfSpace machinery encodes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from n2v.sets.halfspace import HalfSpace

from .networks import EXP4_IN_DIM, EXP4_OUT_DIM, _stable_hash

EXP4_EPS: float = 0.1
EXP4_INSTANCES_PER_DEPTH: int = 10
EXP4_EMPIRICAL_MAX_SAMPLES: int = 4000
EXP4_THRESHOLD_MARGIN: float = 0.1


def instance_seed(depth: int, instance_idx: int) -> int:
    """Cross-process-deterministic seed for instance generation. See the
    ``_stable_hash`` docstring in :mod:`.networks` for why we can't use
    Python's built-in ``hash()`` here.
    """
    return _stable_hash("exp4_instance", depth, instance_idx)


def generate_instance(
    net: torch.nn.Module,
    depth: int,
    instance_idx: int,
    *,
    in_dim: int = EXP4_IN_DIM,
    eps: float = EXP4_EPS,
    n_emax_samples: int = EXP4_EMPIRICAL_MAX_SAMPLES,
    threshold_margin: float = EXP4_THRESHOLD_MARGIN,
) -> Dict[str, Any]:
    """Construct one verification instance for the given network.

    Returns a dict with the input-box endpoints, the threshold ``C``,
    the empirical max ``y_max``, and the spec :class:`HalfSpace` in
    the unsafe-region convention.
    """
    rng = np.random.RandomState(instance_seed(depth, instance_idx))
    x_0 = rng.uniform(-1.0, 1.0, size=in_dim).astype(np.float32)
    lb = (x_0 - eps).astype(np.float32)
    ub = (x_0 + eps).astype(np.float32)

    # Empirical max via uniform samples in the box (informational only —
    # the spec threshold C below uses the Lipschitz upper bound for
    # mathematical UNSAT-by-construction guarantee).
    samples = rng.uniform(lb, ub, size=(n_emax_samples, in_dim)).astype(np.float32)
    samples_t = torch.from_numpy(samples)
    with torch.no_grad():
        ys = net(samples_t).cpu().numpy().reshape(-1)
    y_max = float(ys.max())

    # ── UNSAT-by-construction via the Lipschitz upper bound ──
    # The networks are 1-Lipschitz w.r.t. L2 norm by spectral
    # normalisation (see networks.py::_spectral_normalize_), so for any
    # x in the L∞ ball of radius eps around x_0:
    #     |f(x) - f(x_0)|  ≤  Lip · ‖x - x_0‖_2
    #                     ≤  1   · sqrt(in_dim) · eps
    # Therefore  max_x f(x)  ≤  f(x_0) + sqrt(in_dim) · eps.
    # Setting C strictly above this upper bound guarantees the spec
    # ``y >= C`` is unsatisfiable mathematically (independent of any
    # empirical sampling). We take the max of the empirical-tightness
    # threshold and the Lipschitz-safe threshold so the spec stays as
    # tight as possible while remaining provably UNSAT.
    with torch.no_grad():
        f_x0 = float(net(torch.from_numpy(x_0[None, :])).cpu().numpy().flatten()[0])
    lipschitz_bound = f_x0 + (in_dim ** 0.5) * eps
    C_lip = lipschitz_bound + threshold_margin  # safety margin
    C_emp = y_max + threshold_margin
    C = max(C_emp, C_lip)

    # Unsafe region in n2v convention: G y <= g  ⟺  -y <= -C  ⟺  y >= C.
    G = np.array([[-1.0]], dtype=np.float64)
    g = np.array([[-C]], dtype=np.float64)
    spec = HalfSpace(G, g)

    return {
        'depth': depth,
        'instance_idx': instance_idx,
        'x_0': x_0,
        'lb': lb,
        'ub': ub,
        'empirical_max': y_max,
        'C': C,
        'eps': eps,
        'spec_halfspace': spec,
        'in_dim': in_dim,
        'out_dim': EXP4_OUT_DIM,
        'instance_seed': instance_seed(depth, instance_idx),
    }


def write_vnnlib(
    path: str | Path,
    lb: np.ndarray,
    ub: np.ndarray,
    C: float,
    *,
    in_dim: int = EXP4_IN_DIM,
    out_dim: int = EXP4_OUT_DIM,
) -> None:
    """Write a VNN-LIB file that encodes the input box and the unsafe
    output region ``y >= C`` so a sound verifier can check the spec.

    A VNN-LIB property is a SAT query: the file asserts the *unsafe*
    region; the verifier returns ``unsat`` if no point in the input box
    maps into it. We mirror the standard `acasxu_2023` style.
    """
    path = Path(path)
    lines: list[str] = []
    for i in range(in_dim):
        lines.append(f'(declare-const X_{i} Real)')
    for i in range(out_dim):
        lines.append(f'(declare-const Y_{i} Real)')
    lines.append('')
    lines.append('; Input box')
    for i in range(in_dim):
        lines.append(f'(assert (<= X_{i} {float(ub[i]):.8f}))')
        lines.append(f'(assert (>= X_{i} {float(lb[i]):.8f}))')
    lines.append('')
    lines.append('; Unsafe: y_0 >= C  ⟺  G y <= g  with  G=[[-1]], g=[[-C]]')
    lines.append(f'(assert (>= Y_0 {float(C):.8f}))')
    lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))
