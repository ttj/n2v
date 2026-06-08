"""Raw Monte Carlo baseline for the bounded-AMLS comparison.

Samples ``N`` points from the truncated Gaussian on ``||z|| <= q``,
pushes them through the inverse flow, and bounds the per-group union
mass ``Pr[flow(z) ∈ U_g | ||z|| <= q]`` via the empirical fraction
plus a one-sided Clopper-Pearson upper bound.

This is the brute-force baseline against which AMLS-bounded is
compared in the ablation: same target distribution, same threshold,
same union semantics, but no level splitting / no MCMC kernel.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from scipy.stats import beta as _beta_dist

from n2v.probabilistic.flow.amls import _push_through_flow, _resolve_device
from n2v.probabilistic.flow.amls_bounded import (
    AMLSBoundedResult,
    AMLSBoundedSpecResult,
    _phi_union_torch,
)
from n2v.probabilistic.flow.scenario_verify import sample_truncated_gaussian_ball


def _clopper_pearson_upper(
    n_success: int, n_total: int, beta: float,
) -> float:
    if n_total <= 0:
        return 1.0
    if n_success >= n_total:
        return 1.0
    return float(_beta_dist.ppf(
        1.0 - beta, n_success + 1, n_total - n_success,
    ))


def raw_mc_estimate_union_mass(
    flow_ode,
    halfspaces,
    *,
    q: float,
    n_samples: int = 2000,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedResult:
    """One pass of uniform Monte Carlo for ``Pr[flow(z) ∈ U_g | π_q]``.

    The returned ``AMLSBoundedResult`` keeps the same shape as the
    AMLS-bounded module so downstream dispatch / aggregation code can
    treat both estimators interchangeably. ``levels_used`` is fixed at
    1 (no splitting) and ``adaptive_step_used`` is always False.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f'beta must be in (0, 1), got {beta}')
    if not halfspaces:
        raise ValueError('halfspaces must be non-empty')

    dev = _resolve_device(device)
    dtype = torch.float32

    halfspaces_torch: list = []
    dim: int | None = None
    for hs in halfspaces:
        G_np = np.asarray(hs.G, dtype=np.float64)
        g_np = np.asarray(hs.g, dtype=np.float64).flatten()
        if dim is None:
            dim = G_np.shape[1]
        elif G_np.shape[1] != dim:
            raise ValueError(
                f'all halfspaces must share input dim; got {G_np.shape[1]} '
                f'vs. {dim}'
            )
        G_t = torch.from_numpy(G_np.astype(np.float32)).to(dev)
        g_t = torch.from_numpy(g_np.astype(np.float32)).to(dev)
        halfspaces_torch.append((G_t, g_t))
    assert dim is not None

    if hasattr(flow_ode, 'to'):
        flow_ode = flow_ode.to(dev)

    seed_int = 0 if seed is None else (int(seed) & 0x7FFFFFFF)
    if seed is not None:
        np.random.seed(seed_int)
        torch.manual_seed(seed_int)

    z_np = sample_truncated_gaussian_ball(q=q, dim=dim, n_samples=n_samples)
    z = torch.as_tensor(z_np.astype(np.float32), device=dev, dtype=dtype)

    y = _push_through_flow(
        flow_ode, z, t=t, n_steps=n_ode_steps,
        method=ode_method, atol=ode_atol, rtol=ode_rtol,
    )
    phi_t = _phi_union_torch(y, halfspaces_torch)
    phi = phi_t.detach().cpu().numpy().astype(np.float64)

    n_in_U = int(np.sum(phi <= 0.0))
    pi_hat = n_in_U / float(n_samples)
    pi_upper = _clopper_pearson_upper(n_in_U, n_samples, beta)

    worst_idx = int(np.argmin(phi))
    worst_y = y[worst_idx].detach().cpu().numpy().astype(np.float64)

    return AMLSBoundedResult(
        pi_hat=pi_hat,
        pi_upper=pi_upper,
        levels_used=1,
        final_unsafe_count=n_in_U,
        detected_unsafe=n_in_U > 0,
        final_phi=float(phi[worst_idx]),
        worst_y=worst_y,
        adaptive_step_used=False,
    )


def raw_mc_certify_spec(
    flow_ode,
    spec_groups,
    *,
    q: float,
    eps_2_target: float,
    n_samples: int = 2000,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedSpecResult:
    """AND-of-OR-of-AND dispatcher mirroring ``amls_bounded_certify_spec_union``.

    Per group we estimate the union mass with one pass of uniform MC
    and apply the same gate: certify the group iff no sample landed
    in U *and* ``pi_upper <= eps_2_target``. The spec is UNSAT iff at
    least one group certifies (the spec's outermost OR).
    """
    dev = _resolve_device(device)

    per_hs_results: list = []
    detected_any = False
    eps_2_upper_max = 0.0
    for group in spec_groups:
        if not group:
            raise ValueError('every group must be non-empty')
        r = raw_mc_estimate_union_mass(
            flow_ode=flow_ode, halfspaces=list(group),
            q=q, n_samples=n_samples, beta=beta, seed=seed,
            t=t, n_ode_steps=n_ode_steps, ode_method=ode_method,
            ode_atol=ode_atol, ode_rtol=ode_rtol, device=dev,
        )
        per_hs_results.append([r])
        if r.detected_unsafe:
            detected_any = True
        if r.pi_upper > eps_2_upper_max:
            eps_2_upper_max = r.pi_upper

    unsat_certified = False
    for group_results in per_hs_results:
        r = group_results[0]
        if (not r.detected_unsafe) and (r.pi_upper <= eps_2_target):
            unsat_certified = True
            break

    return AMLSBoundedSpecResult(
        unsat_certified=unsat_certified,
        detected_any=detected_any,
        eps_2_target=eps_2_target,
        eps_2_upper=eps_2_upper_max,
        per_hs_results=per_hs_results,
        spec_groups=spec_groups,
    )
