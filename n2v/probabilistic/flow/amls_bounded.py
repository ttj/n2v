"""Bounded Adaptive Multilevel Splitting for conformal reach-set verification.

This module implements the verification primitive described in
``docs/research/2026-04-28-bounded-amls-design.md``: AMLS rare-event
estimation **constrained to the conformal latent ball** ``{z : ||z|| <= q}``.

The motivation is that the unconstrained AMLS (in :mod:`amls`) samples
from the *full* Gaussian prior on ``R^d`` and uses MCMC to find samples
that map to unsafe outputs — but those samples may have ``||z|| > q``,
i.e. they lie outside the conformal coverage region. Their detection
under unconstrained AMLS produces "false UNKNOWNs" on benchmarks where
the (1-α) reach set is genuinely disjoint from unsafe but the flow has
nonzero tail mass.

Bounded AMLS restricts both the initial sample and the MCMC chain to
``||z|| <= q``. The MH ratio for a symmetric proposal under the
truncated Gaussian target ``π_q(z) ∝ exp(-||z||²/2) · 1[||z|| <= q]``
is identical to the untruncated case (the truncation indicator's
normalising constant cancels) provided we additionally reject any
proposal with ``||z'|| > q``. This is the standard Modified
Metropolis Algorithm of Au-Beck 2001 with the in-ball indicator
conjoined to the level-set indicator.

The Cérou-Guyader 2007 / 2016 fluctuation analysis carries over: the
estimator ``pi_hat = ρ^K · frac_in_U`` is asymptotically Gaussian with
variance ``σ² = pi^2 · K(1-ρ)/(ρN)``. The corresponding (1-β) upper
confidence bound is ``pi_upper = pi_hat · exp(z_β · √(K(1-ρ)/(ρN)))``,
unchanged from :mod:`amls`. See §3.3 of the design document for the
full derivation.

Public functions:
    * :func:`amls_bounded_estimate_halfspace_mass` — single HalfSpace.
    * :func:`amls_bounded_estimate_union_mass` — OR over a list of
      HalfSpaces, single chain on ``phi_union(y) = min_j phi_j(y)``
      instead of K independent chains. Used for K-disjunct
      classification specs (cifar100, Exp 4 multi-class).
    * :func:`amls_bounded_certify_spec` — AND-of-OR-of-AND dispatcher
      using per-HalfSpace chains.
    * :func:`amls_bounded_certify_spec_union` — AND-of-OR-of-AND
      dispatcher using union-mass chains per group. Mathematically
      tighter and ~K× cheaper for K-disjunct OR groups.

The verdict logic for ``amls_bounded_certify_spec`` differs from
``amls_certify_spec`` in two ways: (a) the chain is constrained to
``||z|| <= q``, (b) UNSAT certification requires both
``not detected_unsafe`` AND ``pi_upper <= eps_2_target`` for every
HalfSpace in some group. The default ``eps_2_target`` is the
caller-supplied ``alpha`` so the joint multiplicative bound
``1 - (1-α)(1-ε_2) ≈ 2α`` matches the Phase 5d-style soundness story.

Spec semantics. The ``spec_groups`` argument is interpreted as
AND-of-OR-of-AND: each ``HalfSpace`` is itself an AND of row
inequalities (point in HalfSpace iff every row satisfied), each group
is an OR of HalfSpaces (point in group iff some HalfSpace contains
it), and the spec is the AND across groups (point unsafe iff every
group contains it). The reach set is UNSAT-disjoint from "unsafe" iff
*at least one* group is fully disjoint from the reach set — the
standard AND-of-OR dispatch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

# Reuse helpers from the unconstrained AMLS module to keep the two
# implementations in lockstep.
from n2v.probabilistic.flow.amls import (
    _phi_halfspace_torch,
    _push_through_flow,
    _resolve_device,
)
from n2v.probabilistic.flow.scenario_verify import (
    sample_truncated_gaussian_ball,
)


def _phi_union_torch(
    y: torch.Tensor,
    halfspaces_torch: List,
) -> torch.Tensor:
    """Vectorized phi_union(y) = min_j phi_halfspace(y, G_j, g_j).

    A point is in the union of halfspaces iff phi_union <= 0. Each
    halfspace itself is an AND of rows: phi_halfspace(y, G, g) =
    max_i (G_i y - g_i).

    Args:
        y: ``(N, d)`` tensor of data points.
        halfspaces_torch: list of ``(G_j, g_j)`` tuples on the same
            device/dtype as ``y``. Each ``G_j`` is ``(k_j, d)`` and
            each ``g_j`` is ``(k_j,)``. Must be non-empty.

    Returns:
        ``(N,)`` tensor of ``min_j (max_i (G_j_i y - g_j_i))``.
    """
    if not halfspaces_torch:
        raise ValueError('halfspaces_torch must be non-empty')
    # Per-halfspace phi: each is a (N,) vector. Stack to (N, J) and
    # take the elementwise min across halfspaces. For single-row
    # halfspaces the inner max is a no-op, so this collapses to a
    # vanilla `min_k (G_k y - g_k)` style computation; we keep the
    # general form so multi-row halfspaces work transparently.
    phi_per_hs = [_phi_halfspace_torch(y, G, g) for G, g in halfspaces_torch]
    return torch.stack(phi_per_hs, dim=1).min(dim=1).values


@dataclass
class AMLSBoundedResult:
    """AMLS-bounded estimator output for a single HalfSpace.

    Mirrors :class:`amls.AMLSResult` with one extra field:
    ``adaptive_step_used``, a diagnostic flag reporting whether the
    ε_step adaptive scaling was active for this run.
    """
    pi_hat: float
    pi_upper: float
    levels_used: int
    final_unsafe_count: int
    detected_unsafe: bool
    final_phi: float
    worst_y: np.ndarray
    adaptive_step_used: bool = False


def amls_bounded_estimate_halfspace_mass(
    flow_ode,
    halfspace,
    *,
    q: float,
    n_samples_per_level: int = 2000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    adaptive_step: bool = False,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedResult:
    """Estimate ``Pr[flow(z) ∈ Unsafe | z ~ π_q]`` via bounded AMLS.

    Same algorithm as :func:`amls.amls_estimate_halfspace_mass` except:

    * Initial sample is drawn from the truncated Gaussian on
      ``||z|| <= q`` via :func:`sample_truncated_gaussian_ball` instead
      of from ``N(0, I)``.
    * MCMC proposals with ``||z'|| > q`` are rejected (the in-ball
      indicator is conjoined to the level-set and MH-ratio
      indicators).
    * If ``adaptive_step=True``, the proposal step size is
      ``mcmc_step_size · min(1, q / sqrt(d))`` to maintain a
      reasonable acceptance rate when ``q`` is small relative to the
      data dimension (Roberts-Tweedie 1996 optimal-scaling rule of
      thumb).

    Args:
        q: Conformal radius. Must be > 0. The estimator's claim is
            conditional on ``||z|| <= q``; this is the (1-α) coverage
            region of the calibrated reach set in latent space.
        n_samples_per_level: Population size per level; same N as the
            unconstrained AMLS. Default 2000.
        quantile: ρ — fraction of samples kept after each level
            update. Default 0.1.
        max_levels: Hard cap on number of levels; the run gives up if
            this is reached without detecting U. Default 30.
        n_mcmc_steps: MCMC steps per replicated point per refill round.
            Default 10.
        mcmc_step_size: Random-walk MH proposal scale before any
            adaptive scaling. Default 0.3.
        adaptive_step: If True, use ``mcmc_step_size · min(1, q/√d)``
            as the effective step size. Useful when ``q`` is small
            relative to the data dim (high-dim outputs). Default False
            so behavior matches unconstrained AMLS.
        beta: Confidence parameter for the asymptotic CI upper bound.
        seed, t, n_ode_steps, ode_method, ode_atol, ode_rtol, device:
            Same as :func:`amls.amls_estimate_halfspace_mass`.

    Returns:
        :class:`AMLSBoundedResult`.
    """
    if q <= 0:
        raise ValueError(f'q must be positive, got {q}')
    if n_samples_per_level <= 0:
        raise ValueError(
            f'n_samples_per_level must be positive, got {n_samples_per_level}')
    if not (0.0 < quantile < 1.0):
        raise ValueError(f'quantile must be in (0, 1), got {quantile}')
    if max_levels < 1:
        raise ValueError(f'max_levels must be >= 1, got {max_levels}')
    if not (0.0 < beta < 1.0):
        raise ValueError(f'beta must be in (0, 1), got {beta}')

    dev = _resolve_device(device)

    G_np = np.asarray(halfspace.G, dtype=np.float64)
    g_np = np.asarray(halfspace.g, dtype=np.float64).flatten()
    dim = G_np.shape[1]
    N = n_samples_per_level
    dtype = torch.float32

    G_t = torch.from_numpy(G_np.astype(np.float32)).to(dev)
    g_t = torch.from_numpy(g_np.astype(np.float32)).to(dev)

    # Optional: scale the proposal step size by q / sqrt(d) so that the
    # ratio of step length to ball radius is roughly constant across
    # benchmarks. Disabled by default to match :mod:`amls`.
    if adaptive_step:
        eff_step = mcmc_step_size * min(1.0, q / math.sqrt(max(dim, 1)))
    else:
        eff_step = mcmc_step_size

    # Pre-compute ||z|| <= q test constant on device.
    q_sq_t = torch.tensor(q * q, device=dev, dtype=dtype)

    if hasattr(flow_ode, 'to'):
        flow_ode = flow_ode.to(dev)

    seed_int = 0 if seed is None else (int(seed) & 0x7FFFFFFF)
    gen = torch.Generator(device=dev).manual_seed(seed_int)
    if seed is not None:
        torch.manual_seed(seed_int)

    # ---- Level 0: initial truncated-Gaussian sample on ||z|| <= q ----
    # ``sample_truncated_gaussian_ball`` uses the global numpy RNG;
    # seed it for reproducibility alongside the per-device torch
    # generator.
    if seed is not None:
        np.random.seed(seed_int)
    z_np = sample_truncated_gaussian_ball(q=q, dim=dim, n_samples=N)
    z = torch.as_tensor(z_np.astype(np.float32), device=dev, dtype=dtype)

    y = _push_through_flow(
        flow_ode, z, t=t, n_steps=n_ode_steps,
        method=ode_method, atol=ode_atol, rtol=ode_rtol,
    )
    phi_t = _phi_halfspace_torch(y, G_t, g_t)

    best_idx_t = torch.argmin(phi_t)
    best_phi_t = phi_t[best_idx_t].detach().clone()
    best_y_t = y[best_idx_t].detach().clone()

    # Level-0 detection: any sample already in U => return immediately.
    in_U_mask = phi_t <= 0.0
    if bool(in_U_mask.any().item()):
        n_in = int(in_U_mask.sum().item())
        pi_hat = n_in / N
        from scipy.stats import beta as _beta_dist
        if n_in == N:
            pi_upper = 1.0
        else:
            pi_upper = float(_beta_dist.ppf(1.0 - beta, n_in + 1, N - n_in))
        best_phi = float(best_phi_t.item())
        best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
        return AMLSBoundedResult(
            pi_hat=pi_hat,
            pi_upper=pi_upper,
            levels_used=1,
            final_unsafe_count=n_in,
            detected_unsafe=True,
            final_phi=best_phi,
            worst_y=best_y,
            adaptive_step_used=adaptive_step,
        )

    # ---- Level loop ----
    tau_prev = math.inf
    K = 0
    for level in range(max_levels):
        K = level + 1

        tau_k_t = torch.quantile(phi_t, quantile)
        tau_k = float(tau_k_t.item())
        if tau_k >= tau_prev:
            tau_k = tau_prev - 1e-12
            tau_k_t = torch.tensor(tau_k, device=dev, dtype=dtype)

        if tau_k <= 0.0:
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** (K - 1)) * frac_in_U
            from scipy.stats import norm
            sigma2 = K * (1.0 - quantile) / (quantile * N)
            log_pi_upper = math.log(max(pi_hat, 1e-300)) + \
                norm.ppf(1.0 - beta) * math.sqrt(sigma2)
            pi_upper = math.exp(log_pi_upper)
            best_idx_t = torch.argmin(phi_t)
            best_phi_now = float(phi_t[best_idx_t].item())
            if best_phi_now < float(best_phi_t.item()):
                best_phi_t = phi_t[best_idx_t].detach().clone()
                best_y_t = y[best_idx_t].detach().clone()
            best_phi = float(best_phi_t.item())
            best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
            return AMLSBoundedResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=bool(in_U.any().item()),
                final_phi=best_phi,
                worst_y=best_y,
                adaptive_step_used=adaptive_step,
            )

        keep_count = max(1, int(round(N * quantile)))
        _kept_phi, keep_idx_t = torch.topk(
            phi_t, k=keep_count, largest=False, sorted=False,
        )
        if keep_count == 0:
            break

        sample_indices = torch.randint(
            0, keep_count, (N,), device=dev, generator=gen,
        )
        chosen_t = keep_idx_t[sample_indices]
        z_cur = z.index_select(0, chosen_t).clone()
        y_cur = y.index_select(0, chosen_t).clone()
        phi_cur = phi_t.index_select(0, chosen_t).clone()

        if isinstance(tau_k_t, torch.Tensor):
            tau_k_dev = tau_k_t.to(device=dev, dtype=dtype)
        else:
            tau_k_dev = torch.tensor(tau_k, device=dev, dtype=dtype)

        for _step in range(n_mcmc_steps):
            eta = torch.randn(
                N, dim, device=dev, dtype=dtype, generator=gen,
            )
            z_prop = z_cur + eff_step * eta

            # ---- BOUNDED-AMLS DIFFERENCE ----
            # In-ball indicator on the proposal. Proposals with
            # ||z'||² > q² are unconditionally rejected, mirroring
            # the truncated-Gaussian target's support boundary.
            z_prop_norm_sq = (z_prop * z_prop).sum(dim=1)
            in_ball = z_prop_norm_sq <= q_sq_t

            log_alpha = 0.5 * (
                (z_cur * z_cur).sum(dim=1) - z_prop_norm_sq
            )
            u = torch.rand(N, device=dev, dtype=dtype, generator=gen)
            log_u = torch.log(u)
            mh_pass = log_u < log_alpha

            y_prop = _push_through_flow(
                flow_ode, z_prop, t=t, n_steps=n_ode_steps,
                method=ode_method, atol=ode_atol, rtol=ode_rtol,
            )
            phi_prop = _phi_halfspace_torch(y_prop, G_t, g_t)

            level_pass = phi_prop <= tau_k_dev

            # Conjoin all three predicates: stay in ball, stay in
            # level set, and pass MH ratio.
            accept = mh_pass & level_pass & in_ball

            cur_best_idx = torch.argmin(phi_prop)
            cur_best_phi = phi_prop[cur_best_idx]
            improve = cur_best_phi < best_phi_t
            best_phi_t = torch.where(improve, cur_best_phi, best_phi_t)
            best_y_t = torch.where(
                improve.unsqueeze(-1), y_prop[cur_best_idx], best_y_t,
            )

            accept_b = accept.unsqueeze(-1)
            z_cur = torch.where(accept_b, z_prop, z_cur)
            y_cur = torch.where(accept_b, y_prop, y_cur)
            phi_cur = torch.where(accept, phi_prop, phi_cur)

        z = z_cur
        y = y_cur
        phi_t = phi_cur
        tau_prev = tau_k

        # Single sync per outer-level iteration: have we crossed into U?
        if bool((phi_t <= 0.0).any().item()):
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** K) * frac_in_U
            from scipy.stats import norm
            sigma2 = (K + 1) * (1.0 - quantile) / (quantile * N)
            log_pi_upper = math.log(max(pi_hat, 1e-300)) + \
                norm.ppf(1.0 - beta) * math.sqrt(sigma2)
            pi_upper = math.exp(log_pi_upper)
            best_phi = float(best_phi_t.item())
            best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
            return AMLSBoundedResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K + 1,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=True,
                final_phi=best_phi,
                worst_y=best_y,
                adaptive_step_used=adaptive_step,
            )

    # Exhausted max_levels without detecting U.
    pi_hat = quantile ** K
    from scipy.stats import norm
    sigma2 = K * (1.0 - quantile) / (quantile * N)
    log_pi_upper = math.log(max(pi_hat, 1e-300)) + \
        norm.ppf(1.0 - beta) * math.sqrt(sigma2)
    pi_upper = min(1.0, math.exp(log_pi_upper))

    best_phi = float(best_phi_t.item())
    best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
    return AMLSBoundedResult(
        pi_hat=pi_hat,
        pi_upper=pi_upper,
        levels_used=K,
        final_unsafe_count=0,
        detected_unsafe=bool(best_phi <= 0.0),
        final_phi=best_phi,
        worst_y=best_y,
        adaptive_step_used=adaptive_step,
    )


def amls_bounded_estimate_union_mass(
    flow_ode,
    halfspaces: List,
    *,
    q: float,
    n_samples_per_level: int = 2000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    adaptive_step: bool = False,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedResult:
    """Estimate ``Pr[flow(z) ∈ ⋃_j U_j | z ~ π_q]`` via bounded AMLS.

    Same in-ball algorithm as :func:`amls_bounded_estimate_halfspace_mass`,
    but the level-set indicator is computed against
    ``phi_union(y) = min_j phi_halfspace(y, G_j, g_j)`` so the chain
    targets the union of all halfspaces in ``halfspaces`` simultaneously.
    The resulting ``pi_hat`` estimates the union mass directly, which
    is mathematically tighter and ``len(halfspaces)`` × cheaper than
    running independent chains and Bonferroni-unioning.

    Used for K-disjunct classification specs (cifar100 with K=99
    other-class halfspaces, Exp 4 multi-class) where the OR semantics
    naturally fit a single chain on ``phi_min``.

    Args:
        halfspaces: non-empty list of HalfSpace objects forming the OR.
            Each HalfSpace itself encodes an AND of row inequalities;
            ``phi_halfspace`` returns ``max_i (G_i y - g_i)``.

    See :func:`amls_bounded_estimate_halfspace_mass` for the remaining
    keyword arguments.

    Returns:
        :class:`AMLSBoundedResult`. ``worst_y`` is the sample with the
        smallest ``phi_union`` value (the one most committed to *some*
        halfspace's interior).
    """
    if not halfspaces:
        raise ValueError('halfspaces must be non-empty')
    if q <= 0:
        raise ValueError(f'q must be positive, got {q}')
    if n_samples_per_level <= 0:
        raise ValueError(
            f'n_samples_per_level must be positive, got {n_samples_per_level}')
    if not (0.0 < quantile < 1.0):
        raise ValueError(f'quantile must be in (0, 1), got {quantile}')
    if max_levels < 1:
        raise ValueError(f'max_levels must be >= 1, got {max_levels}')
    if not (0.0 < beta < 1.0):
        raise ValueError(f'beta must be in (0, 1), got {beta}')

    dev = _resolve_device(device)
    dtype = torch.float32

    # Convert each halfspace to torch tensors on the resolved device.
    halfspaces_torch: List = []
    dim = None
    for hs in halfspaces:
        G_np = np.asarray(hs.G, dtype=np.float64)
        g_np = np.asarray(hs.g, dtype=np.float64).flatten()
        if dim is None:
            dim = G_np.shape[1]
        elif G_np.shape[1] != dim:
            raise ValueError(
                f'all halfspaces must share input dim; got {G_np.shape[1]} '
                f'vs. {dim}')
        G_t = torch.from_numpy(G_np.astype(np.float32)).to(dev)
        g_t = torch.from_numpy(g_np.astype(np.float32)).to(dev)
        halfspaces_torch.append((G_t, g_t))

    N = n_samples_per_level

    # Optional Roberts-Tweedie scaling: maintain a stable acceptance rate
    # when q is small relative to the data dim. Disabled by default to
    # match the single-HalfSpace path.
    if adaptive_step:
        eff_step = mcmc_step_size * min(1.0, q / math.sqrt(max(dim, 1)))
    else:
        eff_step = mcmc_step_size

    q_sq_t = torch.tensor(q * q, device=dev, dtype=dtype)

    if hasattr(flow_ode, 'to'):
        flow_ode = flow_ode.to(dev)

    seed_int = 0 if seed is None else (int(seed) & 0x7FFFFFFF)
    gen = torch.Generator(device=dev).manual_seed(seed_int)
    if seed is not None:
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)

    # ---- Level 0: initial truncated-Gaussian sample on ||z|| <= q ----
    z_np = sample_truncated_gaussian_ball(q=q, dim=dim, n_samples=N)
    z = torch.as_tensor(z_np.astype(np.float32), device=dev, dtype=dtype)

    y = _push_through_flow(
        flow_ode, z, t=t, n_steps=n_ode_steps,
        method=ode_method, atol=ode_atol, rtol=ode_rtol,
    )
    phi_t = _phi_union_torch(y, halfspaces_torch)

    best_idx_t = torch.argmin(phi_t)
    best_phi_t = phi_t[best_idx_t].detach().clone()
    best_y_t = y[best_idx_t].detach().clone()

    # Level-0 detection: any sample already in the union => return.
    in_U_mask = phi_t <= 0.0
    if bool(in_U_mask.any().item()):
        n_in = int(in_U_mask.sum().item())
        pi_hat = n_in / N
        from scipy.stats import beta as _beta_dist
        if n_in == N:
            pi_upper = 1.0
        else:
            pi_upper = float(_beta_dist.ppf(1.0 - beta, n_in + 1, N - n_in))
        best_phi = float(best_phi_t.item())
        best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
        return AMLSBoundedResult(
            pi_hat=pi_hat,
            pi_upper=pi_upper,
            levels_used=1,
            final_unsafe_count=n_in,
            detected_unsafe=True,
            final_phi=best_phi,
            worst_y=best_y,
            adaptive_step_used=adaptive_step,
        )

    # ---- Level loop ----
    tau_prev = math.inf
    K = 0
    for level in range(max_levels):
        K = level + 1

        tau_k_t = torch.quantile(phi_t, quantile)
        tau_k = float(tau_k_t.item())
        if tau_k >= tau_prev:
            tau_k = tau_prev - 1e-12
            tau_k_t = torch.tensor(tau_k, device=dev, dtype=dtype)

        if tau_k <= 0.0:
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** (K - 1)) * frac_in_U
            from scipy.stats import norm
            sigma2 = K * (1.0 - quantile) / (quantile * N)
            log_pi_upper = math.log(max(pi_hat, 1e-300)) + \
                norm.ppf(1.0 - beta) * math.sqrt(sigma2)
            pi_upper = math.exp(log_pi_upper)
            best_idx_t = torch.argmin(phi_t)
            best_phi_now = float(phi_t[best_idx_t].item())
            if best_phi_now < float(best_phi_t.item()):
                best_phi_t = phi_t[best_idx_t].detach().clone()
                best_y_t = y[best_idx_t].detach().clone()
            best_phi = float(best_phi_t.item())
            best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
            return AMLSBoundedResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=bool(in_U.any().item()),
                final_phi=best_phi,
                worst_y=best_y,
                adaptive_step_used=adaptive_step,
            )

        keep_count = max(1, int(round(N * quantile)))
        _kept_phi, keep_idx_t = torch.topk(
            phi_t, k=keep_count, largest=False, sorted=False,
        )
        if keep_count == 0:
            break

        sample_indices = torch.randint(
            0, keep_count, (N,), device=dev, generator=gen,
        )
        chosen_t = keep_idx_t[sample_indices]
        z_cur = z.index_select(0, chosen_t).clone()
        y_cur = y.index_select(0, chosen_t).clone()
        phi_cur = phi_t.index_select(0, chosen_t).clone()

        if isinstance(tau_k_t, torch.Tensor):
            tau_k_dev = tau_k_t.to(device=dev, dtype=dtype)
        else:
            tau_k_dev = torch.tensor(tau_k, device=dev, dtype=dtype)

        for _step in range(n_mcmc_steps):
            eta = torch.randn(
                N, dim, device=dev, dtype=dtype, generator=gen,
            )
            z_prop = z_cur + eff_step * eta

            # In-ball indicator on the proposal (bounded-AMLS condition).
            z_prop_norm_sq = (z_prop * z_prop).sum(dim=1)
            in_ball = z_prop_norm_sq <= q_sq_t

            log_alpha = 0.5 * (
                (z_cur * z_cur).sum(dim=1) - z_prop_norm_sq
            )
            u = torch.rand(N, device=dev, dtype=dtype, generator=gen)
            log_u = torch.log(u)
            mh_pass = log_u < log_alpha

            y_prop = _push_through_flow(
                flow_ode, z_prop, t=t, n_steps=n_ode_steps,
                method=ode_method, atol=ode_atol, rtol=ode_rtol,
            )
            phi_prop = _phi_union_torch(y_prop, halfspaces_torch)

            level_pass = phi_prop <= tau_k_dev

            accept = mh_pass & level_pass & in_ball

            cur_best_idx = torch.argmin(phi_prop)
            cur_best_phi = phi_prop[cur_best_idx]
            improve = cur_best_phi < best_phi_t
            best_phi_t = torch.where(improve, cur_best_phi, best_phi_t)
            best_y_t = torch.where(
                improve.unsqueeze(-1), y_prop[cur_best_idx], best_y_t,
            )

            accept_b = accept.unsqueeze(-1)
            z_cur = torch.where(accept_b, z_prop, z_cur)
            y_cur = torch.where(accept_b, y_prop, y_cur)
            phi_cur = torch.where(accept, phi_prop, phi_cur)

        z = z_cur
        y = y_cur
        phi_t = phi_cur
        tau_prev = tau_k

        if bool((phi_t <= 0.0).any().item()):
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** K) * frac_in_U
            from scipy.stats import norm
            sigma2 = (K + 1) * (1.0 - quantile) / (quantile * N)
            log_pi_upper = math.log(max(pi_hat, 1e-300)) + \
                norm.ppf(1.0 - beta) * math.sqrt(sigma2)
            pi_upper = math.exp(log_pi_upper)
            best_phi = float(best_phi_t.item())
            best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
            return AMLSBoundedResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K + 1,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=True,
                final_phi=best_phi,
                worst_y=best_y,
                adaptive_step_used=adaptive_step,
            )

    # Exhausted max_levels without detecting U.
    pi_hat = quantile ** K
    from scipy.stats import norm
    sigma2 = K * (1.0 - quantile) / (quantile * N)
    log_pi_upper = math.log(max(pi_hat, 1e-300)) + \
        norm.ppf(1.0 - beta) * math.sqrt(sigma2)
    pi_upper = min(1.0, math.exp(log_pi_upper))

    best_phi = float(best_phi_t.item())
    best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
    return AMLSBoundedResult(
        pi_hat=pi_hat,
        pi_upper=pi_upper,
        levels_used=K,
        final_unsafe_count=0,
        detected_unsafe=bool(best_phi <= 0.0),
        final_phi=best_phi,
        worst_y=best_y,
        adaptive_step_used=adaptive_step,
    )


@dataclass
class AMLSBoundedSpecResult:
    """Aggregate bounded-AMLS result for a full AND-of-OR-of-AND spec.

    The spec is UNSAT-disjoint iff at least one group has every
    HalfSpace certified-disjoint, where a HalfSpace is certified
    iff (a) bounded AMLS did not detect a witness and (b) the
    asymptotic upper bound ``pi_upper <= eps_2_target``.

    Compared to :class:`amls.AMLSSpecResult`:

    * ``unsat_certified`` here uses both ``not detected_unsafe`` AND
      ``pi_upper <= eps_2_target``. The unconstrained-AMLS version
      uses only ``not detected_unsafe`` and reports
      ``epsilon_2 = 0.0`` upstream, which loses the rare-event
      probability bound entirely. Bounded AMLS is intended for
      callers that want to compose with a multiplicative joint
      bound ``1 - (1-α)(1-ε_2)``.
    """
    unsat_certified: bool
    detected_any: bool
    eps_2_target: float
    eps_2_upper: float           # max pi_upper over all halfspaces
    per_hs_results: List         # list[list[AMLSBoundedResult]]
    spec_groups: List


def amls_bounded_certify_spec(
    flow_ode,
    spec_groups,
    *,
    q: float,
    eps_2_target: float,
    n_samples_per_level: int = 2000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    adaptive_step: bool = False,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedSpecResult:
    """Run bounded AMLS per HalfSpace across an AND-of-OR-of-AND spec.

    Mirrors :func:`amls.amls_certify_spec` with two changes:

    1. Each per-HalfSpace call uses :func:`amls_bounded_estimate_halfspace_mass`
       (in-ball constrained chain).
    2. Verdict gate per HalfSpace: ``not detected_unsafe AND
       pi_upper <= eps_2_target``. This makes the joint multiplicative
       bound ``1 - (1-α)(1-ε_2)`` meaningful — passing implies
       ``ε_2 ≤ ε_2_target`` for that halfspace.

    A group is fully disjoint iff every HalfSpace in it passes the
    above gate. The spec is UNSAT iff at least one group is fully
    disjoint (the standard OR-of-disjoint-groups dispatch).

    Args:
        q: Conformal radius. Same value used by all HalfSpaces (the
            calibrated set is shared).
        eps_2_target: Target upper bound on the rare-event probability
            per halfspace. Typical choice: the conformal alpha
            (``α=0.001``) so the joint multiplicative bound becomes
            ``1 - (1-α)² ≈ 2α``.

    See :func:`amls_bounded_estimate_halfspace_mass` for the remaining
    arguments.
    """
    dev = _resolve_device(device)

    # SEED=47 convention: every halfspace runs with the same seed. The
    # MCMC chains diverge naturally because their phi functions differ.
    per_hs_results: list = []
    detected_any = False
    eps_2_upper_max = 0.0
    for group in spec_groups:
        group_results = []
        for hs in group:
            r = amls_bounded_estimate_halfspace_mass(
                flow_ode=flow_ode, halfspace=hs,
                q=q,
                n_samples_per_level=n_samples_per_level,
                quantile=quantile,
                max_levels=max_levels,
                n_mcmc_steps=n_mcmc_steps,
                mcmc_step_size=mcmc_step_size,
                adaptive_step=adaptive_step,
                beta=beta, seed=seed,
                t=t, n_ode_steps=n_ode_steps,
                ode_method=ode_method,
                ode_atol=ode_atol, ode_rtol=ode_rtol,
                device=dev,
            )
            group_results.append(r)
            if r.detected_unsafe:
                detected_any = True
            if r.pi_upper > eps_2_upper_max:
                eps_2_upper_max = r.pi_upper
        per_hs_results.append(group_results)

    # UNSAT iff some group has every HalfSpace passing the gate.
    #
    # Verdict gate: BOTH ``not detected_unsafe`` AND ``pi_upper <=
    # eps_2_target`` must hold. This is the scenario+mass interpretation
    # of AMLS verification:
    #   - ``not detected_unsafe`` is the SCENARIO check: AMLS's
    #     importance-weighted chain found NO samples in U during its
    #     search. (AMLS is purpose-built to find rare unsafe samples
    #     so detection is a meaningful signal even when the
    #     extrapolated mass bound says they're rare.)
    #   - ``pi_upper <= eps_2_target`` is the MASS-bound check: the
    #     (1-beta)-confidence upper bound on the rare-event mass meets
    #     the joint certificate's target.
    # Both contribute. The mass bound alone (relaxed gate) caused
    # ~14% false-UNSAT rate on lsnc_relu and ~40% on metaroom because
    # the flow's learned density on rare unsafe regions can be small
    # enough to pass the mass check while AMLS still finds samples
    # there — exactly the scenario the detected_unsafe gate is
    # designed to catch. See conversation 2026-05-04 for the audit
    # that motivated reverting from the relaxed gate back to this
    # strict scenario+mass gate.
    unsat_certified = False
    for group_results in per_hs_results:
        if all(
            (not r.detected_unsafe) and (r.pi_upper <= eps_2_target)
            for r in group_results
        ):
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


def amls_bounded_certify_spec_union(
    flow_ode,
    spec_groups,
    *,
    q: float,
    eps_2_target: float,
    n_samples_per_level: int = 2000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    adaptive_step: bool = False,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedSpecResult:
    """AND-of-OR-of-AND dispatcher using one union-mass chain per group.

    Mirrors :func:`amls_bounded_certify_spec` but runs a single bounded
    AMLS chain per group on ``phi_union(y) = min_j phi_halfspace_j(y)``
    instead of ``len(group)`` independent chains. The estimator
    ``pi_hat`` and asymptotic upper bound ``pi_upper`` are then for the
    *union* of halfspaces in the group — so the gate ``not
    detected_unsafe AND pi_upper <= eps_2_target`` directly certifies
    the group disjoint, with no Bonferroni union step needed.

    Use this for K-disjunct OR groups (cifar100 with 99 other-class
    halfspaces, Exp 4 multi-class). Compared to the per-HalfSpace
    dispatcher this is mathematically tighter (one chain estimates
    the union mass directly) and roughly ``len(group)`` × cheaper.

    The verdict logic is identical to :func:`amls_bounded_certify_spec`:
    UNSAT iff at least one group passes the gate.

    Note on ``per_hs_results`` layout. Each group has *one* result
    object (the union-mass run), not one per halfspace; we still wrap
    it in a list so the field type matches
    :class:`AMLSBoundedSpecResult`.

    See :func:`amls_bounded_estimate_union_mass` for the remaining
    keyword arguments.
    """
    dev = _resolve_device(device)

    per_hs_results: list = []
    detected_any = False
    eps_2_upper_max = 0.0
    for group in spec_groups:
        if not group:
            raise ValueError('every group must be non-empty')
        r = amls_bounded_estimate_union_mass(
            flow_ode=flow_ode, halfspaces=list(group),
            q=q,
            n_samples_per_level=n_samples_per_level,
            quantile=quantile,
            max_levels=max_levels,
            n_mcmc_steps=n_mcmc_steps,
            mcmc_step_size=mcmc_step_size,
            adaptive_step=adaptive_step,
            beta=beta, seed=seed,
            t=t, n_ode_steps=n_ode_steps,
            ode_method=ode_method,
            ode_atol=ode_atol, ode_rtol=ode_rtol,
            device=dev,
        )
        per_hs_results.append([r])
        if r.detected_unsafe:
            detected_any = True
        if r.pi_upper > eps_2_upper_max:
            eps_2_upper_max = r.pi_upper

    # Same scenario+mass gate as in amls_bounded_certify_spec above:
    # certify only when AMLS found NO samples in U (scenario check)
    # AND the formal pi_upper bound is met (mass check).
    unsat_certified = False
    for group_results in per_hs_results:
        # Each group has exactly one union-mass result.
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
