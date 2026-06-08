"""Importance sampling with a flow-tilted proposal for rare-event flow-set queries.

Given a calibrated FlowODE and an unsafe polyhedron ``U = { y : G y <= g }``,
this module estimates ``P_flow(U)`` via self-normalised IS with weights
that exponentially tilt toward ``U``. By the VNN-LIB convention,

    phi(y) := max_i (G_i y - g_i)

is non-positive iff ``y in U`` (every row of ``G y <= g`` is satisfied).
The weight tilts samples toward ``U`` (small ``phi``):

    w(z) := exp(-lambda * max(0, phi(phi_flow^{-1}(z))))

For points already in ``U`` (``phi <= 0``) the weight is ``1``; for points
outside ``U`` (``phi > 0``) the weight decays exponentially in distance to
``U``, with rate ``lambda > 0``.

Sampling is performed directly from the prior ``p_z = N(0, I_d)``; the
weights are an estimator-side reweighting that concentrates empirical
mass near ``U``. The result includes:

    pi_hat = sum_i w_i * 1[phi(y_i) <= 0] / sum_i w_i  (self-normalised)
    ess    = (sum w_i)^2 / sum w_i^2
    pi_upper = empirical-Bernstein upper bound (diagnostic)
    detected_unsafe = any(phi(y_i) <= 0)

References:
    Owen 2013 -- Monte Carlo theory, methods and examples (ch. 9).
    Gao et al. ICLR 2024 -- NOFIS.
    Asghar et al. Nature MI 2024 -- FlowRES.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ISResult:
    """Result of a single-HalfSpace tilted-IS run.

    Attributes:
        pi_hat: Self-normalised IS estimator. With samples drawn from
            ``p_z`` and weights ``w(z)``, this is
            ``E_{p_z}[w * 1[U]] / E_{p_z}[w] = E_q[1[U]] = P(U | q)``,
            i.e. the q-CONDITIONAL mass of ``U``, not ``P_flow(U)``.
            Use as a diagnostic; the detection flag (``detected_unsafe``)
            is the operationally important output.
        pi_upper: Empirical-Bernstein ``(1 - beta)`` upper bound on
            ``P_flow(y in U)`` (diagnostic; not a finite-sample certificate
            on the joint conformal+flow guarantee).
        ess: Effective sample size ``(sum w)^2 / sum w^2``. ``ess << N``
            indicates the tilt is too aggressive.
        n_samples: ``N``, total samples drawn from ``p_z``.
        n_in_U: Number of unweighted samples with ``phi(y) <= 0``
            (i.e. inside ``U``).
        detected_unsafe: True iff at least one sample (regardless of its
            tilt weight) landed in ``U``.
        final_phi: MIN ``phi(y_i)`` observed -- the deepest-into-``U``
            witness if detected (most negative), else the closest-to-``U``
            sample (smallest positive).
        worst_y: ``(d,)`` array -- the sample that achieved ``final_phi``.
        lambda_tilt: The tilt parameter used (passed through for logging).
    """
    pi_hat: float
    pi_upper: float
    ess: float
    n_samples: int
    n_in_U: int
    detected_unsafe: bool
    final_phi: float
    worst_y: np.ndarray
    lambda_tilt: float


def _phi_halfspace(y_np: np.ndarray, G: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Signed-distance-to-U importance function: max row margin per sample.

    ``phi(y) = max_i (G_i y - g_i)`` is non-positive iff ``y in U``
    (matches the AMLS / VNN-LIB convention).
    """
    margins = y_np @ G.T - g[None, :]  # (N, k)
    return margins.max(axis=1)  # (N,)


def _push_through_flow(
    flow_ode,
    z: torch.Tensor,
    *,
    t: float,
    n_steps: int,
    method: str,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    # Move ``z`` onto whichever device ``flow_ode`` currently lives on
    # (sibling certify functions — amls / amls_bounded / raw_mc_uniform —
    # mutate ``flow_ode`` device in-place, so a shared-flow ablation
    # sweep may have left it on CUDA before this call). Returns result on
    # CPU so downstream numpy conversion is safe.
    #
    # Test doubles may not be nn.Module instances (no .parameters()); in
    # that case we leave ``z`` on its current device and trust the mock.
    flow_device = None
    if hasattr(flow_ode, 'parameters'):
        try:
            flow_device = next(flow_ode.parameters()).device
        except StopIteration:
            flow_device = None
    if flow_device is not None:
        z = z.to(flow_device)
    with torch.no_grad():
        y = flow_ode.inverse(
            z, t=t, n_steps=n_steps,
            method=method, atol=atol, rtol=rtol,
        )
    return y.detach().cpu() if isinstance(y, torch.Tensor) else y


def is_tilted_estimate_halfspace_mass(
    flow_ode,
    halfspace,
    *,
    n_samples: int = 2000,
    lambda_tilt: float = 5.0,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    chunk_size: int = 8192,
) -> ISResult:
    """Self-normalised IS estimate of ``P_flow(y in U)`` with tilted weights.

    Algorithm:
      1. Draw ``z_1, ..., z_N ~ N(0, I_d)``.
      2. Push through the flow inverse to obtain ``y_i = phi_flow^{-1}(z_i)``.
      3. Compute ``phi_i = max_j (G_j y_i - g_j)`` (non-positive iff in U).
      4. Tilted weights ``w_i = exp(-lambda * max(0, phi_i))``. For
         samples already in ``U`` (phi <= 0) the weight is 1; for samples
         outside ``U`` the weight decays exponentially in phi.
      5. Self-normalised estimator:
         ``pi_hat = sum_i w_i * 1[phi_i <= 0] / sum_i w_i``.
      6. ``ESS = (sum w_i)^2 / sum w_i^2``.
      7. Empirical-Bernstein upper bound (Maurer-Pontil 2009) with
         weights clipped at 1.

    Args:
        flow_ode: trained ``FlowODE`` instance.
        halfspace: object with ``.G`` (k, d) and ``.g`` (k, 1) or (k,)
            attributes.
        n_samples: ``N``, the number of latent draws from ``p_z``.
        lambda_tilt: ``lambda`` >= 0, tilt strength. ``0`` reverts to
            uniform; ``5-10`` is a reasonable starting range.
        beta: confidence parameter for the upper bound.
        seed: RNG seed.
        t, n_ode_steps, ode_method, ode_atol, ode_rtol: ODE solver
            settings; passed to ``flow_ode.inverse``.
        chunk_size: batch size for flow inverse calls. Reduces peak
            memory when ``n_samples`` is large.

    Returns:
        ``ISResult``.
    """
    if n_samples <= 0:
        raise ValueError(f'n_samples must be positive, got {n_samples}')
    if lambda_tilt < 0:
        raise ValueError(f'lambda_tilt must be >= 0, got {lambda_tilt}')
    if not (0.0 < beta < 1.0):
        raise ValueError(f'beta must be in (0, 1), got {beta}')

    G = np.asarray(halfspace.G, dtype=np.float64)
    g = np.asarray(halfspace.g, dtype=np.float64).flatten()
    dim = G.shape[1]

    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(int(seed) & 0x7FFFFFFF)

    # 1-2. Sample z ~ N(0, I) and push through the flow inverse, in chunks
    # to bound memory.
    all_y = np.empty((n_samples, dim), dtype=np.float64)
    z_all = rng.standard_normal((n_samples, dim)).astype(np.float32)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        z_chunk = torch.from_numpy(z_all[start:stop])
        y_chunk = _push_through_flow(
            flow_ode, z_chunk, t=t, n_steps=n_ode_steps,
            method=ode_method, atol=ode_atol, rtol=ode_rtol,
        )
        all_y[start:stop] = y_chunk.numpy().astype(np.float64)

    # 3. Compute phi (margins). phi <= 0 iff in U (VNN-LIB convention).
    phi_vals = _phi_halfspace(all_y, G, g)

    # 4. Tilted weights. Weights are in (0, 1].
    # For phi <= 0 (in U): w = 1; otherwise w = exp(-lambda * phi).
    pos_phi = np.maximum(0.0, phi_vals)
    log_w = -lambda_tilt * pos_phi
    # Numerical safety: log_w in (-inf, 0]; exp -> (0, 1].
    weights = np.exp(log_w)

    # 5. Self-normalised estimator.
    in_U = phi_vals <= 0.0
    sum_w = float(weights.sum())
    if sum_w == 0.0:
        # Degenerate: all weights underflowed. Fall back to flat MC indicator.
        pi_hat = float(in_U.mean())
    else:
        pi_hat = float((weights * in_U).sum() / sum_w)

    # 6. Effective sample size.
    sum_w2 = float((weights ** 2).sum())
    ess = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else 0.0

    # 7. Empirical-Bernstein upper bound (Maurer-Pontil 2009) for a
    # bounded weighted indicator. Variates X_i = w_i * 1[in_U_i] in [0, 1].
    # Upper bound on E[X] = pi_hat (here we treat the SN-IS estimate as
    # the empirical mean of X_i directly; the small bias from
    # normalisation is O(1/N) and dominated by the variance term for
    # the diagnostic).
    X = weights * in_U.astype(np.float64)
    X_mean = float(X.mean())
    X_var = float(X.var(ddof=1)) if n_samples > 1 else 0.0
    if n_samples > 1:
        ln_2_over_beta = math.log(2.0 / beta)
        eb_var_term = math.sqrt(2.0 * X_var * ln_2_over_beta / n_samples)
        eb_range_term = (7.0 / 3.0) * 1.0 * ln_2_over_beta / (n_samples - 1)
        pi_upper = min(1.0, X_mean + eb_var_term + eb_range_term)
    else:
        pi_upper = 1.0

    # Best (smallest-phi) witness: deepest into U if detected, else
    # closest to U.
    best_idx = int(np.argmin(phi_vals))
    final_phi = float(phi_vals[best_idx])
    worst_y = all_y[best_idx].copy()

    return ISResult(
        pi_hat=pi_hat,
        pi_upper=pi_upper,
        ess=ess,
        n_samples=n_samples,
        n_in_U=int(in_U.sum()),
        detected_unsafe=bool(in_U.any()),
        final_phi=final_phi,
        worst_y=worst_y,
        lambda_tilt=float(lambda_tilt),
    )


@dataclass
class ISSpecResult:
    """Aggregate IS result for a full AND-of-OR-of-AND spec.

    Mirrors ``AMLSSpecResult``. The spec is UNSAT-disjoint iff every
    group is disjoint, and a group is disjoint iff every member
    HalfSpace is disjoint. ``unsat_certified`` is True iff at least
    one group has no detected HalfSpace; ``detected_any`` is True iff
    any HalfSpace in any group was detected.
    """
    unsat_certified: bool
    detected_any: bool
    per_hs_results: list  # list[list[ISResult]]
    spec_groups: list


def is_tilted_certify_spec(
    flow_ode,
    spec_groups,
    *,
    n_samples: int = 2000,
    lambda_tilt: float = 5.0,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
) -> ISSpecResult:
    """Run tilted IS per HalfSpace across an AND-of-OR-of-AND spec.

    Mirrors :func:`amls_certify_spec`: outer AND across groups, inner OR
    within each group. A group is disjoint iff every HalfSpace in it is
    disjoint (no detection); the spec is UNSAT iff at least one group is
    disjoint. Detection in any HalfSpace is the operationally important
    output -- it flips the verdict to UNKNOWN.
    """
    per_hs_results = []
    detected_any = False
    h_idx = 0
    for group in spec_groups:
        group_results = []
        for hs in group:
            sub_seed = (
                None if seed is None
                else (int(seed) + h_idx * 7919) & 0x7FFFFFFF
            )
            r = is_tilted_estimate_halfspace_mass(
                flow_ode=flow_ode, halfspace=hs,
                n_samples=n_samples, lambda_tilt=lambda_tilt,
                beta=beta, seed=sub_seed,
                t=t, n_ode_steps=n_ode_steps,
                ode_method=ode_method,
                ode_atol=ode_atol, ode_rtol=ode_rtol,
            )
            group_results.append(r)
            h_idx += 1
            if r.detected_unsafe:
                detected_any = True
        per_hs_results.append(group_results)

    unsat_certified = False
    for group_results in per_hs_results:
        if all(not r.detected_unsafe for r in group_results):
            unsat_certified = True
            break

    return ISSpecResult(
        unsat_certified=unsat_certified,
        detected_any=detected_any,
        per_hs_results=per_hs_results,
        spec_groups=spec_groups,
    )
