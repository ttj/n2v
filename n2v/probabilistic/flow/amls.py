"""Adaptive Multilevel Splitting (AMLS) for rare-event flow-set queries.

Given a calibrated FlowODE and an unsafe polyhedron ``U = { y : G y <= g }``,
AMLS estimates ``P_flow(U)`` by decomposing it as a product of conditional
probabilities at adaptive level sets:

    A_0 := R^d  superset  A_1 := { y : phi(y) <= tau_1 }  superset  ...  superset  A_K = U

with ``phi(y) := max_i (G_i y - g_i)``. By the VNN-LIB convention,
``y in U`` iff ``G y <= g`` element-wise iff ``phi(y) <= 0``; we DESCEND
levels (lower ``tau`` is closer to / inside `U`). Levels ``tau_k`` are
chosen so the empirical fraction of samples crossing into ``A_k`` from
``A_{k-1}`` equals a fixed quantile ``rho`` (default 0.1). MCMC
(random-walk MH in latent space) propagates samples from one level to
the next.

The operationally important output of this module is the BOOLEAN
``detected_unsafe`` field: whether AMLS reached the polyhedron `U` at
all. When True, the original flow-set verification step's UNSAT
certification is invalidated (a witness sample inside `U` exists,
modulo MCMC mixing).

Device handling:
    Both ``amls_estimate_halfspace_mass`` and ``amls_certify_spec`` take
    an optional ``device`` kwarg. When unspecified, AMLS autodetects
    CUDA via ``torch.cuda.is_available()``. The flow ODE module and the
    halfspace tensors are moved to that device once at entry; all
    flow-pushforward, per-sample halfspace evaluations, MCMC proposal
    generation, resampling, and MH accept-reject decisions are batched
    on-device using ``torch.Generator(device=device)`` for randomness.
    No host-device sync occurs in the MCMC hot loop. Per-level scalar
    quantile / threshold computations stay on-device too. The result's
    ``worst_y`` is the only tensor brought back to CPU at the end.

    CPU vs GPU verdicts: when ``device='cpu'`` the run uses torch CPU
    ops with a torch Generator seeded from ``seed``. This is NOT
    bit-identical to the previous numpy-based implementation (different
    RNG streams) but the algorithm and seed determinism are preserved
    — same seed + same device => bit-identical run.

References:
    Au & Beck 2001 -- Subset Simulation.
    Cerou & Guyader 2007 -- Adaptive Multilevel Splitting.
    Webb et al. 2019 -- AMLS for NN robustness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


@dataclass
class AMLSResult:
    """Result of an AMLS run targeting a single HalfSpace polyhedron.

    Attributes:
        pi_hat: Final point estimate of ``P_flow(y in U)``.
        pi_upper: Asymptotic ``(1 - beta)`` upper bound on
            ``P_flow(y in U)`` from the Cerou-Guyader CLT. Use as a
            diagnostic; not a finite-sample certificate.
        levels_used: Number of adaptive levels run (K). When the first
            level already crossed ``tau >= 0`` this equals 1.
        final_unsafe_count: Number of final-level samples inside ``U``
            (i.e. with ``phi(y) >= 0``).
        detected_unsafe: True iff at least one sample landed in ``U`` at
            any point during the run.
        final_phi: MINIMUM ``phi(y)`` observed across all samples seen
            during the run. Non-positive iff ``detected_unsafe``.
        worst_y: ``(d,)`` array — the sample that achieved
            ``final_phi`` (the deepest-into-`U` witness if detected;
            otherwise the closest-to-`U` sample).
    """
    pi_hat: float
    pi_upper: float
    levels_used: int
    final_unsafe_count: int
    detected_unsafe: bool
    final_phi: float
    worst_y: np.ndarray


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Resolve the device kwarg to a concrete ``torch.device``.

    ``None`` autodetects: CUDA when available, else CPU.
    """
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _phi_halfspace_torch(
    y: torch.Tensor,
    G: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Vectorized signed-distance-to-U: max row margin per sample.

    Args:
        y: ``(N, d)`` torch tensor of data points (any dtype/device).
        G: ``(k, d)`` torch tensor of halfspace normals on the same
            device and dtype as ``y``.
        g: ``(k,)`` torch tensor of offsets on the same device/dtype.

    Returns:
        ``(N,)`` tensor of ``max_i (G_i y - g_i)`` per sample.
    """
    # margins: (N, k) = y @ G^T - g
    margins = y @ G.T - g.unsqueeze(0)
    return margins.max(dim=1).values


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
    """Run the inverse flow on a batched latent tensor (on z's device)."""
    with torch.no_grad():
        return flow_ode.inverse(
            z, t=t, n_steps=n_steps, method=method, atol=atol, rtol=rtol,
        )


def amls_estimate_halfspace_mass(
    flow_ode,
    halfspace,
    *,
    n_samples_per_level: int = 1000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSResult:
    """Estimate ``P_flow(y in U)`` for ``U = { y : G y <= g }`` via AMLS.

    Algorithm (Cerou-Guyader 2007 with random-walk MH mutation):

        1. Sample ``N`` latents ``z_0 ~ N(0, I_d)``. Push through the
           flow inverse to obtain ``y_0``.
        2. Compute ``phi(y_i) = max_j (G_j y_i - g_j)``. If min phi <= 0
           we have detected `U` already; return.
        3. Otherwise set ``tau_1`` to the ``rho`` quantile of
           ``phi(y_i)`` (so ``rho * N`` samples lie BELOW ``tau_1``).
        4. Discard samples with ``phi(y_i) > tau_1``. Refill to ``N``
           via random-walk MH in z-space, accepting proposals iff their
           mapped output still satisfies ``phi(y) <= tau_1``.
        5. Repeat until ``tau_k <= 0`` or ``max_levels`` exhausted.

    The returned ``pi_hat`` is ``rho^(K-1) * frac_in_U_at_final_level``.
    ``pi_upper`` is the Cerou-Guyader asymptotic ``1 - beta`` upper
    bound (diagnostic; not a finite-sample certificate).

    Args:
        flow_ode: trained ``FlowODE`` instance.
        halfspace: object with ``.G`` (k, d) and ``.g`` (k, 1) or (k,)
            attributes describing the unsafe polyhedron ``G y <= g``.
        n_samples_per_level: ``N``, samples held at each level.
        quantile: ``rho``, fraction of samples kept after each level
            update. Default 0.1 (the standard subset-simulation choice).
        max_levels: hard cap on number of levels; the run gives up if
            this is reached without detecting U.
        n_mcmc_steps: MCMC steps per replicated point per refill round.
        mcmc_step_size: random-walk MH proposal scale (std of the
            Gaussian step in latent z-space).
        beta: confidence parameter for the asymptotic upper bound.
        seed: RNG seed; controls latent draw and MCMC proposals.
        t, n_ode_steps, ode_method, ode_atol, ode_rtol: passed to
            ``flow_ode.inverse``.
        device: torch device for flow integration, halfspace evaluation,
            MCMC proposal generation, resampling, and MH accept-reject
            decisions. ``None`` autodetects CUDA. The flow ODE and the
            halfspace tensors are moved to this device once at entry,
            and a per-device ``torch.Generator`` seeded from ``seed``
            drives all random draws. Same seed + same device =>
            bit-identical run; the CPU and GPU paths produce
            float-equivalent but NOT bit-identical results (different
            RNG implementations across torch backends).

    Returns:
        ``AMLSResult``.
    """
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

    # Halfspace dimensions; G/g move to-device as float32 tensors.
    G_np = np.asarray(halfspace.G, dtype=np.float64)
    g_np = np.asarray(halfspace.g, dtype=np.float64).flatten()
    dim = G_np.shape[1]
    N = n_samples_per_level
    dtype = torch.float32

    G_t = torch.from_numpy(G_np.astype(np.float32)).to(dev)
    g_t = torch.from_numpy(g_np.astype(np.float32)).to(dev)

    # Move the flow_ode to the chosen device. FlowODE inherits from
    # nn.Module so .to(device) is a no-op or a parameter migration.
    if hasattr(flow_ode, 'to'):
        flow_ode = flow_ode.to(dev)

    # Per-device torch Generator drives ALL randomness (latents, MCMC
    # proposals, MH uniforms, resample indices). No numpy RNG, no
    # host-device sync in the MCMC hot loop.
    seed_int = 0 if seed is None else (int(seed) & 0x7FFFFFFF)
    gen = torch.Generator(device=dev).manual_seed(seed_int)
    if seed is not None:
        torch.manual_seed(seed_int)

    # ---- Level 0: initial sample ----
    z = torch.randn(N, dim, device=dev, dtype=dtype, generator=gen)
    y = _push_through_flow(
        flow_ode, z, t=t, n_steps=n_ode_steps,
        method=ode_method, atol=ode_atol, rtol=ode_rtol,
    )
    phi_t = _phi_halfspace_torch(y, G_t, g_t)

    # On-device "best" tracking: argmin(phi) is a single device op.
    # We keep best_phi/best_y as device tensors and only sync at return.
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
        return AMLSResult(
            pi_hat=pi_hat,
            pi_upper=pi_upper,
            levels_used=1,
            final_unsafe_count=n_in,
            detected_unsafe=True,
            final_phi=best_phi,
            worst_y=best_y,
        )

    # Working buffers stay on `dev` for the entire run. phi_t is (N,)
    # device tensor; z is (N, dim); y is (N, dim).
    tau_prev = math.inf
    K = 0
    for level in range(max_levels):
        K = level + 1

        # rho-quantile of phi (lower tail) — fully on device.
        tau_k_t = torch.quantile(phi_t, quantile)
        tau_k = float(tau_k_t.item())
        # Monotone safeguard: tau must not increase across levels.
        if tau_k >= tau_prev:
            tau_k = tau_prev - 1e-12
            tau_k_t = torch.tensor(tau_k, device=dev, dtype=dtype)

        # If tau_k crossed zero (negative), U has been reached; finish.
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
            return AMLSResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=bool(in_U.any().item()),
                final_phi=best_phi,
                worst_y=best_y,
            )

        # Elite set: phi <= tau_k. Use top-k (smallest) for a
        # deterministic-size kept set; refill via uniform indices.
        keep_count = max(1, int(round(N * quantile)))
        # torch.topk with largest=False gives the keep_count smallest.
        _kept_phi, keep_idx_t = torch.topk(
            phi_t, k=keep_count, largest=False, sorted=False,
        )
        if keep_count == 0:
            break

        # Uniform resample with replacement to refill to N. All on-device.
        sample_indices = torch.randint(
            0, keep_count, (N,), device=dev, generator=gen,
        )
        chosen_t = keep_idx_t[sample_indices]
        z_cur = z.index_select(0, chosen_t).clone()
        y_cur = y.index_select(0, chosen_t).clone()
        phi_cur = phi_t.index_select(0, chosen_t).clone()

        # MCMC mutate each sample; symmetric random-walk in z. ENTIRELY
        # on device, no numpy, no .cpu()/.numpy()/.item() in the hot loop.
        # We track the running best (z, y, phi) and the running min-phi
        # across all proposals using torch.where; no Python-level branch
        # inside the loop forces a host sync.
        if isinstance(tau_k_t, torch.Tensor):
            tau_k_dev = tau_k_t.to(device=dev, dtype=dtype)
        else:
            tau_k_dev = torch.tensor(tau_k, device=dev, dtype=dtype)
        for _step in range(n_mcmc_steps):
            eta = torch.randn(
                N, dim, device=dev, dtype=dtype, generator=gen,
            )
            z_prop = z_cur + mcmc_step_size * eta
            # MH log-ratio for symmetric proposal under N(0, I) target:
            # log alpha = 0.5 * (||z||^2 - ||z'||^2)
            log_alpha = 0.5 * (
                (z_cur * z_cur).sum(dim=1) - (z_prop * z_prop).sum(dim=1)
            )
            u = torch.rand(N, device=dev, dtype=dtype, generator=gen)
            log_u = torch.log(u)
            mh_pass = log_u < log_alpha

            # Batched flow pushforward + halfspace eval on device.
            y_prop = _push_through_flow(
                flow_ode, z_prop, t=t, n_steps=n_ode_steps,
                method=ode_method, atol=ode_atol, rtol=ode_rtol,
            )
            phi_prop = _phi_halfspace_torch(y_prop, G_t, g_t)

            level_pass = phi_prop <= tau_k_dev
            accept = mh_pass & level_pass

            # Track best phi seen so far across all proposals (device-only).
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
            pi_hat = (quantile ** K) * frac_in_U  # K levels passed
            from scipy.stats import norm
            sigma2 = (K + 1) * (1.0 - quantile) / (quantile * N)
            log_pi_upper = math.log(max(pi_hat, 1e-300)) + \
                norm.ppf(1.0 - beta) * math.sqrt(sigma2)
            pi_upper = math.exp(log_pi_upper)
            best_phi = float(best_phi_t.item())
            best_y = best_y_t.detach().cpu().numpy().astype(np.float64)
            return AMLSResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K + 1,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=True,
                final_phi=best_phi,
                worst_y=best_y,
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
    return AMLSResult(
        pi_hat=pi_hat,
        pi_upper=pi_upper,
        levels_used=K,
        final_unsafe_count=0,
        detected_unsafe=bool(best_phi <= 0.0),
        final_phi=best_phi,
        worst_y=best_y,
    )


@dataclass
class AMLSSpecResult:
    """Aggregate AMLS result for a full AND-of-OR-of-AND spec.

    The spec is UNSAT-disjoint iff every group is disjoint, and a group
    is disjoint iff every member HalfSpace is disjoint. AMLS is run per
    HalfSpace; ``unsat_certified`` is True iff NO HalfSpace was detected
    inside the flow set.

    Note: AMLS does NOT give a finite-sample probabilistic certificate
    (the bound is asymptotic). When ``unsat_certified=True`` we cannot
    claim the same UNSAT-with-(eps, delta) statement that scenario gives
    — the result is more accurately phrased as "no AMLS run detected
    a witness in the unsafe region". Callers should treat this as a
    detector, not a certifier; the verdict is UNKNOWN unless every
    AMLS run terminated cleanly with detected_unsafe=False AND scenario
    bounds are layered on top in a future Phase B.
    """
    unsat_certified: bool
    detected_any: bool
    per_hs_results: list  # list[list[AMLSResult]]
    spec_groups: list


def amls_certify_spec(
    flow_ode,
    spec_groups,
    *,
    n_samples_per_level: int = 1000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSSpecResult:
    """Run AMLS per HalfSpace across an AND-of-OR-of-AND spec.

    Mirrors the layer-3 dispatch of ``certify_spec_disjoint``: outer
    AND across groups, inner OR within each group, AND across rows
    inside each HalfSpace (the joint G y <= g check is what AMLS
    estimates).

    A group is disjoint iff every HalfSpace in it is disjoint. The spec
    is UNSAT iff at least one group is disjoint. AMLS *detects*
    membership; UNSAT certification is provisional (no finite-sample
    bound) — callers should treat detection in any HalfSpace as a flip
    to UNKNOWN.

    Args:
        device: torch device threaded into each per-HalfSpace AMLS run.
            ``None`` autodetects CUDA.
    """
    # Resolve device once and forward as-is to each per-HalfSpace call.
    dev = _resolve_device(device)

    per_hs_results = []
    detected_any = False
    # Use a different seed per HalfSpace so each AMLS run is decoupled.
    h_idx = 0
    for group in spec_groups:
        group_results = []
        for hs in group:
            sub_seed = (
                None if seed is None
                else (int(seed) + h_idx * 7919) & 0x7FFFFFFF
            )
            r = amls_estimate_halfspace_mass(
                flow_ode=flow_ode, halfspace=hs,
                n_samples_per_level=n_samples_per_level,
                quantile=quantile, max_levels=max_levels,
                n_mcmc_steps=n_mcmc_steps,
                mcmc_step_size=mcmc_step_size,
                beta=beta, seed=sub_seed,
                t=t, n_ode_steps=n_ode_steps,
                ode_method=ode_method,
                ode_atol=ode_atol, ode_rtol=ode_rtol,
                device=dev,
            )
            group_results.append(r)
            h_idx += 1
            if r.detected_unsafe:
                detected_any = True
        per_hs_results.append(group_results)

    # UNSAT iff at least one group has all HalfSpaces *not* detected.
    # i.e. some group is fully disjoint. This is the same OR-of-disjoint-
    # group early-exit logic the scenario dispatcher uses, but applied to
    # the (negated) detection flag.
    unsat_certified = False
    for group_results in per_hs_results:
        if all(not r.detected_unsafe for r in group_results):
            unsat_certified = True
            break

    return AMLSSpecResult(
        unsat_certified=unsat_certified,
        detected_any=detected_any,
        per_hs_results=per_hs_results,
        spec_groups=spec_groups,
    )
