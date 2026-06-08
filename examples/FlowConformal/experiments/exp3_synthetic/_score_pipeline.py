"""Calibration + verdict + volume pipeline for Exp 3 score families.

Each Exp 3 nonconformity score family produces a different shape of
predicted reach set ``R_q = {y : score(y) <= q}``:

* ``hyperrect`` — axis-aligned box. Closed-form volume, closed-form
  halfspace disjointness via interval propagation.
* ``ellipsoid`` (Mahalanobis) — ellipsoid. Closed-form volume,
  closed-form halfspace disjointness via
  ``min_{y in E} g^T y = g^T mu - q sqrt(g^T Sigma g)``.
* ``gmm`` — naive GMM (per-component max log-likelihood, k=3). The
  resulting level set is a *union* of per-component ellipsoids
  ``E_i = {y : (y-mu_i)^T Sigma_i^-1 (y-mu_i) <= q_i^2}``, where
  ``q_i^2 = 2 (q + log w_i + log A_i)`` and
  ``A_i = (2 pi)^(-d/2) det(Sigma_i)^(-1/2)``. Disjointness with a
  halfspace holds iff every E_i is disjoint; volume is reported as
  the union-bound sum of ellipsoid volumes (conservative upper
  bound; documented as such).
* ``flow`` — current default. Delegates to
  :func:`examples.FlowConformal.experiments._shared_flow_runner.run_flow_pipeline`
  (which wraps :func:`n2v.probabilistic.flow_reach` and
  :func:`n2v.utils.verify_specification.verify_specification`).

All four score families produce comparable artifacts: closed-form (or
union-bound) volume, exact disjointness on Lipschitz-friendly specs,
empirical coverage on a held-out calibration split. ``flow`` is the
only family that uses MCMC for verdict (bounded AMLS); the other
three are closed-form.

This module provides a single :func:`run_score_pipeline` entry point
called from :mod:`exp3_run_ours`.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from examples.FlowConformal.experiments.baselines._common import (
    halfspace_disjoint_from_box,
)
from n2v.sets.halfspace import HalfSpace


def _exact_volume_lipschitz(network, lb: np.ndarray, ub: np.ndarray) -> float:
    """Closed-form reach-set volume for an identity-activation
    OneLipschitzNet on input box ``[lb, ub]``, with a cached MC fallback
    for known nonlinear classifiers.

    Linear identity-activation case: the composed map is
    ``f(x) = W_total @ x`` and the reach set is the parallelotope
    image with volume ``|det(W_total)| * prod(ub - lb)``.

    Nonlinear fallback: if the class name matches a known cached
    ground truth (``ThreeBlobClassifier3D`` on ``[-1, +1]^3``), return
    that. Otherwise return nan.
    """
    if (network.__class__.__name__ == 'ThreeBlobClassifier3D'
            and np.allclose(lb, -1.0) and np.allclose(ub, 1.0)):
        from examples.FlowConformal.experiments.exp3_synthetic.exact_volumes import (
            exact_volume_three_blob_3d,
        )
        return exact_volume_three_blob_3d(alpha=0.0)
    if (network.__class__.__name__ == 'RotatedBananaNet'
            and np.allclose(lb, 0.0) and np.allclose(ub, 1.0)):
        from examples.FlowConformal.experiments.exp3_synthetic.exact_volumes import (
            exact_volume_two_banana,
        )
        return exact_volume_two_banana(alpha=0.0)
    if not hasattr(network, 'W_list'):
        return float('nan')
    if getattr(network, 'activation_name', None) != 'identity':
        return float('nan')
    W_total = None
    for W in network.W_list:
        Wn = W.detach().cpu().numpy().astype(np.float64)
        W_total = Wn if W_total is None else Wn @ W_total
    if W_total is None:
        return float('nan')
    return float(abs(np.linalg.det(W_total)) * np.prod(ub - lb))


def _hyperrect_score_bounds(y_train: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-axis center and scale for the hyperrect score.

    score(y) = max_i |y_i - center_i| / scale_i

    Center is the mean of the training set; scale is the standard
    deviation, clipped to a small positive minimum to avoid zero.
    """
    center = y_train.mean(axis=0)
    scale = y_train.std(axis=0)
    scale = np.maximum(scale, 1e-8)
    return center, scale


def _hyperrect_q(y_calib: np.ndarray, center: np.ndarray,
                  scale: np.ndarray, alpha: float) -> float:
    """Conformal q: the (1-α) quantile of hyperrect scores on calib set."""
    scores = np.max(np.abs(y_calib - center) / scale, axis=1)
    # Hashemi double-step ell: ceil((m+1)*(1-alpha)).
    m = scores.shape[0]
    ell = max(1, min(m, int(np.ceil((m + 1) * (1 - alpha)))))
    sorted_scores = np.sort(scores)
    return float(sorted_scores[ell - 1])


def _run_hyperrect(network, input_lb: np.ndarray, input_ub: np.ndarray,
                   spec, *, n_train: int, alpha: float,
                   seed: int) -> Dict[str, Any]:
    """End-to-end hyperrect-score pipeline."""
    rng = np.random.default_rng(seed)
    t_start = time.time()

    # 1. Sample calibration data.
    t0 = time.time()
    x = rng.uniform(input_lb, input_ub,
                     size=(n_train, input_lb.size)).astype(np.float32)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        y = network(x_t).cpu().numpy()
    train_s = time.time() - t0  # all calibration cost; no flow training

    # 2. Calibrate.
    t0 = time.time()
    half = n_train // 2
    y_train, y_calib = y[:half], y[half:]
    center, scale = _hyperrect_score_bounds(y_train)
    q = _hyperrect_q(y_calib, center, scale, alpha)
    box_lb = (center - q * scale).astype(np.float64)
    box_ub = (center + q * scale).astype(np.float64)

    # 3. Verdict via halfspace-vs-box disjointness (closed-form, no
    # falsification: Exp 3's design is "no falsifier" so the verdict is
    # purely geometric).
    cex_x_str = ''
    cex_y_str = ''
    disjoint = halfspace_disjoint_from_box(spec, box_lb, box_ub)
    verdict = 'UNSAT' if disjoint is True else 'UNKNOWN'
    verify_s = time.time() - t0

    # 4. Volumes.
    volume_estimate = float(np.prod(box_ub - box_lb))
    volume_exact = _exact_volume_lipschitz(
        network, np.asarray(input_lb), np.asarray(input_ub))
    volume_ratio = (volume_estimate / volume_exact
                     if math.isfinite(volume_exact) and volume_exact > 0
                     else float('nan'))

    total_s = time.time() - t_start
    return {
        'verdict': verdict,
        'q': q,
        'wall_s': total_s,
        'train_s': train_s,
        'verify_s': verify_s,
        'volume_estimate': volume_estimate,
        'volume_exact': volume_exact,
        'volume_ratio': volume_ratio,
        'cex_x': cex_x_str, 'cex_y': cex_y_str,
        'box_lb': box_lb.tolist(), 'box_ub': box_ub.tolist(),
    }


def _ellipsoid_volume(q: float, Sigma: np.ndarray) -> float:
    """Volume of the ellipsoid ``{y : (y-mu).T Sigma_inv (y-mu) <= q^2}``."""
    from scipy.special import gammaln
    d = Sigma.shape[0]
    if q <= 0.0:
        return 0.0
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return float('nan')
    log_vol = (
        d * math.log(q)
        + (d / 2.0) * math.log(math.pi)
        - gammaln(d / 2.0 + 1.0)
        + 0.5 * logdet
    )
    return float(math.exp(log_vol))


def _ellipsoid_excludes_halfspace(
    mu: np.ndarray, Sigma: np.ndarray, q: float, hs: HalfSpace,
) -> bool:
    """Closed-form ellipsoid-vs-AND-of-halfspace-rows disjointness check.

    The ellipsoid ``E = {y : (y-mu)^T Sigma^-1 (y-mu) <= q^2}`` is
    DISJOINT from the unsafe AND-conjunction ``G y <= g`` iff at least
    one row ``g_i^T y <= g_i`` is violated everywhere on E, i.e.

        min_{y in E} g_i^T y > g_i
        ⇔ g_i^T mu - q * sqrt(g_i^T Sigma g_i) > g_i.
    """
    G = np.asarray(hs.G, dtype=np.float64)
    g = np.asarray(hs.g, dtype=np.float64).flatten()
    if G.shape[1] != mu.size:
        return False
    GSGt = np.einsum('ij,jk,ik->i', G, Sigma, G)  # row-wise g_i^T Sigma g_i
    GSGt = np.maximum(GSGt, 0.0)                  # guard tiny negatives
    g_mu = G @ mu
    row_min = g_mu - q * np.sqrt(GSGt)
    return bool(np.any(row_min > g + 1e-9))


def _set_excludes_unsafe(
    excludes_halfspace_fn,
    spec,
) -> bool:
    """Generic disjointness wrapper that mirrors
    :func:`halfspace_disjoint_from_box` but takes a callable predicate
    ``excludes_halfspace_fn(hs) -> bool`` (True iff the *predicted reach
    set* is fully disjoint from the AND-conjunction encoded by ``hs``).

    Returns True iff the predicted set is disjoint from EVERY unsafe
    disjunct in ``spec``. Mirrors the spec-shape dispatch in
    :func:`halfspace_disjoint_from_box`.
    """
    if isinstance(spec, HalfSpace):
        return excludes_halfspace_fn(spec)
    if isinstance(spec, list) and len(spec) > 0:
        first = spec[0]
        if isinstance(first, HalfSpace):
            return all(excludes_halfspace_fn(hs) for hs in spec)
        if isinstance(first, dict):
            for group in spec:
                disjunct = group.get('Hg', None)
                if disjunct is None:
                    return False
                if isinstance(disjunct, HalfSpace):
                    if not excludes_halfspace_fn(disjunct):
                        return False
                elif isinstance(disjunct, list):
                    if all(isinstance(d, HalfSpace) for d in disjunct):
                        if not all(excludes_halfspace_fn(d) for d in disjunct):
                            return False
                    else:
                        return False
                else:
                    return False
            return True
    return False


def _run_ellipsoid(
    network, input_lb: np.ndarray, input_ub: np.ndarray, spec, *,
    n_train: int, alpha: float, seed: int,
) -> Dict[str, Any]:
    """End-to-end Mahalanobis-ellipsoid score pipeline.

    Mirrors :func:`_run_hyperrect` step-for-step (calibration split,
    conformal q on the held-out half, closed-form volume, closed-form
    disjointness) so the geometric advantage of ellipsoid over
    hyperrect is the *only* difference visible in the output CSV.
    """
    rng = np.random.default_rng(seed)
    t_start = time.time()

    t0 = time.time()
    x = rng.uniform(input_lb, input_ub,
                    size=(n_train, input_lb.size)).astype(np.float32)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        y = network(x_t).cpu().numpy()
    train_s = time.time() - t0

    t0 = time.time()
    half = n_train // 2
    y_train, y_calib = y[:half], y[half:]
    mu = y_train.mean(axis=0).astype(np.float64)
    Sigma = np.cov(y_train, rowvar=False).astype(np.float64)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)
    # Ridge for numerical stability: prevents det(Sigma)=0 on rank-deficient
    # output spaces (e.g. when the network's image is lower-dim).
    Sigma = Sigma + 1e-8 * np.eye(Sigma.shape[0])
    Sigma_inv = np.linalg.inv(Sigma)
    diffs = y_calib - mu
    scores = np.sqrt(np.einsum('ij,jk,ik->i', diffs, Sigma_inv, diffs))
    n_calib = scores.shape[0]
    ell = max(1, int(math.ceil((1 - alpha) * (n_calib + 1))))
    sorted_scores = np.sort(scores)
    q = float(sorted_scores[min(ell, n_calib) - 1])

    # Closed-form ellipsoid-vs-halfspace disjointness (no falsification;
    # Exp 3's design is "no falsifier" so the verdict is purely geometric).
    cex_x_str = ''
    cex_y_str = ''
    excludes = _set_excludes_unsafe(
        lambda hs: _ellipsoid_excludes_halfspace(mu, Sigma, q, hs),
        spec,
    )
    verdict = 'UNSAT' if excludes else 'UNKNOWN'
    verify_s = time.time() - t0

    volume_estimate = _ellipsoid_volume(q, Sigma)
    volume_exact = _exact_volume_lipschitz(
        network, np.asarray(input_lb), np.asarray(input_ub))
    volume_ratio = (volume_estimate / volume_exact
                    if math.isfinite(volume_exact) and volume_exact > 0
                    else float('nan'))

    return {
        'verdict': verdict,
        'q': q,
        'wall_s': time.time() - t_start,
        'train_s': train_s,
        'verify_s': verify_s,
        'volume_estimate': volume_estimate,
        'volume_exact': volume_exact,
        'volume_ratio': volume_ratio,
        'cex_x': cex_x_str, 'cex_y': cex_y_str,
    }


def _run_gmm(
    network, input_lb: np.ndarray, input_ub: np.ndarray, spec, *,
    n_train: int, alpha: float, seed: int,
    n_components: int = 3,
) -> Dict[str, Any]:
    """Naive-GMM score pipeline (per-component max log-likelihood).

    Score: ``-log max_i [w_i N(y; mu_i, Sigma_i)]``. The level set
    ``{y : score(y) <= q}`` is the union over components of ellipsoids
    ``E_i = {y : (y-mu_i)^T Sigma_i^-1 (y-mu_i) <= q_i^2}`` with
    per-component radius ``q_i = sqrt(2 (q + log w_i + log A_i))``,
    ``A_i = (2 pi)^(-d/2) det(Sigma_i)^(-1/2)``.

    Disjointness: union ∩ unsafe = ∅ ⇔ every E_i ∩ unsafe = ∅. Volume:
    union-bound (sum of E_i volumes) — conservative upper bound.

    The k=3 default is the established Exp 3 setting; ``n_components``
    lets the test path probe other values.
    """
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    t_start = time.time()

    t0 = time.time()
    x = rng.uniform(input_lb, input_ub,
                    size=(n_train, input_lb.size)).astype(np.float32)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        y = network(x_t).cpu().numpy()
    train_s = time.time() - t0

    t0 = time.time()
    half = n_train // 2
    y_train, y_calib = y[:half], y[half:]

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=seed,
        reg_covar=1e-6,
    )
    gmm.fit(y_train)

    weights = gmm.weights_.astype(np.float64)
    means = gmm.means_.astype(np.float64)
    covs = gmm.covariances_.astype(np.float64)
    d = means.shape[1]

    # Per-component log A_i = -d/2 * log(2 pi) - 1/2 * log(det Sigma_i)
    sign_logdets = [np.linalg.slogdet(C) for C in covs]
    if any(s <= 0 for (s, _) in sign_logdets):
        # Singular covariance — return UNKNOWN with NaN volume.
        return {
            'verdict': 'UNKNOWN',
            'q': float('nan'),
            'wall_s': time.time() - t_start,
            'train_s': train_s,
            'verify_s': time.time() - t0,
            'volume_estimate': float('nan'),
            'volume_exact': _exact_volume_lipschitz(
                network, np.asarray(input_lb), np.asarray(input_ub)),
            'volume_ratio': float('nan'),
            'cex_x': '', 'cex_y': '',
            'error': 'gmm covariance singular',
        }
    log_dets = np.array([ld for (_, ld) in sign_logdets])
    log_A = -0.5 * d * math.log(2.0 * math.pi) - 0.5 * log_dets

    # score(y) = -log max_i [w_i N(y; mu_i, Sigma_i)]
    cov_invs = np.array([np.linalg.inv(C) for C in covs])
    diffs = y_calib[:, None, :] - means[None, :, :]   # (n, k, d)
    quad = np.einsum('nki,kij,nkj->nk', diffs, cov_invs, diffs)
    log_phi = (np.log(weights)[None, :] + log_A[None, :]
               - 0.5 * quad)                          # log w_i N(y; ...)
    scores = -np.max(log_phi, axis=1)
    n_calib = scores.shape[0]
    ell = max(1, int(math.ceil((1 - alpha) * (n_calib + 1))))
    sorted_scores = np.sort(scores)
    q = float(sorted_scores[min(ell, n_calib) - 1])

    # Per-component ellipsoid radii q_i and centres + shapes.
    component_q2 = 2.0 * (q + np.log(weights) + log_A)  # may be negative
    component_active = component_q2 > 0.0
    component_q = np.sqrt(np.where(component_active, component_q2, 0.0))

    # Union-of-ellipsoids vs halfspace: union ∩ unsafe = ∅ ⇔ every
    # component E_i ∩ unsafe = ∅ (closed-form per-component support
    # function; no falsification per Exp 3's "no falsifier" design).
    cex_x_str = ''
    cex_y_str = ''
    def _excludes(hs: HalfSpace) -> bool:
        for k in range(n_components):
            if not component_active[k]:
                continue  # empty component cannot intersect
            if not _ellipsoid_excludes_halfspace(
                means[k], covs[k], component_q[k], hs,
            ):
                return False
        return True
    excludes = _set_excludes_unsafe(_excludes, spec)
    verdict = 'UNSAT' if excludes else 'UNKNOWN'
    verify_s = time.time() - t0

    # Volume: union-bound sum over per-component ellipsoid volumes.
    volume_estimate = float(sum(
        _ellipsoid_volume(component_q[k], covs[k])
        for k in range(n_components) if component_active[k]
    ))
    volume_exact = _exact_volume_lipschitz(
        network, np.asarray(input_lb), np.asarray(input_ub))
    volume_ratio = (volume_estimate / volume_exact
                    if math.isfinite(volume_exact) and volume_exact > 0
                    else float('nan'))

    return {
        'verdict': verdict,
        'q': q,
        'wall_s': time.time() - t_start,
        'train_s': train_s,
        'verify_s': verify_s,
        'volume_estimate': volume_estimate,
        'volume_exact': volume_exact,
        'volume_ratio': volume_ratio,
        'cex_x': cex_x_str, 'cex_y': cex_y_str,
    }


def run_score_pipeline(
    network,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec,
    *,
    score: str,
    n_train: int,
    alpha: float,
    seed: int,
    flow_epochs: int = 2_000,
    flow_config: str = 'base',
    scenario_n_samples: int = 2_000,
    scenario_beta: float = 0.001,
    volume_m: int = 8_000,
    volume_ell: int = 7_999,
    volume_n_samples: int = 200_000,
) -> Dict[str, Any]:
    """Run the named score family end-to-end on one Exp 3 instance.

    Returns dict with keys: ``verdict``, ``q``, ``wall_s``, ``train_s``,
    ``verify_s``, ``volume_estimate``, ``volume_exact``,
    ``volume_ratio``, ``cex_x``, ``cex_y``. May include score-specific
    extras under ``score_extras``.

    ``flow`` delegates to the shared three-stage runner
    (:func:`run_flow_pipeline`) on the bounded-AMLS path, so the
    existing ``--score flow`` smoke output is preserved.
    """
    if score == 'flow':
        # Delegate to the shared three-stage runner (new public API).
        from examples.FlowConformal.experiments._shared_flow_runner import (
            run_flow_pipeline,
        )
        cfg = dict(
            alpha=alpha,
            flow_config=flow_config,
            n_train=n_train,
            flow_epochs=flow_epochs,
            scenario_n_samples=scenario_n_samples,
            verification_method='amls_bounded',
            amls_max_levels=30,
            use_falsifier=False,
        )
        r = run_flow_pipeline(
            network,
            np.asarray(input_lb), np.asarray(input_ub),
            spec, cfg, seed=seed,
        )
        # MC-estimate the calibrated reach-set volume directly from the
        # ProbabilisticSet returned by the shared runner (which exposes
        # ``prob_set`` for downstream consumers like this volume path).
        volume_estimate = float('nan')
        pset = r.get('prob_set')
        if pset is not None and r['verdict'] in ('UNSAT', 'UNKNOWN'):
            try:
                from n2v.probabilistic.flow.sampling import sample_box
                lb_t = torch.as_tensor(input_lb, dtype=torch.float32)
                ub_t = torch.as_tensor(input_ub, dtype=torch.float32)
                x = sample_box(lb_t, ub_t, n_samples=4_000,
                               seed=seed + 12345)
                with torch.no_grad():
                    y = network(x)
                y_lo = y.min(dim=0).values
                y_hi = y.max(dim=0).values
                pad = 0.05 * (y_hi - y_lo).clamp(min=1e-6)
                bbox = (y_lo - pad, y_hi + pad)
                vol, _se = pset.estimate_volume(
                    n_samples=volume_n_samples, bounding_box=bbox,
                )
                volume_estimate = float(vol)
            except Exception:
                volume_estimate = float('nan')

        volume_exact = _exact_volume_lipschitz(
            network, np.asarray(input_lb), np.asarray(input_ub))
        volume_ratio = (
            volume_estimate / volume_exact
            if (math.isfinite(volume_estimate)
                and math.isfinite(volume_exact)
                and volume_exact > 0)
            else float('nan')
        )
        return {
            'verdict': r['verdict'],
            'q': r.get('q'),
            'wall_s': r.get('total_time_s'),
            'train_s': r.get('flow_train_time_s'),
            'verify_s': r.get('verification_time_s'),
            'volume_estimate': volume_estimate,
            'volume_exact': volume_exact,
            'volume_ratio': volume_ratio,
            'cex_x': '', 'cex_y': '',
            'epsilon_total': r.get('epsilon_total'),
            'delta_total': r.get('delta_total'),
            'amls_levels_used': r.get('amls_levels_used'),
            'amls_bounded_eps_2_upper': r.get('amls_bounded_eps_2_upper'),
        }
    if score == 'hyperrect':
        return _run_hyperrect(
            network,
            np.asarray(input_lb), np.asarray(input_ub), spec,
            n_train=n_train, alpha=alpha, seed=seed,
        )
    if score == 'ellipsoid':
        return _run_ellipsoid(
            network,
            np.asarray(input_lb), np.asarray(input_ub), spec,
            n_train=n_train, alpha=alpha, seed=seed,
        )
    if score == 'gmm':
        return _run_gmm(
            network,
            np.asarray(input_lb), np.asarray(input_ub), spec,
            n_train=n_train, alpha=alpha, seed=seed,
        )
    raise ValueError(f'unknown score: {score!r}')
