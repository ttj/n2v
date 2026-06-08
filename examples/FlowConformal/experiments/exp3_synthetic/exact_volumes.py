"""Exact (or MC ground-truth) reach-set volume for synthetic Exp 3 nets.

For an identity-activation 1-Lipschitz network (purely linear), the
reach set of an axis-aligned input box ``[lb, ub]`` is the zonotope
image and its volume is exactly:

    vol(R) = |det(W_total)| * prod(ub - lb)

The 1-alpha conformal subset of the *uniform-input* pushforward sits
inside R with mass exactly 1-alpha. For identity-activation nets the
pushforward is uniform on R as well (the linear map is volume-uniform
modulo the global scale ``|det(W_total)|``), so the (1-alpha)-conformal
subset of R has volume:

    vol_{1-alpha}(R) = (1 - alpha) * vol(R)

The verifier's score-based reach set is then measured against this
floor.

For non-identity activations the reach set is no longer a zonotope and
we fall back to a Monte Carlo bounding-box volume (the smallest
axis-aligned bbox containing all sampled outputs, scaled by the
empirical hit rate at quantile 1-alpha). MC ground truth is loose but
consistent across seeds.
"""
from __future__ import annotations

import numpy as np
import torch


# MC ground-truth reach-set volume for ThreeBlobClassifier3D over input
# box [-1, +1]^3 (== center=0, radius=1.0). Computed offline with N=10M
# Star-union samples; cached at
#   .claude/research/flow-matching-probabilistic-reach/_archive/
#   training_quality_ablation/_exact_volume_cache/
#   ThreeBlobClassifier3D__center_0.000_0.000_0.000__radius_1.000.pkl
# (volume_est = 213.7253; Hoeffding 99% CI half-width = 67.50). The
# (1-alpha) factor is applied at call time so the conformal subset
# volume matches the linear-net convention.
_THREE_BLOB_3D_REACH_VOL = 213.7252667820

# Star-union exact reach-set area for RotatedBananaNet over input
# box [0, 1]^2 (center=(0.5, 0.5), radius=0.5). Computed via 2D
# rasterization on the union of 49 stars. Analytical truth for the
# target map (x1, x2) -> (x1, x1^2 + 0.3*x2) is exactly 0.3; the
# trained net is a small approximation of that.
_TWO_BANANA_REACH_VOL = 0.2951579791


def exact_volume_three_blob_3d(alpha: float = 0.001) -> float:
    """Cached MC reach-set volume for ThreeBlobClassifier3D on [-1, +1]^3.

    Returned value is ``(1 - alpha) * 213.7253``. Only valid for the
    canonical exp3 input box (center=0, radius=1). Re-run
    :func:`examples.FlowConformal.benchmarks._common.exact_star_union_volume`
    if the input box ever changes.
    """
    return float((1.0 - alpha) * _THREE_BLOB_3D_REACH_VOL)


def exact_volume_two_banana(alpha: float = 0.001) -> float:
    """Cached Star-union reach-set area for RotatedBananaNet on [0, 1]^2.

    Returned value is ``(1 - alpha) * 0.2952``. Only valid for the
    canonical exp3 input box.
    """
    return float((1.0 - alpha) * _TWO_BANANA_REACH_VOL)


def exact_volume_linear_net(
    net,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    alpha: float = 0.001,
) -> float:
    """Closed-form volume for an identity-activation OneLipschitzNet.

    Args:
        net: ``OneLipschitzNet`` with ``activation='identity'``.
        input_lb, input_ub: ``(dim,)`` arrays giving the input box.
        alpha: target miscoverage for the conformal subset. The full
            zonotope volume is returned at ``alpha=0`` (no shrinkage);
            at ``alpha>0`` it is multiplied by ``1 - alpha``.

    Returns:
        The closed-form volume ``(1 - alpha) * |det(W_total)| * prod(ub - lb)``.
    """
    if getattr(net, 'activation_name', 'identity') != 'identity':
        raise ValueError(
            "exact_volume_linear_net requires activation='identity'; "
            f"got {net.activation_name!r}. Use mc_ground_truth_volume "
            "for nonlinear nets."
        )
    W_total = net.total_weight().cpu().numpy().astype(np.float64)
    box_volume = float(np.prod(np.asarray(input_ub) - np.asarray(input_lb)))
    det_W = abs(float(np.linalg.det(W_total)))
    return (1.0 - alpha) * det_W * box_volume


def mc_ground_truth_volume(
    net,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    *,
    n_samples: int = 100_000,
    alpha: float = 0.001,
    seed: int = 0,
) -> dict:
    """MC ground-truth volume via bbox-and-mass for a (possibly nonlinear) net.

    Strategy: sample ``n_samples`` inputs uniformly from ``[lb, ub]``,
    push through ``net``, and report:

      - ``volume_bbox``: volume of the smallest axis-aligned bbox
        enclosing the (1-alpha)-quantile of outputs (drop the
        per-dimension top/bottom alpha/2 fractions before bounding).
      - ``volume_full``: volume of the full empirical bbox (no
        quantile clipping).

    The (1-alpha)-quantile bbox is a loose proxy for the reach set's
    smallest-(1-alpha)-mass enclosure. Use this when the closed-form
    linear volume is unavailable (nonlinear activation).

    Returns:
        Dict with keys ``volume``, ``volume_full``, ``alpha``,
        ``n_samples``, ``output_dim``.
    """
    rng = np.random.default_rng(seed)
    dim_in = np.asarray(input_lb).shape[0]
    x = rng.uniform(low=input_lb, high=input_ub, size=(n_samples, dim_in))
    with torch.no_grad():
        y = net(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    # Per-dim quantile bbox: drop alpha/2 from each tail.
    half = alpha / 2.0
    lo = np.quantile(y, half, axis=0)
    hi = np.quantile(y, 1.0 - half, axis=0)
    vol_q = float(np.prod(hi - lo))
    vol_full = float(np.prod(y.max(axis=0) - y.min(axis=0)))
    return {
        'volume': vol_q,
        'volume_full': vol_full,
        'alpha': alpha,
        'n_samples': n_samples,
        'output_dim': y.shape[1],
    }
