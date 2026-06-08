"""
Nonconformity score functions for flow-based conformal reachability.

Each score maps a batch of output vectors to non-negative scalars.
The sublevel set {y : score(y) <= q} defines the prediction region.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class NonconformityScore:
    """Base class for nonconformity score functions.

    All scores: (batch, d) -> (batch,) non-negative scalars.
    """

    def __call__(self, y: Tensor) -> Tensor:
        raise NotImplementedError


class HyperrectScore(NonconformityScore):
    """Hyperrectangular score: max_k |y_k - c_k| / tau_k.

    The sublevel set is an axis-aligned box.
    This is the naive score from Hashemi et al. 2025.

    Args:
        center: (d,) center of the score function.
        scales: (d,) per-dimension normalization factors.
    """

    def __init__(self, center: Tensor, scales: Tensor):
        self.center = center
        self.scales = scales

    def __call__(self, y: Tensor) -> Tensor:
        return ((y - self.center).abs() / self.scales).max(dim=1).values

    def sublevel_set_volume(self, q: Tensor) -> float:
        """Closed-form volume of {y : score(y) <= q}."""
        return (2 * q * self.scales).prod().item()


class EllipsoidScore(NonconformityScore):
    """Mahalanobis score: sqrt((y-c)^T Sigma^{-1} (y-c)).

    The sublevel set is an ellipsoid.

    Args:
        center: (d,) center.
        cov_inv: (d, d) inverse covariance matrix.
    """

    def __init__(self, center: Tensor, cov_inv: Tensor):
        self.center = center
        self.cov_inv = cov_inv

    def __call__(self, y: Tensor) -> Tensor:
        diff = y - self.center
        return (diff @ self.cov_inv * diff).sum(dim=1).sqrt()


class BallScore(NonconformityScore):
    """L2 ball score: ||y - c||_2.

    The sublevel set is a Euclidean ball.

    Args:
        center: (d,) center.
    """

    def __init__(self, center: Tensor):
        self.center = center

    def __call__(self, y: Tensor) -> Tensor:
        return (y - self.center).norm(dim=1)


class FlowScore(NonconformityScore):
    """Flow-based score: ||phi_t(y)||_2.

    The sublevel set follows the geometry of the learned flow.

    Args:
        flow_model: Object with forward(y, t, ...) method (e.g., FlowODE).
        t: Flow time parameter (default 1.0).
        n_steps: Number of ODE integration steps.
        method: ODE solver name passed through to flow_model.forward.
            'dopri5' for adaptive (default), 'rk4'/'euler' for fast
            fixed-step inference.
        batch_size: if not None, chunk the incoming y into this size and
            concatenate — lets callers evaluate very large batches (e.g.
            MC volume) without OOM.
    """

    def __init__(self, flow_model, t: float = 1.0, n_steps: int = 100,
                 method: str = 'dopri5', batch_size: int | None = None,
                 atol: float = 1e-5, rtol: float = 1e-5):
        self.flow_model = flow_model
        self.t = t
        self.n_steps = n_steps
        self.method = method
        self.batch_size = batch_size
        self.atol = atol
        self.rtol = rtol

    def _flow_device(self) -> torch.device | None:
        """Return the device of the underlying velocity field, if any."""
        vf = getattr(self.flow_model, 'velocity_field', self.flow_model)
        try:
            return next(vf.parameters()).device
        except (StopIteration, AttributeError):
            return None

    def _integrate(self, y: Tensor) -> Tensor:
        return self.flow_model.forward(
            y, t=self.t, n_steps=self.n_steps, method=self.method,
            atol=self.atol, rtol=self.rtol,
        )

    def __call__(self, y: Tensor) -> Tensor:
        dev = self._flow_device()
        src_device = y.device
        if dev is not None and y.device != dev:
            y = y.to(dev)
        if self.batch_size is None or y.shape[0] <= self.batch_size:
            out = self._integrate(y).norm(dim=1)
        else:
            outs = []
            for i in range(0, y.shape[0], self.batch_size):
                outs.append(self._integrate(y[i:i + self.batch_size]).norm(dim=1))
            out = torch.cat(outs, dim=0)
        if dev is not None and out.device != src_device:
            out = out.to(src_device)
        return out

    def set_t(self, t: float):
        """Update the flow time parameter."""
        self.t = t


class GMMScore(NonconformityScore):
    """Negative log-likelihood under a fitted Gaussian Mixture Model.

    score(y) = -log p_GMM(y) where p_GMM(y) = sum_k pi_k N(y | mu_k, Sigma_k).
    The sublevel set {y : score(y) <= q} is the high-density region of
    the GMM. Use ``GMMScore.fit(y_train, n_components)`` to fit a GMM
    to a training set and wrap it as a score function.

    Args:
        gmm: a fitted ``sklearn.mixture.GaussianMixture`` instance.
    """

    def __init__(self, gmm):
        self.gmm = gmm

    @classmethod
    def fit(
        cls,
        y_train,
        n_components: int = 10,
        *,
        covariance_type: str = 'full',
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        random_state: int = 0,
    ) -> 'GMMScore':
        """Fit a GaussianMixture to ``y_train`` and return a GMMScore."""
        from sklearn.mixture import GaussianMixture
        if isinstance(y_train, torch.Tensor):
            y_np = y_train.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_train)
        gmm = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type,
            reg_covar=reg_covar, max_iter=max_iter, random_state=random_state,
        )
        gmm.fit(y_np)
        return cls(gmm)

    def __call__(self, y: Tensor) -> Tensor:
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
            device, dtype = y.device, y.dtype
        else:
            y_np = np.asarray(y)
            device, dtype = None, torch.float32
        logp = self.gmm.score_samples(y_np)  # log p(y), shape (N,)
        out = torch.as_tensor(-logp, dtype=dtype)
        if device is not None:
            out = out.to(device)
        return out



# ---- Pipeline-glue score decorators -----------------------------------
#
# These two classes wrap an existing :class:`FlowScore` (or any
# score-fn-compatible callable) with extra behaviour useful in the
# flow-matching reachability pipeline. They are not the algorithm itself —
# they are composition tools used by the reach + verify dispatch.


class _WhiteningFlowScore:
    """Score function that whitens its input before delegating.

    Lets callers (e.g. volume validation) keep passing raw network
    outputs: whitening happens transparently before the underlying
    :class:`FlowScore` operates.
    """

    def __init__(self, base_score_fn, y_mean: torch.Tensor,
                 y_std: torch.Tensor):
        self.base = base_score_fn
        self.y_mean = y_mean
        self.y_std = y_std

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        dev = y.device
        y_w = (y - self.y_mean.to(dev)) / self.y_std.to(dev)
        return self.base(y_w)

    @property
    def flow_model(self):
        return self.base.flow_model


