"""Synthetic networks for Experiment 3 with exact-computable reach.

We use **identity-activation 1-Lipschitz networks** so that the composed
map is purely linear:

    f(x) = W_total @ x        with  W_total = W_L @ ... @ W_1

Each W_j is constructed by SVD-clipping a random matrix to spectral
norm <= 1. The reach set under an axis-aligned input box [lb, ub] is
then the exact zonotope (parallelotope) image:

    R = { W_total @ x : x in [lb, ub] }

The volume of the parallelotope is closed-form:

    vol(R) = |det(W_total)| * prod(ub - lb)

(See ``exact_volumes.exact_volume_linear_net``.) Identity activation
also lets us bypass MC ground-truth volume estimation in the linear
regime — handy as a sanity check before scaling to the volume-MC
estimator on the same network.

If you swap ``activation`` to ``'tanh'`` or ``'silu'`` the network is
still 1-Lipschitz (both activations have Lipschitz constant 1) but the
exact reach-set volume is no longer closed-form and you must fall back
to MC ground truth (see ``exact_volumes.mc_ground_truth_volume``).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OneLipschitzNet(nn.Module):
    """L-layer 1-Lipschitz net with each linear layer's spectral norm
    clipped to 1 via SVD truncation.

    No bias on each ``nn.Linear`` so that the composed map is purely
    linear when ``activation='identity'``. The cached ``W_list`` holds
    the per-layer weight matrices for downstream exact-volume
    computation.
    """

    def __init__(self, dim: int, n_layers: int = 4,
                 activation: str = 'identity', seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.n_layers = n_layers
        self.activation_name = activation
        self.layers = nn.ModuleList()
        self.W_list: list[torch.Tensor] = []
        for _ in range(n_layers):
            W = torch.randn(dim, dim, generator=gen)
            U, S, Vh = torch.linalg.svd(W)
            S_capped = torch.clamp(S, max=1.0)
            W_unit = (U * S_capped) @ Vh
            layer = nn.Linear(dim, dim, bias=False)
            with torch.no_grad():
                layer.weight.copy_(W_unit)
            self.layers.append(layer)
            self.W_list.append(W_unit.detach().clone())
        if activation == 'tanh':
            self._act = torch.tanh
        elif activation == 'silu':
            self._act = nn.functional.silu
        elif activation == 'identity':
            self._act = lambda x: x
        else:
            raise ValueError(f"unknown activation: {activation!r}")
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # no activation after last
                x = self._act(x)
        return x

    def total_weight(self) -> torch.Tensor:
        """Return the composed linear map W_L @ ... @ W_1 (identity-act case)."""
        W = torch.eye(self.dim)
        for layer in self.layers:
            W = layer.weight.detach() @ W
        return W


def make_synthetic_2d(seed: int = 0) -> OneLipschitzNet:
    return OneLipschitzNet(dim=2, n_layers=4, activation='identity', seed=seed)


def make_synthetic_3d(seed: int = 0) -> OneLipschitzNet:
    return OneLipschitzNet(dim=3, n_layers=4, activation='identity', seed=seed)


def make_synthetic_5d(seed: int = 0) -> OneLipschitzNet:
    return OneLipschitzNet(dim=5, n_layers=4, activation='identity', seed=seed)


def make_synthetic_10d(seed: int = 0) -> OneLipschitzNet:
    return OneLipschitzNet(dim=10, n_layers=4, activation='identity', seed=seed)


def make_synthetic_20d(seed: int = 0) -> OneLipschitzNet:
    return OneLipschitzNet(dim=20, n_layers=4, activation='identity', seed=seed)
