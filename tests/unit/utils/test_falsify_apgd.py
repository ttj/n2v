"""Tests for APGD (Auto-PGD) falsifier."""
from __future__ import annotations
import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.halfspace import HalfSpace
from n2v.utils.falsify import falsify


class _LinearNet(nn.Module):
    """y = x @ W.T + b with deterministic init."""
    def __init__(self, dim_in=2, dim_out=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x):
        return self.lin(x)


def test_apgd_finds_known_sat():
    """APGD should find a real counterexample on a simple 2D linear net."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    # Unsafe region that every x in the box maps into: y_0 <= 1e6.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='apgd', seed=0)
    assert result == 0
    assert cex is not None
    x_cex, y_cex = cex
    assert x_cex.shape == (2,)
    assert y_cex.shape == (2,)


def test_apgd_returns_unknown_if_no_sat():
    """APGD on an unreachable unsafe region returns 2 (unknown)."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    # Unsafe region unreachable by any input in [-1, 1]^2.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='apgd',
                          seed=0, n_restarts=3, n_steps=20)
    assert result == 2
    assert cex is None


def test_apgd_respects_input_bounds():
    """Counterexample must be inside [lb, ub] element-wise."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-0.5, -0.5], dtype=np.float32)
    ub = np.array([ 0.5,  0.5], dtype=np.float32)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='apgd', seed=0)
    assert result == 0
    x_cex, _ = cex
    assert (x_cex >= lb - 1e-5).all()
    assert (x_cex <= ub + 1e-5).all()


def test_random_pgd_apgd_ensemble_short_circuits():
    """The ensemble method returns as soon as any stage finds SAT."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='random+pgd+apgd', seed=0)
    assert result == 0
    assert cex is not None


def test_autoattack_scaffold_raises_when_not_installed():
    """If the 'autoattack' package is not installed, method='autoattack'
    raises ImportError with install instructions."""
    from n2v.utils.falsify import _HAS_AUTOATTACK
    from n2v.utils.falsify import falsify
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    if _HAS_AUTOATTACK:
        # Package is installed: scaffold raises NotImplementedError
        with pytest.raises(NotImplementedError):
            falsify(net, lb, ub, hs, method='autoattack')
    else:
        # Package not installed: import-time guard raises ImportError
        with pytest.raises(ImportError):
            falsify(net, lb, ub, hs, method='autoattack')
