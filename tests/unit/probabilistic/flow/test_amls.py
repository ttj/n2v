"""Tests for C1 / AMLS (Adaptive Multilevel Splitting) flow-set detector.

These tests exercise the AMLS rare-event estimator on small synthetic
flows, plus the dispatch into ``verify_specification`` via
``verification_method='amls'``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from n2v.probabilistic.flow.amls import (
    amls_certify_spec,
    amls_estimate_halfspace_mass,
)
from n2v.sets.halfspace import HalfSpace


# ---------- Identity flow stub ----------


class _IdentityFlow:
    """A trivial 'flow' whose inverse is the identity.

    Lets us drive the AMLS algorithm with a known target distribution
    (the latent ``z ~ N(0, I_d)`` is also the data distribution).
    """

    def __init__(self):
        self.velocity_field = None  # unused

    def inverse(self, z, **_kw):
        return z

    def forward(self, y, **_kw):
        return y

    def eval(self):
        return self

    def to(self, *_args, **_kw):
        return self


# ---------- Easy case: U is the bulk; detect at level 0 ----------


def test_amls_finds_unsafe_in_easy_case():
    """U = { y_0 >= -10 } contains nearly all probability mass under
    standard normal. AMLS should detect at level 0.
    """
    flow = _IdentityFlow()
    # G y <= g  i.e.  -y_0 <= 10  i.e.  y_0 >= -10
    G = np.array([[-1.0, 0.0]])
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = amls_estimate_halfspace_mass(
        flow, hs, n_samples_per_level=500, max_levels=10, seed=0,
    )
    assert res.detected_unsafe is True
    assert res.levels_used == 1
    assert res.pi_hat > 0.99  # vast majority in U


# ---------- Hard case: U is in the deep tail, detect via splitting ----------


def test_amls_finds_unsafe_in_hard_case():
    """U = { y_0 <= -3 } has mass ~0.00135 under N(0,1). AMLS should
    detect within a small number of levels (rho^k ~ 0.001 => k ~ 3).
    """
    flow = _IdentityFlow()
    # G y <= g  =>  y_0 <= -3
    G = np.array([[1.0]])
    g = np.array([-3.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=1000, quantile=0.1, max_levels=10,
        n_mcmc_steps=20, mcmc_step_size=0.5, seed=0,
    )
    assert res.detected_unsafe is True
    # rho=0.1, mass ~1.35e-3, log_0.1(1.35e-3) ~ 2.87 => 3-4 levels
    assert 1 <= res.levels_used <= 6


def test_amls_correctly_estimates_known_mass():
    """For U = { y_0 <= -3 } in 1D with N(0,1) data, P_true ~ 0.00135.

    AMLS estimate should be within ~3x (loose tolerance for stochastic
    estimator with N=1000 and MCMC mixing).
    """
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([-3.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=2000, quantile=0.1, max_levels=10,
        n_mcmc_steps=20, mcmc_step_size=0.4, seed=0,
    )
    assert res.detected_unsafe is True
    true_mass = 0.00135
    # Loose 3x bound: AMLS with random-walk MH on a 1D Gaussian is
    # noisy at small N; we mostly care about order of magnitude.
    assert res.pi_hat <= 3.0 * true_mass
    assert res.pi_hat >= true_mass / 10.0


# ---------- Empty / unreachable polyhedron ----------


def test_amls_no_detect_when_truly_disjoint():
    """U = { y_0 <= -100 } has effectively zero mass under N(0,1).
    AMLS should exhaust max_levels without detecting U.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=200, quantile=0.1, max_levels=5,
        n_mcmc_steps=5, mcmc_step_size=0.3, seed=0,
    )
    assert res.detected_unsafe is False
    # final_phi is the MIN phi observed; outside U => phi > 0.
    assert res.final_phi > 0.0


# ---------- Group / spec dispatcher ----------


def test_amls_certify_spec_unsat_when_all_disjoint():
    """Spec with one group containing one unreachable HalfSpace =>
    unsat_certified True, detected_any False.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = amls_certify_spec(
        flow, [[hs]],
        n_samples_per_level=200, max_levels=5, n_mcmc_steps=5, seed=0,
    )
    assert res.detected_any is False
    assert res.unsat_certified is True


def test_amls_certify_spec_unknown_when_group_member_detected():
    """Single group with one reachable HalfSpace: detected => no group
    is fully disjoint => unsat_certified False.
    """
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = amls_certify_spec(
        flow, [[hs]],
        n_samples_per_level=200, max_levels=5, n_mcmc_steps=5, seed=0,
    )
    assert res.detected_any is True
    assert res.unsat_certified is False


def test_amls_certify_spec_two_groups_one_disjoint_unsat():
    """Spec with TWO groups (AND across groups): one group disjoint
    suffices for UNSAT.
    """
    flow = _IdentityFlow()
    # Group 1: unreachable HalfSpace
    hs_far = HalfSpace(
        np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    # Group 2: reachable bulk
    hs_bulk = HalfSpace(
        np.array([[-1.0, 0.0]]), np.array([[10.0]]))
    res = amls_certify_spec(
        flow, [[hs_far], [hs_bulk]],
        n_samples_per_level=200, max_levels=5, n_mcmc_steps=5, seed=0,
    )
    # Group 1 is disjoint => unsat_certified
    assert res.unsat_certified is True
    # But a HalfSpace in group 2 was detected
    assert res.detected_any is True


