"""Tests for C2 / Tilted Importance Sampling flow-set detector.

These tests exercise the self-normalised IS estimator on small synthetic
flows, plus the dispatch into ``verify_specification`` via
``verification_method='is_tilted'``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from n2v.probabilistic.flow.importance_sampling import (
    is_tilted_certify_spec,
    is_tilted_estimate_halfspace_mass,
)
from n2v.sets.halfspace import HalfSpace


# ---------- Identity flow stub ----------


class _IdentityFlow:
    """A trivial 'flow' whose inverse is the identity.

    Lets us drive IS with a known target distribution -- the latent
    ``z ~ N(0, I_d)`` is also the data distribution.
    """

    def __init__(self):
        self.velocity_field = None

    def inverse(self, z, **_kw):
        return z

    def forward(self, y, **_kw):
        return y

    def eval(self):
        return self

    def to(self, *_args, **_kw):
        return self


# ---------- Smoke + return-shape ----------


def test_is_returns_correct_shape_and_keys():
    """Smoke: function runs and returns a populated ISResult."""
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.n_samples == 200
    assert 0.0 <= res.pi_hat <= 1.0
    assert 0.0 <= res.pi_upper <= 1.0
    assert res.ess > 0
    assert res.worst_y.shape == (2,)
    assert isinstance(res.detected_unsafe, bool)
    # On a bulk halfspace IS should detect U.
    assert res.detected_unsafe is True


# ---------- Unbiasedness on a known case ----------


@pytest.mark.slow
def test_is_zero_lambda_recovers_flat_mc():
    """At lambda=0 all weights are 1 and pi_hat reduces to the flat
    Monte-Carlo indicator average. That is exactly P_flow(U) under p_z.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([-2.0])  # U = { y_0 <= -2 }, true mass ~0.02275
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs,
        n_samples=10_000, lambda_tilt=0.0, seed=0,
    )
    true_mass = 0.02275  # 1 - Phi(2)
    assert res.detected_unsafe is True
    # Flat MC at N=10K is unbiased; loose 2x tolerance for noise.
    assert res.pi_hat <= 2.0 * true_mass
    assert res.pi_hat >= true_mass / 2.0


@pytest.mark.slow
def test_is_self_normalised_estimates_q_conditional_mass():
    """Self-normalised IS with samples drawn from p_z and weights w(z)
    estimates the q-CONDITIONAL mass of U, not P_flow(U):

        SN-IS = E_{p_z}[ w(z) * 1[U] ] / E_{p_z}[ w(z) ]
              = E_q[ 1[U] ] = P(U | q)

    Since q tilts mass toward U, P(U|q) >= P_flow(U). Test that the
    SN-IS estimate is at least the flat MC estimate (lambda=0) on the
    same data.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([-2.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res_flat = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=10_000, lambda_tilt=0.0, seed=0,
    )
    res_tilt = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=10_000, lambda_tilt=2.0, seed=0,
    )
    assert res_tilt.pi_hat >= res_flat.pi_hat


def test_is_no_detect_when_truly_disjoint():
    """U = { y_0 <= -100 } has effectively zero mass under N(0,1).
    IS should NOT detect U with n_samples=200 (no sample reaches the deep
    tail; that's the limitation IS shares with flat MC -- it does not
    sample biased toward U, only reweights).
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.detected_unsafe is False
    # final_phi is the MIN slack observed; outside U => slack > 0.
    assert res.final_phi > 0.0
    # pi_hat should be 0 (no detection means no in-U samples; the SN-IS
    # numerator is identically zero).
    assert res.pi_hat == 0.0


def test_is_zero_lambda_recovers_uniform_mc():
    """With lambda=0 weights are all 1 and the estimator collapses to
    flat MC: pi_hat = (in_U).mean()."""
    flow = _IdentityFlow()
    G = np.array([[-1.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=2000, lambda_tilt=0.0, seed=0,
    )
    # All samples are in U => pi_hat ~ 1.
    assert res.pi_hat > 0.99
    # ESS should be ~ N when all weights equal.
    assert abs(res.ess - res.n_samples) < 1e-6


def test_is_ess_decreases_with_lambda():
    """Larger lambda => more concentrated weights => smaller ESS."""
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([-1.0])  # U = {y_0 <= -1}, moderate mass ~0.16
    hs = HalfSpace(G, g.reshape(-1, 1))

    res_small = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=2000, lambda_tilt=0.5, seed=0,
    )
    res_large = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=2000, lambda_tilt=10.0, seed=0,
    )
    assert res_large.ess < res_small.ess


# ---------- Spec-level dispatcher ----------


def test_is_certify_spec_unsat_when_all_disjoint():
    """Spec with one group containing one unreachable HalfSpace =>
    unsat_certified True, detected_any False.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = is_tilted_certify_spec(
        flow, [[hs]],
        n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.detected_any is False
    assert res.unsat_certified is True


def test_is_certify_spec_unknown_when_group_member_detected():
    """Single group with one reachable HalfSpace: detected => no group
    is fully disjoint => unsat_certified False.
    """
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = is_tilted_certify_spec(
        flow, [[hs]],
        n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.detected_any is True
    assert res.unsat_certified is False


def test_is_certify_spec_two_groups_one_disjoint_unsat():
    """Spec with TWO groups (AND across groups): one disjoint group
    suffices for UNSAT.
    """
    flow = _IdentityFlow()
    hs_far = HalfSpace(
        np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    hs_bulk = HalfSpace(
        np.array([[-1.0, 0.0]]), np.array([[10.0]]))
    res = is_tilted_certify_spec(
        flow, [[hs_far], [hs_bulk]],
        n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.unsat_certified is True
    assert res.detected_any is True


