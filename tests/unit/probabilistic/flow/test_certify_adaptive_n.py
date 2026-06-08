"""Tests for the adaptive 2-stage N option in certify_halfspace_disjoint."""
from __future__ import annotations
import math
import numpy as np
import pytest
import torch

from n2v.sets.halfspace import HalfSpace

from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow


@pytest.mark.slow
def test_adaptive_n_escalates_when_margin_is_marginal():
    """When the worst-sample's max-row-margin is small (i.e., certification
    was marginal at N=200), the function should re-run with the larger N
    and return the larger-N epsilon_2 + result."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint

    flow = _train_small_2d_flow(seed=0)
    # HalfSpace whose disjoint check at N=200 produces a small positive
    # margin (a "marginal" disjoint): y_0 <= -2.8 with seed=1 yields
    # disjoint=True with worst_max_margin ~0.5 on the standard-normal
    # 2D flow. Adaptive threshold above that margin must trigger
    # escalation.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-2.8]]))

    # First, run without adaptive — capture the small-N margin
    r_small = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=200, beta_2=0.001, seed=1,
    )

    # Sanity check: this configuration must produce a marginal
    # disjoint=True at small N. If this assertion fails the flow
    # training drift has changed; pick another (b, seed) combo with
    # disjoint=True and worst_max_margin > 0.
    assert r_small.disjoint is True, (
        "test setup expects disjoint=True at small N; "
        f"got disjoint=False with margin={r_small.worst_max_margin}"
    )
    assert r_small.worst_max_margin > 0

    # Now run with adaptive: threshold above the small-N margin so it
    # ESCALATES to a larger N
    threshold = r_small.worst_max_margin + 1.0   # generous; ensures escalation
    r_adapt = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=200, beta_2=0.001, seed=1,
        adaptive_threshold=threshold, adaptive_n_samples=2000,
    )

    # epsilon_2 must reflect the LARGER N (the adaptive escalation)
    assert r_adapt.n_samples_used == 2000
    assert r_adapt.epsilon_2 == pytest.approx(math.log(1000) / 2000, rel=1e-6)
    # Smaller N's epsilon_2 was log(1000)/200 = 5x larger; the escalated
    # epsilon must be at most 1/5 of that.
    assert r_adapt.epsilon_2 < r_small.epsilon_2 / 4


@pytest.mark.slow
def test_adaptive_n_does_not_escalate_when_margin_is_strong():
    """When the worst-sample's max-row-margin is well above the threshold
    (strong certification at small N), no escalation. Returns the small-N
    result as-is."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint

    flow = _train_small_2d_flow(seed=0)
    # Unreachable polyhedron: y_0 <= -100. Worst sample's max-row-margin
    # will be large (>> 1).
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))

    r = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=500, beta_2=0.001, seed=0,
        adaptive_threshold=1.0, adaptive_n_samples=20000,
    )
    assert r.n_samples_used == 500   # no escalation
    assert r.epsilon_2 == pytest.approx(math.log(1000) / 500, rel=1e-6)


@pytest.mark.slow
def test_adaptive_n_returns_not_disjoint_without_escalation():
    """When the small-N run returns disjoint=False (some sample landed
    inside), there's no need to escalate. Return the small-N result."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint

    flow = _train_small_2d_flow(seed=0)
    # Trivially-reachable polyhedron: y_0 <= 100. Many samples inside.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))

    r = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=200, beta_2=0.001, seed=0,
        adaptive_threshold=0.5, adaptive_n_samples=20000,
    )
    assert r.disjoint is False
    assert r.n_samples_used == 200   # no escalation; first stage already conclusive
