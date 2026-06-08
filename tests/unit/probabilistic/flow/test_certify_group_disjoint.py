"""Tests for certify_group_disjoint (layer 2: OR within group).

Early-exit semantics for epsilon_2: when ``certify_group_disjoint``
short-circuits on the first not-disjoint HalfSpace, only the executed
HalfSpaces contribute to the Bonferroni bound — the unexecuted ones
were not subjected to a hypothesis test (no samples drawn for them),
so they do not consume any error budget. Therefore
``epsilon_2 == sum(r.epsilon_2 for r in per_hs_results)`` over the
executed members, NOT a planned ``len(group) * log(1/beta)/N`` total.
This matches standard scenario-theory accounting for sequentially-run
hypothesis tests with adaptive stopping.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from n2v.sets.halfspace import HalfSpace


def test_single_halfspace_group_reduces_to_layer_1():
    """A group with one HalfSpace is disjoint iff that HalfSpace is disjoint."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    group = [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is True


def test_group_not_disjoint_if_any_hs_hit():
    """If even one HalfSpace in a group is reachable, the group is not disjoint."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    # HS_A: y_0 <= -100 (unreachable). HS_B: y_0 <= 100 (trivially reachable).
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is False


def test_group_disjoint_epsilon_is_bonferroni_over_hs():
    """epsilon_2 = |group| * log(1/beta_2) / n_samples."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        HalfSpace(np.array([[0.0, 1.0]]), np.array([[-100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.epsilon_2 == pytest.approx(
        2 * math.log(1000) / 500, rel=1e-6
    )


def test_group_disjoint_returns_per_hs_results():
    """The result includes per-HalfSpace sub-results. The first
    HalfSpace is disjoint (unreachable), so the loop continues to the
    second (reachable), yielding a two-entry per_hs_results list and
    a group verdict of not-disjoint."""
    from n2v.probabilistic.flow.scenario_verify import (
        certify_group_disjoint, HalfSpaceDisjointResult,
    )
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        HalfSpace(np.array([[0.0, 1.0]]), np.array([[100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    # Order matters: first HS is disjoint, so the loop proceeds to HS 2.
    assert len(result.per_hs_results) == 2
    assert all(isinstance(r, HalfSpaceDisjointResult)
               for r in result.per_hs_results)
    assert result.per_hs_results[0].disjoint is True
    assert result.per_hs_results[1].disjoint is False
    # Group is not disjoint (HS 2 is reachable).
    assert result.disjoint is False


def test_group_disjoint_early_exit_on_first_failure():
    """If the FIRST HalfSpace is not disjoint, the loop short-circuits
    and per_hs_results has just one entry."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    # First HalfSpace is trivially reachable (y_0 <= 100).
    # Second would be disjoint (y_0 <= -100) but shouldn't run due to early exit.
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]])),
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is False
    assert len(result.per_hs_results) == 1
    assert result.per_hs_results[0].disjoint is False
    # epsilon is Bonferroni over EXECUTED HalfSpaces only — the unexecuted
    # second HS did not consume scenario budget (no samples drawn).
    import math
    assert result.epsilon_2 == pytest.approx(1 * math.log(1000) / 500, rel=1e-6)
