"""Tests for certify_spec_disjoint (layer 3: AND across groups).

Early-exit semantics for epsilon_2: when ``certify_spec_disjoint``
finds a disjoint group it short-circuits and stops testing later groups.
Similarly, ``certify_group_disjoint`` short-circuits on the first
not-disjoint HalfSpace within a group. The Bonferroni bound is
summed over ONLY the HalfSpaces actually executed (i.e. the ones that
ran a hypothesis test on N samples). Unexecuted HalfSpaces do not
consume error budget. So
``epsilon_2 == sum(g.epsilon_2 for g in per_group_results)`` over the
executed prefix of groups, where each ``g.epsilon_2`` already aggregates
its executed-HalfSpaces sum.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from n2v.sets.halfspace import HalfSpace


def test_single_group_spec_unsat_iff_group_disjoint():
    """A spec with one group reduces to layer 2 on that group."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [[HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))]]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is True
    assert result.certifying_group_idx == 0


def test_two_group_spec_unsat_if_any_group_disjoint():
    """Group 0 reachable, group 1 unreachable. Spec is UNSAT (one group suffices)."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))],   # reachable
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))],  # unreachable
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is True
    assert result.certifying_group_idx == 1


def test_no_group_disjoint_means_not_unsat():
    """Every group reachable → can't certify UNSAT."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))],   # reachable
        [HalfSpace(np.array([[0.0, 1.0]]), np.array([[100.0]]))],   # reachable
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is False
    assert result.certifying_group_idx is None


def test_spec_epsilon_is_bonferroni_across_executed_hs():
    """epsilon_2_spec = sum over EXECUTED (group, hs) of log(1/beta_2)/n_samples.

    Group 0 has two unreachable HalfSpaces (both disjoint → group 0 is
    disjoint → spec UNSAT, early-exit before group 1). Within group 0,
    no early exit fires (every HS is disjoint), so 2 HalfSpace tests
    run. Group 1's HalfSpace never executes. Total: 2 executed tests.
    """
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
         HalfSpace(np.array([[0.0, 1.0]]), np.array([[-100.0]]))],  # 2 hs (disjoint)
        [HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-100.0]]))],  # not executed
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    # 2 executed halfspaces × log(1000)/500 (group 1 skipped via spec early exit)
    assert result.epsilon_2 == pytest.approx(2 * math.log(1000) / 500, rel=1e-6)


def test_spec_early_exit_on_first_disjoint_group():
    """When group 0 is disjoint, we can stop and report UNSAT
    without running later groups. per_group_results may be shorter
    than len(spec_groups)."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))],  # disjoint (UNSAT witness)
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))],   # reachable (would fail)
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is True
    assert result.certifying_group_idx == 0
    assert len(result.per_group_results) == 1  # early exit
    # epsilon is Bonferroni over EXECUTED HalfSpaces only — group 1
    # never ran a hypothesis test so it does not consume error budget.
    assert result.epsilon_2 == pytest.approx(1 * math.log(1000) / 500, rel=1e-6)


def test_empty_spec_raises():
    """An empty spec (no groups) is invalid."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    with pytest.raises(ValueError):
        certify_spec_disjoint(
            flow_ode=flow, threshold_q=3.0, spec_groups=[],
            n_samples=500, beta_2=0.001, seed=0,
        )
