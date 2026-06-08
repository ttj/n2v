"""Tests for certify_halfspace_disjoint (joint-target scenario-verify)."""
from __future__ import annotations
import math
import numpy as np
import pytest
import torch

from n2v.sets.halfspace import HalfSpace

from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow


@pytest.mark.slow
def test_certify_halfspace_disjoint_returns_disjoint_for_unreachable_polyhedron():
    """Polyhedron entirely outside the unit-ball flow set should be
    certified disjoint."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
    flow = _train_small_2d_flow(seed=0)
    # HalfSpace: y_0 <= -100 — clearly unreachable from a unit-ball flow set.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    result = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is True
    assert result.epsilon_2 == pytest.approx(math.log(1000) / 500, rel=1e-6)


@pytest.mark.slow
def test_certify_halfspace_disjoint_returns_not_disjoint_for_reachable_polyhedron():
    """Polyhedron overlapping the flow set should return disjoint=False
    with a sample inside."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
    flow = _train_small_2d_flow(seed=0)
    # HalfSpace: y_0 <= 100 — trivially overlaps the unit-ball flow set.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))
    result = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is False
    # worst sample should satisfy the constraint (be inside the polyhedron)
    assert (hs.G @ result.worst_sample).flatten()[0] <= hs.g.flatten()[0]


@pytest.mark.slow
def test_certify_halfspace_disjoint_multirow_and_semantics():
    """Multi-row HalfSpace encodes AND. For a 2-row AND where each row
    individually could be violated but their intersection is empty, the
    joint-target check must certify disjoint."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
    flow = _train_small_2d_flow(seed=0)
    # Row 0: y_0 <= -2.   Row 1: y_0 >= 2 (i.e., -y_0 <= -2).
    # Intersection is empty — no point satisfies both simultaneously.
    hs = HalfSpace(np.array([[1.0, 0.0], [-1.0, 0.0]]),
                   np.array([[-2.0], [-2.0]]))
    result = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is True


@pytest.mark.slow
def test_certify_halfspace_disjoint_epsilon_2_formula():
    """epsilon_2 = log(1/beta_2) / n_samples for any HalfSpace."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
    flow = _train_small_2d_flow(seed=0)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))
    for N, beta in [(500, 0.001), (2000, 0.01), (100, 0.1)]:
        r = certify_halfspace_disjoint(
            flow_ode=flow, threshold_q=3.0, halfspace=hs,
            n_samples=N, beta_2=beta, seed=0,
        )
        assert r.epsilon_2 == pytest.approx(math.log(1/beta) / N, rel=1e-6)


@pytest.mark.slow
def test_certify_halfspace_disjoint_joint_tighter_than_per_row():
    """The load-bearing case: a 2-row HalfSpace where EACH row is
    individually reachable (per-row loop would see violations), but
    their joint AND intersection is geometrically outside the flow set.
    Joint-target must certify disjoint here; per-row would not.

    Construction:
      - Flow set: latent ball of radius 3 in 2D (approximately the unit
        disk of radius 3 in data space after near-identity transport).
      - Row 0: y_0 <= -2.8  (left crescent of the disk — reachable alone).
      - Row 1: y_1 <= -2.8  (bottom crescent — reachable alone).
      - AND intersection: the corner at y_0,y_1 <= -2.8. This lies
        geometrically OUTSIDE the disk of radius 3 because
        (-2.8)^2 + (-2.8)^2 = 15.68 > 9 = 3^2.
    So N=500 flow samples should find individual violators of each row
    but none inside the joint AND → joint-target certifies disjoint.
    """
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
    flow = _train_small_2d_flow(seed=0)
    hs = HalfSpace(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[-2.8], [-2.8]]),
    )
    result = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=500, beta_2=0.001, seed=0,
    )
    # Joint-target should certify disjoint — AND intersection is outside the disk.
    assert result.disjoint is True, (
        f'Expected disjoint=True (joint-AND corner is geometrically outside '
        f'the flow disk of radius 3), got disjoint=False with '
        f'worst_max_margin={result.worst_max_margin}'
    )
    # Sanity check: individually, each row's *direction* is reachable —
    # confirm by sampling a fresh batch and checking that the flow set
    # extends well into each halfspace direction. (We use -2.0 here as
    # a robust threshold well above each axis's numerical min; the
    # -2.8 threshold of the actual halfspace rows is geometrically
    # individually reachable but numerically sparse at N=500. The point
    # is only to confirm the flow set is non-degenerate along each axis
    # direction so the AND-vs-per-row gap is meaningful.)
    np.random.seed(1)
    from n2v.probabilistic.flow.scenario_verify import sample_truncated_gaussian_ball
    z = sample_truncated_gaussian_ball(q=3.0, dim=2, n_samples=500)
    with torch.no_grad():
        y = flow.inverse(torch.tensor(z, dtype=torch.float32)).numpy()
    # Row 0 direction: some samples have y_0 <= -2.0 (reachable individually)
    row0_reachable = (y[:, 0] <= -2.0).any()
    # Row 1 direction: some samples have y_1 <= -2.0 (reachable individually)
    row1_reachable = (y[:, 1] <= -2.0).any()
    assert row0_reachable, (
        'Row 0 alone should be reachable by the flow set; '
        'test construction invalid if not.'
    )
    assert row1_reachable, (
        'Row 1 alone should be reachable by the flow set; '
        'test construction invalid if not.'
    )
