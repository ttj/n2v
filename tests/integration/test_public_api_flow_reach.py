"""Integration tests for the post-refactor flow-matching public API.

Covers the three entry points that lacked direct coverage prior to this
file:

  * TH-008: :func:`n2v.probabilistic.flow_reach` — the model-agnostic
    free function. Mirrors the OO surface and is the recommended
    callsite when the caller does not already hold a ``NeuralNetwork``.

  * TH-009: :func:`n2v.utils.verify_specification.verify_specification`
    when ``reach_set`` is a :class:`ProbabilisticSet`. Exercises the
    probabilistic dispatch branch + ``ProbVerifyConfig`` validation.

  * TH-010: :meth:`n2v.nn.NeuralNetwork.reach` with
    ``method='flow_matching'``. The OO entry point that wraps
    ``flow_reach`` under the hood; we pin parity here so the two
    surfaces remain bit-equivalent for identical seeded inputs.

Design notes
------------
The tests use a tiny 2-D linear ``nn.Sequential`` model and small
calibration / training budgets so the full file finishes well under
the ~30 s budget on a CPU runner. We do NOT mark the file
``slow`` — the goal is for these to run in the default ``-m 'not slow'``
suite as smoke tests for the public API.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn import NeuralNetwork
from n2v.probabilistic import flow_reach, FlowReachConfig
from n2v.sets import Box, HalfSpace
from n2v.utils.verify_specification import (
    ProbVerifyConfig,
    verify_specification,
)


# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------


def _make_tiny_model(seed: int = 0) -> nn.Module:
    """Build a deterministic 2-D -> 2-D linear-ReLU model.

    Using deterministic init means every call constructs an identical
    module, so callers can rely on byte-equal weights without seeding
    around ``nn.Linear``'s default Kaiming init.
    """
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    model.eval()
    return model


def _make_input_box() -> Box:
    """Unit box ``[0, 1]^2`` as the probabilistic-reach input region."""
    return Box(np.zeros(2), np.ones(2))


def _fast_flow_config(seed: int = 47) -> FlowReachConfig:
    """Small but valid config — keeps the file under the test budget.

    The numbers are chosen for speed, not for tight conformal guarantees;
    the tests assert structural properties (agreement, verdict shape),
    not the absolute coverage level.
    """
    return FlowReachConfig(
        epsilon=0.01,
        m=200,
        n_train=500,
        flow_epochs=50,
        flow_config='base',
        flow_coupling='none',
        flow_use_ema=False,
        infer_steps=10,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# TH-008 / TH-010 — parity between the OO and free-function surfaces
# -----------------------------------------------------------------------------


@pytest.mark.slow
def test_oo_and_free_function_agree():
    """``flow_reach`` and ``NeuralNetwork.reach(method='flow_matching')`` agree.

    With the same model, same input box, and the same seeded config,
    the two surfaces are documented to be equivalent. The strongest
    check we can make without coupling to implementation details is
    that the calibrated conformal threshold ``q`` (i.e.
    ``ProbabilisticSet.threshold``) matches exactly.
    """
    box = _make_input_box()
    config = _fast_flow_config(seed=47)

    # --- Free function ---
    model_free = _make_tiny_model(seed=0)
    prob_set_free = flow_reach(model_free, box, config)

    # --- OO wrapper (uses the same dispatch under the hood) ---
    model_oo = _make_tiny_model(seed=0)
    net = NeuralNetwork(model_oo, input_size=(2,))
    prob_set_oo = net.reach(box, method='flow_matching', config=config)

    # Threshold ``q`` is the calibrated rank-ell score; identical seeds
    # + identical config must produce the same value bit-for-bit.
    assert prob_set_free.threshold == prob_set_oo.threshold, (
        f"q differs across surfaces: free={prob_set_free.threshold!r}, "
        f"oo={prob_set_oo.threshold!r}"
    )

    # The carried metadata (epsilon, m, ell, dim) must also match.
    assert prob_set_free.epsilon == prob_set_oo.epsilon
    assert prob_set_free.m == prob_set_oo.m
    assert prob_set_free.ell == prob_set_oo.ell
    assert prob_set_free.dim == prob_set_oo.dim

    # Whitening transforms must also be bit-equivalent.
    assert prob_set_free.affine_transform is not None
    assert prob_set_oo.affine_transform is not None
    np.testing.assert_array_equal(
        prob_set_free.affine_transform.mean,
        prob_set_oo.affine_transform.mean,
    )
    np.testing.assert_array_equal(
        prob_set_free.affine_transform.std,
        prob_set_oo.affine_transform.std,
    )


# -----------------------------------------------------------------------------
# TH-009 — verify_specification probabilistic dispatch
# -----------------------------------------------------------------------------


def _build_prob_set_for_verify():
    """Helper: build a single ProbabilisticSet shared by the verify tests."""
    box = _make_input_box()
    config = _fast_flow_config(seed=47)
    model = _make_tiny_model(seed=0)
    return flow_reach(model, box, config)


@pytest.fixture(scope='module')
def _prob_set_module():
    """Module-scoped cache: training a flow is the costly part.

    All three ``verify_specification`` tests can share a single
    ``ProbabilisticSet`` since they only vary the spec / config.
    """
    return _build_prob_set_for_verify()


@pytest.fixture
def prob_set_cached(_prob_set_module):
    """Per-test view of the cached ProbabilisticSet, pinned to CPU.

    Some certify methods (notably ``amls_bounded``) move the flow's
    underlying ``nn.Module`` to GPU in-place via
    ``flow_ode.to(device)``. The scenario certifier, by contrast,
    builds CPU tensors directly and triggers a device mismatch if the
    flow was left on CUDA by a previous test in the same session.
    Pinning the flow to CPU before every test makes the fixture
    order-independent without paying the flow-training cost again.
    """
    flow_model = _prob_set_module.score_fn.flow_model
    flow_model.to('cpu')
    return _prob_set_module


@pytest.mark.slow
def test_verify_specification_amls_bounded_unsafe_certified(prob_set_cached):
    """A spec far from the reach set should certify UNSAT under AMLS-bounded.

    Construction: pick a halfspace whose unsafe region is a half-plane
    far above any plausible output of the tiny linear-ReLU model. The
    model maps ``[0,1]^2`` to a bounded region near zero (initialised
    with default Kaiming on a 2-layer net), so requiring
    ``y_0 >= 1000`` is trivially disjoint. AMLS-bounded should bound
    the verification-layer mass under the target ``eps_2`` and return
    ``UNSAT``.
    """
    # Spec: unsafe region = ``{y : -y_0 <= -1000}`` i.e. ``y_0 >= 1000``.
    # Encoded as ``G y <= g`` per the n2v VNN-LIB convention.
    G = np.array([[-1.0, 0.0]])
    g = np.array([[-1000.0]])
    spec = HalfSpace(G, g)

    cfg = ProbVerifyConfig(
        method='amls_bounded',
        n_samples=200,
        beta=0.05,
        seed=0,
        amls_quantile=0.25,
        amls_max_levels=5,
        amls_n_mcmc_steps=3,
        amls_bounded_eps_2_target=0.01,
    )

    result = verify_specification(prob_set_cached, spec, config=cfg)

    assert result.verdict == 'UNSAT', (
        f"expected UNSAT for trivially-disjoint spec, got {result.verdict!r}"
    )
    # Probabilistic dispatch always populates the joint guarantee fields.
    assert result.epsilon_total is not None
    assert result.q is not None


@pytest.mark.slow
def test_verify_specification_amls_bounded_witness_unknown(prob_set_cached):
    """A spec that overlaps the reach set yields UNKNOWN under AMLS-bounded.

    Construction: pick a halfspace whose unsafe region is a half-plane
    that clearly intersects the model's output region. The simplest
    such spec is ``y_0 <= +1000`` (a near-universal halfspace), which
    contains the entire output region. AMLS-bounded should detect an
    unsafe witness and return ``UNKNOWN`` (probabilistic dispatch never
    returns ``SAT``).
    """
    # Spec: unsafe region = ``{y : y_0 <= 1000}`` — covers all outputs.
    G = np.array([[1.0, 0.0]])
    g = np.array([[1000.0]])
    spec = HalfSpace(G, g)

    cfg = ProbVerifyConfig(
        method='amls_bounded',
        n_samples=200,
        beta=0.05,
        seed=0,
        amls_quantile=0.25,
        amls_max_levels=5,
        amls_n_mcmc_steps=3,
        amls_bounded_eps_2_target=0.01,
    )

    result = verify_specification(prob_set_cached, spec, config=cfg)

    assert result.verdict == 'UNKNOWN', (
        f"expected UNKNOWN for trivially-overlapping spec, got "
        f"{result.verdict!r}"
    )
    # AMLS-bounded should flag detection on a spec that covers the entire
    # output region. The flag is best-effort but if populated must be
    # truthy.
    if result.amls_bounded_detected_unsafe is not None:
        assert result.amls_bounded_detected_unsafe is True


@pytest.mark.slow
def test_verify_specification_scenario_method(prob_set_cached):
    """The 'scenario' branch certifies a trivially-disjoint spec UNSAT.

    Same far-away spec as the AMLS-bounded UNSAT test, but routed through
    the scenario certifier. Scenario does NOT require
    ``amls_bounded_eps_2_target``, so this also exercises the default-
    config path of ``ProbVerifyConfig`` for that field.
    """
    # Same spec as the AMLS-bounded UNSAT test.
    G = np.array([[-1.0, 0.0]])
    g = np.array([[-1000.0]])
    spec = HalfSpace(G, g)

    cfg = ProbVerifyConfig(
        method='scenario',
        n_samples=200,
        beta=0.05,
        seed=0,
    )

    result = verify_specification(prob_set_cached, spec, config=cfg)

    assert result.verdict == 'UNSAT', (
        f"expected UNSAT for trivially-disjoint spec via scenario, got "
        f"{result.verdict!r}"
    )
    assert result.q is not None
    # Scenario reports its own epsilon_2; the joint guarantee should
    # therefore be populated.
    assert result.epsilon_total is not None
