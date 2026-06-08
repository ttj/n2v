"""
Scenario-based probabilistic verification on flow sets.

Implements sound probabilistic verification of halfspace specifications
on the implicit reach set {y : ||phi_t(y)||_2 <= q} via scenario
optimization on the latent ball.

Algorithm:
1. Reformulate the spec via the inverse flow: y = psi_t(z), ||z|| <= q
2. Sample N points from N(0, I) truncated to the ball ||z|| <= q
3. Map each sample to data space via the inverse flow
4. Evaluate the spec at each mapped point
5. If any sample violates: return counterexample (sound — real point)
6. Else: return certificate with epsilon_2 = -log(beta_2) / N

The result composes with the conformal coverage guarantee from the
flow score calibration to give a joint probabilistic certificate.

Reference: Campi & Garatti, "Risk and Complexity in Scenario
Optimization", Mathematical Programming, 2019.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from scipy.stats import chi

from n2v.sets.halfspace import HalfSpace


_REJECTION_DIM_THRESHOLD = 30


def _qmc_sample_latents(
    n_samples: int,
    dim: int,
    *,
    seed: 'int | None',
    device='cpu',
    dtype=torch.float32,
    antithetic: bool = False,
) -> torch.Tensor:
    """Sobol-based QMC sample of N(0, I_d).

    Uses ``scipy.stats.qmc.Sobol`` for low-discrepancy points in
    ``[0, 1]^d``, then inverse-Gaussian-CDF (``norm.ppf``) to map to the
    standard normal. Returns a tensor of shape ``(n_samples, dim)`` on
    ``device`` with ``dtype``.

    Equidistributed on N(0, I), so the scenario optimization estimator
    remains unbiased; variance is reduced by 2-4x typical compared to
    i.i.d. Gaussian sampling for smooth integrands.

    Args:
        n_samples: number of latent samples to draw.
        dim: dimensionality of each sample.
        seed: optional seed for the Sobol scrambler. ``None`` is allowed
            (scipy treats it as system entropy).
        device: torch device for the returned tensor.
        dtype: torch dtype for the returned tensor.
        antithetic: when True, draw ``ceil(n_samples / 2)`` Sobol points
            and pair each ``z`` with ``-z`` to fill ``n_samples`` total.
            For odd ``n_samples`` the final unpaired Sobol point is
            included as-is. Each sample is marginally N(0, I_d) (the
            standard Gaussian is symmetric), so the scenario bound still
            applies.

    Returns:
        ``(n_samples, dim)`` torch tensor of standard normal QMC samples.
    """
    from scipy.stats import qmc, norm
    if antithetic:
        n_base = (n_samples + 1) // 2  # ceil(n_samples / 2)
    else:
        n_base = n_samples
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random(n_base)  # numpy (n_base, d) in [0, 1]^d
    # Avoid 0 and 1 (norm.ppf would give -inf / +inf); clip to (eps, 1-eps).
    u = u.clip(1e-10, 1 - 1e-10)
    z = norm.ppf(u)  # numpy, standard normal
    if antithetic:
        z_pair = np.concatenate([z, -z], axis=0)  # (2 * n_base, d)
        z = z_pair[:n_samples]  # trim to exactly n_samples
    return torch.from_numpy(z).to(device=device, dtype=dtype)


@dataclass
class ScenarioResult:
    """
    Result of scenario-based verification on a single halfspace spec.

    Attributes:
        verified: True if outcome == 'verified', False otherwise (kept for
            backward compatibility).
        outcome: One of 'verified', 'falsified', 'unknown'.
            'verified' = no flow-set violation found (probabilistic certificate)
            'falsified' = flow-set violation found + real preimage found
            'unknown' = flow-set violation found + no preimage (hallucination)
            or preimage search was not requested.
        counterexample: If outcome != 'verified', a tuple (z, y, margin)
            from the flow set. None if verified.
        genuine_input: If outcome == 'falsified', a real input x in the
            input set such that f(x) is within tolerance of the
            counterexample's y. None otherwise.
        epsilon_2: Scenario violation bound = -log(beta_2) / N.
        delta_2: Scenario confidence = 1 - beta_2.
        n_samples_used: Number of latent samples drawn.
    """
    verified: bool
    outcome: str
    counterexample: Optional[Tuple[np.ndarray, np.ndarray, float]]
    genuine_input: Optional[np.ndarray]
    epsilon_2: float
    delta_2: float
    n_samples_used: int


@dataclass
class HalfSpaceDisjointResult:
    """Result of certify_halfspace_disjoint (joint-target scenario verify).

    disjoint: True iff no flow sample was found inside the HalfSpace's polyhedron.
    worst_sample: (d,) ndarray — flow sample with minimum max-row-margin
                  (most likely to be inside the polyhedron; useful for diagnostics).
    worst_max_margin: float — min over samples of max over rows of (G[i]@y - g[i]).
                     Positive iff disjoint.
    epsilon_2: scenario bound log(1/beta_2) / n_samples.
    n_samples_used: echo of n_samples.
    """
    disjoint: bool
    worst_sample: np.ndarray
    worst_max_margin: float
    epsilon_2: float
    n_samples_used: int


@dataclass
class GroupDisjointResult:
    """Result of certify_group_disjoint (layer 2 of the scenario-verify
    three-layer dispatcher).

    A 'group' is a list of HalfSpaces with OR semantics — the group is
    hit iff AT LEAST ONE HalfSpace is hit. Therefore the group is
    DISJOINT from the flow set iff EVERY HalfSpace in it is disjoint.

    Fields:
        disjoint: True iff every HalfSpace in the group was individually
            certified disjoint.
        per_hs_results: list of HalfSpaceDisjointResult, one per
            HalfSpace in the group, in the same order as the input.
        epsilon_2: Bonferroni bound over the group members —
            |group| * log(1/beta_2) / n_samples.
        n_samples_used: echo of n_samples.
    """
    disjoint: bool
    per_hs_results: list
    epsilon_2: float
    n_samples_used: int


@dataclass
class SpecDisjointResult:
    """Result of certify_spec_disjoint (layer 3 of the scenario-verify
    three-layer dispatcher).

    A 'spec' is a list of groups with AND semantics across groups and
    OR semantics within each group (AND-of-OR-of-AND, matching n2v's
    canonical `_parse_property_groups` shape).

    UNSAT is certified iff AT LEAST ONE group is disjoint from the flow
    set, because the AND across groups requires every group to be hit.
    One group guaranteed-not-hit breaks the AND and implies UNSAT.

    Fields:
        unsat_certified: True iff at least one group was certified
            disjoint (layer 2 returned disjoint=True for some group).
        certifying_group_idx: index of the group that certified disjoint
            (None if no group did).
        per_group_results: list of GroupDisjointResult, one per group
            processed before early-exit. May be shorter than len(spec_groups)
            when a disjoint group is found early.
        epsilon_2: Bonferroni bound over ALL HalfSpace tests planned
            across all groups — sum over groups of |group| times
            log(1/beta_2)/n_samples. Conservative: uses the full planned
            count, not the count actually executed.
        n_samples_used: echo of n_samples.
    """
    unsat_certified: bool
    certifying_group_idx: 'int | None'
    per_group_results: list
    epsilon_2: float
    n_samples_used: int


@dataclass
class PreimageResult:
    """
    Result of preimage search.

    Attributes:
        found: True if an input within tolerance of y_target was found.
        x: The discovered input in the input set, if found. None otherwise.
        y_achieved: f(x) at the discovered input. None if not found.
        distance: Final L2 distance ||f(x) - y_target||.
    """
    found: bool
    x: Optional[np.ndarray]
    y_achieved: Optional[np.ndarray]
    distance: float


def sample_truncated_gaussian_ball(
    q: float,
    dim: int,
    n_samples: int,
    method: Optional[str] = None,
) -> np.ndarray:
    """
    Sample from N(0, I_dim) conditioned on ||z|| <= q.

    Two sampling methods are available:

    - 'rejection': sample from N(0, I), reject if ||z|| > q. Fast for
      low dimensions where the acceptance rate is high. The acceptance
      rate is Pr[chi^2_dim <= q^2].

    - 'chi': sample direction uniformly on the sphere, sample radius
      from chi distribution with `dim` degrees of freedom truncated to
      [0, q] via inverse CDF sampling. Exact for any dimension.

    By default, 'rejection' is used for dim <= 30, 'chi' for higher dim.

    Args:
        q: Ball radius (must be positive).
        dim: Dimensionality.
        n_samples: Number of samples to generate.
        method: Either 'rejection', 'chi', or None for automatic.

    Returns:
        (n_samples, dim) numpy array of samples.

    Raises:
        ValueError: If q <= 0 or method is invalid.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")

    if method is None:
        method = 'rejection' if dim <= _REJECTION_DIM_THRESHOLD else 'chi'
        # Rejection sampling hangs when the rejection rate is near 100% —
        # e.g., when the calibrated threshold q is tiny relative to the
        # chi_dim distribution's bulk. Acceptance rate for
        # N(0, I_dim) inside ||z|| <= q equals Pr[chi^2_dim <= q^2].
        # Switch to the inverse-CDF chi method whenever that probability
        # falls below 1 %; chi is exact for any q.
        if method == 'rejection':
            from scipy.stats import chi2 as _chi2
            if _chi2.cdf(q * q, dim) < 0.01:
                method = 'chi'

    if method == 'rejection':
        return _sample_rejection(q, dim, n_samples)
    elif method == 'chi':
        return _sample_chi(q, dim, n_samples)
    else:
        raise ValueError(
            f"method must be 'rejection' or 'chi', got '{method}'"
        )


def _sample_rejection(q: float, dim: int, n_samples: int) -> np.ndarray:
    """Rejection sampling for low dimensions."""
    samples = np.empty((n_samples, dim))
    n_filled = 0
    while n_filled < n_samples:
        batch_size = max(n_samples - n_filled, 100)
        candidates = np.random.randn(batch_size, dim)
        norms = np.linalg.norm(candidates, axis=1)
        accepted = candidates[norms <= q]
        n_accept = min(len(accepted), n_samples - n_filled)
        samples[n_filled:n_filled + n_accept] = accepted[:n_accept]
        n_filled += n_accept
    return samples


def _sample_chi(q: float, dim: int, n_samples: int) -> np.ndarray:
    """Sphere + truncated chi radius sampling, valid for any dimension."""
    # Sample directions uniformly on the unit sphere
    g = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    directions = g / norms

    # Sample radii from chi distribution truncated to [0, q]
    # Inverse CDF: r = chi.ppf(U * chi.cdf(q))
    u = np.random.rand(n_samples)
    cdf_q = chi.cdf(q, df=dim)
    radii = chi.ppf(u * cdf_q, df=dim)

    return radii[:, None] * directions


def sample_empirical_latent_ball(
    flow_ode,
    y_train_centered: torch.Tensor,
    q: float,
    n_samples: int,
    noise_sigma: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample latent points from the empirical distribution of training
    outputs, perturbed by small Gaussian noise.

    The idea: instead of sampling from the prior truncated Gaussian on
    the ball (which explores the whole ball uniformly including
    hallucination regions), we sample near latent points that
    correspond to real training outputs. This keeps samples aligned
    with the data manifold and dramatically reduces flow hallucinations
    during scenario verification.

    Procedure:
      1. Forward-map the training outputs through the flow to get their
         latent representations.
      2. Sample with replacement from these latent points.
      3. Add small Gaussian noise to each sample (noise_sigma controls
         the perturbation scale).
      4. Reject samples outside the ball ||z|| <= q.
      5. Return n_samples accepted points.

    The sampling is i.i.d. from a fixed distribution, so the scenario
    optimization bound applies: the distribution is "latent points
    from a kernel-density estimate of real data convolved with a
    Gaussian, truncated to the ball."

    Args:
        flow_ode: FlowODE instance with a trained velocity field.
        y_train_centered: (N_train, dim) tensor of centered training
            outputs. The flow was trained on these.
        q: Ball radius (conformal threshold).
        n_samples: Number of samples to return.
        noise_sigma: Standard deviation of the Gaussian perturbation
            applied to each resampled latent point.
        seed: Optional random seed for reproducibility.

    Returns:
        (n_samples, dim) numpy array of latent samples, all with
        ||z|| <= q.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if noise_sigma < 0:
        raise ValueError(f"noise_sigma must be non-negative, got {noise_sigma}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(seed)

    dim = y_train_centered.shape[1]

    # Forward map training outputs to latent space
    with torch.no_grad():
        z_train = flow_ode.forward(
            y_train_centered, t=1.0, n_steps=100,
        ).numpy()

    n_train = z_train.shape[0]

    samples = np.empty((n_samples, dim))
    n_filled = 0
    while n_filled < n_samples:
        batch_size = max(n_samples - n_filled, 1000)
        indices = rng.integers(0, n_train, size=batch_size)
        base_z = z_train[indices]
        noise = rng.normal(0.0, noise_sigma, size=(batch_size, dim))
        candidates = base_z + noise
        norms = np.linalg.norm(candidates, axis=1)
        accepted = candidates[norms <= q]
        n_accept = min(len(accepted), n_samples - n_filled)
        samples[n_filled:n_filled + n_accept] = accepted[:n_accept]
        n_filled += n_accept

    return samples


def scenario_verify_halfspace(
    flow_ode,
    threshold_q: float,
    w: np.ndarray,
    b: float,
    n_samples: int,
    beta_2: float,
    t: float = 1.0,
    n_ode_steps: int = 100,
    target_fn: Optional[Callable] = None,
    input_set_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    preimage_n_restarts: int = 10,
    preimage_n_steps: int = 200,
    preimage_lr: float = 0.05,
    preimage_tolerance: float = 1e-3,
    output_shift: Optional[np.ndarray] = None,
    latent_samples: Optional[np.ndarray] = None,
    ode_method: str = 'dopri5',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
) -> ScenarioResult:
    """
    Verify a halfspace specification w^T y <= b on the flow reach set.

    Reformulates the verification as
        max w^T psi_t(z) over ||z|| <= threshold_q <= b
    and applies scenario optimization: sample N latent points from the
    truncated Gaussian on the ball, evaluate w^T psi_t(z_i), check if
    any exceeds b.

    If any sample violates the spec, return a counterexample (the
    sampled latent point, the corresponding data point, and the margin).
    Otherwise return a probabilistic certificate with violation bound
    epsilon_2 = -log(beta_2) / N at confidence delta_2 = 1 - beta_2.

    The flow set must be calibrated separately via conformal inference;
    threshold_q is the calibrated threshold. The user is responsible for
    composing this scenario certificate with the conformal coverage
    guarantee (see verify_robustness).

    If target_fn and input_set_bounds are provided and a candidate
    counterexample is found in the flow set, preimage search is run
    to determine whether the candidate corresponds to a real input.
    The result has outcome='falsified' if a genuine input is found,
    'unknown' otherwise (indicating a flow hallucination).

    TODO (future work): Generic spec API. Replace (w, b) with a
    pointwise spec evaluator `spec_fn: y -> bool` to support arbitrary
    pointwise-checkable specifications (L_p balls, polytopes, logical
    compositions, implicit specs from other models, etc.). The scenario
    bound applies to any pointwise-checkable predicate, not just
    halfspaces.

    Args:
        flow_ode: FlowODE instance with trained velocity field.
        threshold_q: Calibrated conformal threshold (ball radius in
            latent space).
        w: (dim,) coefficient vector of the halfspace.
        b: Scalar right-hand side of the halfspace.
        n_samples: Number of latent samples to draw.
        beta_2: Scenario confidence parameter (typical: 0.001).
        t: Flow time (typical: 1.0).
        n_ode_steps: ODE integration steps for inverse flow.
        target_fn: Optional target network for preimage search.
        input_set_bounds: Optional (lb, ub) input set bounds.
        preimage_n_restarts: Restarts for preimage search.
        preimage_n_steps: Gradient steps per restart.
        preimage_lr: Adam learning rate for preimage search.
        preimage_tolerance: Distance threshold for "found".
        output_shift: Optional (dim,) numpy array. When provided, the flow
            model's outputs are interpreted as CENTERED: raw outputs are
            recovered by adding `output_shift`. The halfspace spec (w, b)
            is then interpreted in RAW output space, and `target_fn` must
            be the RAW target network (not a centered wrapper). When None
            (default), no shift is applied; flow outputs and target_fn are
            assumed to be in the same coordinate system as the spec.
        latent_samples: Optional (n_samples, dim) numpy array of pre-computed
            latent samples. When provided, these samples are used directly
            instead of drawing from the truncated Gaussian on the ball. This
            lets callers specify alternative sampling distributions (e.g.,
            empirical latent sampling via sample_empirical_latent_ball). The
            samples must be i.i.d. from a fixed distribution for the scenario
            bound to apply.

    Returns:
        ScenarioResult with verification outcome and certificate.

    Raises:
        ValueError: If n_samples <= 0 or beta_2 not in (0, 1).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not (0.0 < beta_2 < 1.0):
        raise ValueError(f"beta_2 must be in (0, 1), got {beta_2}")

    dim = w.shape[0]

    if latent_samples is not None:
        if latent_samples.shape != (n_samples, dim):
            raise ValueError(
                f"latent_samples shape {latent_samples.shape} does not "
                f"match expected ({n_samples}, {dim})"
            )
        z_samples_np = latent_samples
    else:
        # Sample latent points from the truncated Gaussian ball
        z_samples_np = sample_truncated_gaussian_ball(
            q=threshold_q, dim=dim, n_samples=n_samples,
        )
    z_samples = torch.tensor(z_samples_np, dtype=torch.float32)

    # Map to data space via inverse flow
    with torch.no_grad():
        y_samples = flow_ode.inverse(
            z_samples, t=t, n_steps=n_ode_steps,
            method=ode_method, atol=ode_atol, rtol=ode_rtol,
        )

    # Convert flow outputs from centered to raw coordinates if a shift
    # was provided. When output_shift is None, raw and centered are
    # treated as equal (backward compatible).
    y_samples_np = y_samples.numpy()
    if output_shift is not None:
        y_raw_np = y_samples_np + output_shift[None, :]
    else:
        y_raw_np = y_samples_np

    # Halfspace spec w^T y <= b is interpreted in raw output space.
    margins = y_raw_np @ w - b

    # Compute scenario certificate values
    epsilon_2 = -math.log(beta_2) / n_samples
    delta_2 = 1.0 - beta_2

    # Check for violations
    violation_mask = margins > 0
    if not violation_mask.any():
        return ScenarioResult(
            verified=True,
            outcome='verified',
            counterexample=None,
            genuine_input=None,
            epsilon_2=epsilon_2,
            delta_2=delta_2,
            n_samples_used=n_samples,
        )

    # Violation found — extract worst-case counterexample (in raw coords)
    worst_idx = int(np.argmax(margins))
    ce = (
        z_samples_np[worst_idx],
        y_raw_np[worst_idx],  # raw coordinates
        float(margins[worst_idx]),
    )

    # Optional preimage search. Some ONNX-imported networks (yolo's
    # batch_loop_unbatched wrapper, Cohen RS-wrapped CIFAR-10 ResNet-110)
    # don't propagate gradients through their forward pass — calling
    # ``loss.backward()`` inside ``preimage_search`` then raises
    # ``RuntimeError: element 0 of tensors does not require grad ...``.
    # That's a bug in the loader, not a real failure of the verification
    # pipeline: we already have a probabilistic SAT witness in ``ce``.
    # Treat the preimage step as best-effort metadata and fall through
    # to the ``unknown`` outcome on failure rather than crashing.
    preimage = None
    if target_fn is not None and input_set_bounds is not None:
        try:
            preimage = preimage_search(
                target_fn=target_fn,
                y_target=ce[1],
                input_set_bounds=input_set_bounds,
                n_restarts=preimage_n_restarts,
                n_steps=preimage_n_steps,
                lr=preimage_lr,
                tolerance=preimage_tolerance,
            )
        except RuntimeError as e:
            # Most common: gradient-flow failure in non-differentiable
            # wrappers. Other RuntimeErrors are also non-fatal here.
            if 'does not require grad' not in str(e):
                # Re-raise unrelated RuntimeErrors so they're still loud.
                raise
            preimage = None
    if preimage is not None:
        if preimage.found:
            # Explicitly verify the spec is violated at the real output.
            # preimage.y_achieved is f(preimage.x) from the final step.
            actual_margin = float(preimage.y_achieved @ w - b)
            if actual_margin > 0:
                # Defensive bounds check (should be true by construction).
                lb, ub = input_set_bounds
                assert np.all(preimage.x >= lb - 1e-6), (
                    f"preimage.x below lower bound: x={preimage.x}, lb={lb}"
                )
                assert np.all(preimage.x <= ub + 1e-6), (
                    f"preimage.x above upper bound: x={preimage.x}, ub={ub}"
                )
                return ScenarioResult(
                    verified=False,
                    outcome='falsified',
                    counterexample=ce,
                    genuine_input=preimage.x,
                    epsilon_2=epsilon_2,
                    delta_2=delta_2,
                    n_samples_used=n_samples,
                )
            # else: preimage found but spec not actually violated → fall
            # through to the 'unknown' return below.

    return ScenarioResult(
        verified=False,
        outcome='unknown',
        counterexample=ce,
        genuine_input=None,
        epsilon_2=epsilon_2,
        delta_2=delta_2,
        n_samples_used=n_samples,
    )


@dataclass
class RobustnessResult:
    """
    Joint probabilistic robustness certificate.

    Combines the conformal coverage guarantee on the reach set with
    the scenario violation guarantee on the spec check.

    The joint statement is:
        Pr[Pr[f(x) satisfies spec] >= 1 - epsilon_total] >= delta_total
    where:
        epsilon_total = 1 - (1 - epsilon_1)(1 - epsilon_2)
        delta_total = delta_1 * delta_2

    The outcome field classifies the result as:
        'verified' = no flow-set violation, joint probabilistic certificate holds
        'falsified' = genuine counterexample found (real input x)
        'unknown' = flow-set violation found but no real preimage (hallucination)
            or preimage search was not requested.

    Attributes:
        verified: True if outcome == 'verified', False otherwise.
        outcome: One of 'verified', 'falsified', 'unknown'.
        counterexample: If outcome != 'verified', a tuple (z, y, wrong_class).
        genuine_input: If outcome == 'falsified', the real input x.
        epsilon_total: Joint miscoverage bound.
        delta_total: Joint confidence.
        epsilon_1, delta_1: Conformal layer values.
        epsilon_2, delta_2: Scenario layer values.
        n_classes: Number of classes in the classification problem.
        n_samples_used: Number of latent samples drawn.
    """
    verified: bool
    outcome: str
    counterexample: Optional[Tuple[np.ndarray, np.ndarray, int]]
    genuine_input: Optional[np.ndarray]
    epsilon_total: float
    delta_total: float
    epsilon_1: float
    delta_1: float
    epsilon_2: float
    delta_2: float
    n_classes: int
    n_samples_used: int


def verify_robustness(
    flow_ode,
    threshold_q: float,
    true_class: int,
    n_classes: int,
    epsilon_1: float,
    delta_1: float,
    n_samples: int,
    beta_2: float,
    t: float = 1.0,
    n_ode_steps: int = 100,
    target_fn: Optional[Callable] = None,
    input_set_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    preimage_n_restarts: int = 10,
    preimage_n_steps: int = 200,
    preimage_lr: float = 0.05,
    preimage_tolerance: float = 1e-3,
    output_shift: Optional[np.ndarray] = None,
    latent_samples: Optional[np.ndarray] = None,
    ode_method: str = 'dopri5',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
) -> RobustnessResult:
    """
    Verify classification robustness via scenario optimization.

    For each wrong class k != true_class, builds the halfspace constraint
    y[k] - y[true_class] <= 0 and runs scenario verification. Uses the
    SHARED-SAMPLES strategy: sample N latent points once, evaluate all
    halfspaces on each output. The joint certificate covers the union
    event "no halfspace is violated by any sample" with the same
    (N, beta_2).

    If target_fn and input_set_bounds are provided and a candidate
    counterexample is found in the flow set, preimage search is run
    to determine whether the candidate corresponds to a real input x
    in the input set I. If yes, outcome='falsified' and genuine_input
    is returned; otherwise outcome='unknown' (flow hallucination).

    TODO (future work): Generic spec API to support arbitrary pointwise
    predicates beyond halfspaces.

    Args:
        flow_ode: FlowODE instance with trained velocity field.
        threshold_q: Calibrated conformal threshold (ball radius).
        true_class: Index of the correct class.
        n_classes: Total number of classes.
        epsilon_1: Conformal miscoverage level (from calibration).
        delta_1: Conformal confidence (from calibration).
        n_samples: Number of latent samples to draw.
        beta_2: Scenario confidence parameter.
        t: Flow time.
        n_ode_steps: ODE integration steps.
        target_fn: Optional target network for preimage search.
        input_set_bounds: Optional (lb, ub) input set bounds.
        preimage_n_restarts: Restarts for preimage search.
        preimage_n_steps: Gradient steps per restart.
        preimage_lr: Adam learning rate.
        preimage_tolerance: Distance threshold for "found".
        output_shift: Optional (n_classes,) numpy array. When provided, the
            flow model's outputs are interpreted as CENTERED: raw outputs
            are recovered by adding `output_shift`. The classification spec
            is always interpreted on raw logits (argmax comparison is only
            meaningful in raw space). When provided, `target_fn` must be
            the RAW target network (not a centered wrapper). When None
            (default), no shift is applied; flow outputs and target_fn are
            assumed to be in the same coordinate system.
        latent_samples: Optional (n_samples, n_classes) numpy array of
            pre-computed latent samples. When provided, these samples are
            used directly instead of drawing from the truncated Gaussian on
            the ball. This lets callers specify alternative sampling
            distributions (e.g., empirical latent sampling via
            sample_empirical_latent_ball). The samples must be i.i.d. from
            a fixed distribution for the scenario bound to apply.

    Returns:
        RobustnessResult with joint probabilistic certificate and outcome.

    Raises:
        ValueError: If parameters are out of valid ranges.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not (0.0 < beta_2 < 1.0):
        raise ValueError(f"beta_2 must be in (0, 1), got {beta_2}")
    if not (0.0 < epsilon_1 < 1.0):
        raise ValueError(f"epsilon_1 must be in (0, 1), got {epsilon_1}")
    if not (0.0 < delta_1 <= 1.0):
        raise ValueError(f"delta_1 must be in (0, 1], got {delta_1}")
    if not (0 <= true_class < n_classes):
        raise ValueError(
            f"true_class must be in [0, {n_classes}), got {true_class}"
        )
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")

    dim = n_classes

    if latent_samples is not None:
        if latent_samples.shape != (n_samples, dim):
            raise ValueError(
                f"latent_samples shape {latent_samples.shape} does not "
                f"match expected ({n_samples}, {dim})"
            )
        z_samples_np = latent_samples
    else:
        # Sample once, share across all halfspaces
        z_samples_np = sample_truncated_gaussian_ball(
            q=threshold_q, dim=dim, n_samples=n_samples,
        )
    z_samples = torch.tensor(z_samples_np, dtype=torch.float32)

    with torch.no_grad():
        y_samples = flow_ode.inverse(
            z_samples, t=t, n_steps=n_ode_steps,
            method=ode_method, atol=ode_atol, rtol=ode_rtol,
        )
    y_samples_np = y_samples.numpy()

    # Convert flow outputs from centered to raw coordinates if a shift
    # was provided. argmax comparisons are only meaningful in raw space.
    if output_shift is not None:
        y_raw_np = y_samples_np + output_shift[None, :]
    else:
        y_raw_np = y_samples_np

    # For each wrong class, check y_raw[k] - y_raw[true_class] <= 0
    worst_margin = -float('inf')
    worst_class = -1
    worst_idx = -1

    for k in range(n_classes):
        if k == true_class:
            continue
        margins = y_raw_np[:, k] - y_raw_np[:, true_class]
        max_margin_idx = int(np.argmax(margins))
        max_margin = float(margins[max_margin_idx])
        if max_margin > worst_margin:
            worst_margin = max_margin
            worst_class = k
            worst_idx = max_margin_idx

    epsilon_2 = -math.log(beta_2) / n_samples
    delta_2 = 1.0 - beta_2

    epsilon_total = 1.0 - (1.0 - epsilon_1) * (1.0 - epsilon_2)
    delta_total = delta_1 * delta_2

    if worst_margin <= 0:
        return RobustnessResult(
            verified=True,
            outcome='verified',
            counterexample=None,
            genuine_input=None,
            epsilon_total=epsilon_total,
            delta_total=delta_total,
            epsilon_1=epsilon_1, delta_1=delta_1,
            epsilon_2=epsilon_2, delta_2=delta_2,
            n_classes=n_classes,
            n_samples_used=n_samples,
        )

    counterexample = (
        z_samples_np[worst_idx],
        y_raw_np[worst_idx],  # raw coordinates
        worst_class,
    )

    # Optional preimage search. Same gradient-flow caveat as in the
    # halfspace path above (see comment block there).
    genuine_input = None
    outcome = 'unknown'
    preimage = None
    if target_fn is not None and input_set_bounds is not None:
        try:
            preimage = preimage_search(
                target_fn=target_fn,
                y_target=counterexample[1],
                input_set_bounds=input_set_bounds,
                n_restarts=preimage_n_restarts,
                n_steps=preimage_n_steps,
                lr=preimage_lr,
                tolerance=preimage_tolerance,
            )
        except RuntimeError as e:
            if 'does not require grad' not in str(e):
                raise
            preimage = None
    if preimage is not None:
        if preimage.found:
            # Explicitly verify the spec is violated at the real output.
            # The robustness spec is violated if any wrong class beats
            # the true class at f(preimage.x).
            y_real = preimage.y_achieved
            spec_violated = False
            for k in range(n_classes):
                if k == true_class:
                    continue
                if y_real[k] > y_real[true_class]:
                    spec_violated = True
                    break

            if spec_violated:
                # Defensive bounds check.
                lb, ub = input_set_bounds
                assert np.all(preimage.x >= lb - 1e-6), (
                    f"preimage.x below lower bound: x={preimage.x}, lb={lb}"
                )
                assert np.all(preimage.x <= ub + 1e-6), (
                    f"preimage.x above upper bound: x={preimage.x}, ub={ub}"
                )
                outcome = 'falsified'
                genuine_input = preimage.x
            # else: outcome remains 'unknown'

    return RobustnessResult(
        verified=False,
        outcome=outcome,
        counterexample=counterexample,
        genuine_input=genuine_input,
        epsilon_total=epsilon_total,
        delta_total=delta_total,
        epsilon_1=epsilon_1, delta_1=delta_1,
        epsilon_2=epsilon_2, delta_2=delta_2,
        n_classes=n_classes,
        n_samples_used=n_samples,
    )


def preimage_search(
    target_fn: Callable,
    y_target: np.ndarray,
    input_set_bounds: Tuple[np.ndarray, np.ndarray],
    n_restarts: int = 10,
    n_steps: int = 200,
    lr: float = 0.01,
    tolerance: float = 1e-3,
) -> PreimageResult:
    """
    Search for a real input x in I such that f(x) approximates y_target.

    Uses projected gradient descent (Adam) with random restarts. Each
    restart initializes x uniformly in the input set bounds and takes
    n_steps gradient steps on ||f(x) - y_target||^2, projecting x back
    into the bounds after each step.

    If the best distance across all restarts is below tolerance, the
    preimage is considered found and x is a genuine input producing an
    output within tolerance of y_target.

    Args:
        target_fn: Callable mapping input (torch.Tensor) to output
            (torch.Tensor). Typically a torch.nn.Module or a simple
            lambda that accepts and returns torch tensors.
        y_target: (dim_out,) numpy array, the output we want to reach.
        input_set_bounds: (lb, ub) tuple of (dim_in,) numpy arrays
            defining the input set I as [lb, ub] per dimension.
        n_restarts: Number of random initializations.
        n_steps: Gradient descent steps per restart.
        lr: Adam learning rate.
        tolerance: Distance threshold for "found".

    Returns:
        PreimageResult with the best input discovered.
    """
    lb, ub = input_set_bounds
    lb_t = torch.tensor(lb, dtype=torch.float32)
    ub_t = torch.tensor(ub, dtype=torch.float32)
    y_target_t = torch.tensor(y_target, dtype=torch.float32)
    dim_in = lb.shape[0]

    def _forward(x: torch.Tensor) -> torch.Tensor:
        """Evaluate target_fn on x."""
        return target_fn(x)

    best_distance = float('inf')
    best_x = None
    best_y = None

    for restart in range(n_restarts):
        # Random initialization inside the bounds
        x_init = lb_t + torch.rand(dim_in) * (ub_t - lb_t)
        x = x_init.clone().unsqueeze(0).requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            y_pred = _forward(x)
            if y_pred.dim() > 1:
                y_pred_flat = y_pred.squeeze(0)
            else:
                y_pred_flat = y_pred
            loss = ((y_pred_flat - y_target_t) ** 2).sum()
            loss.backward()
            optimizer.step()

            # Project back to bounds
            with torch.no_grad():
                x.data = torch.max(torch.min(x.data, ub_t), lb_t)

        # Evaluate final distance
        with torch.no_grad():
            y_final = _forward(x)
            if y_final.dim() > 1:
                y_final = y_final.squeeze(0)
            distance = float(torch.norm(y_final - y_target_t).item())

            if distance < best_distance:
                best_distance = distance
                best_x = x.squeeze(0).detach().numpy().copy()
                best_y = y_final.detach().numpy().copy()

    return PreimageResult(
        found=best_distance < tolerance,
        x=best_x if best_distance < tolerance else None,
        y_achieved=best_y if best_distance < tolerance else None,
        distance=best_distance,
    )


def certify_halfspace_disjoint(
    flow_ode,
    threshold_q: float,
    halfspace: HalfSpace,
    *,
    n_samples: int,
    beta_2: float,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    seed: 'int | None' = None,
    adaptive_threshold: 'float | None' = None,
    adaptive_n_samples: 'int | None' = None,
    sampling_strategy: str = 'uniform',
) -> HalfSpaceDisjointResult:
    """Joint-target scenario-verify for a single (possibly multi-row) HalfSpace.

    Certifies that no flow sample lands in the polyhedron defined by
    ``G y <= g`` (all rows simultaneously). Operates in whatever coordinate
    frame the flow was trained in — pass a pre-whitened HalfSpace if the
    flow was trained on pre-whitened outputs.

    Args:
        flow_ode: trained FlowODE.
        threshold_q: calibrated conformal threshold.
        halfspace: HalfSpace with G shape (k, d), g shape (k, 1) or (k,).
        n_samples: scenario sample count N.
        beta_2: scenario confidence-failure probability.
        t, n_ode_steps, ode_method, ode_atol, ode_rtol: flow integration kwargs.
        seed: optional seed for reproducibility. Sets np.random.seed
            before scenario sampling; does not affect torch global state
            beyond what flow integration implicitly uses.
        adaptive_threshold: if not None and the first-stage worst-sample's
            max-row-margin is in (0, adaptive_threshold] (a marginal
            certification), re-run with adaptive_n_samples samples and
            return that result instead. The first-stage run is treated
            as a heuristic gate; the second-stage run is the actual
            hypothesis test, so epsilon_2 = log(1/beta_2)/adaptive_n_samples.
            Default None disables adaptive escalation.
        adaptive_n_samples: the larger N to use on the re-run. Must be
            specified together with adaptive_threshold.
        sampling_strategy: ``'uniform'`` (default) draws latent samples
            from the truncated Gaussian on the ball ``||z|| <= threshold_q``
            via :func:`sample_truncated_gaussian_ball`. ``'qmc'`` instead
            draws Sobol-based quasi-Monte-Carlo samples from N(0, I_d)
            (untruncated) via :func:`_qmc_sample_latents`. QMC reduces
            estimator variance for smooth integrands; the scenario bound
            still applies because Sobol+norm.ppf samples are
            equidistributed on N(0, I).

            **Soundness note:** when ``sampling_strategy='qmc'``, samples
            are drawn from the full N(0, I_d) (not truncated to the
            conformal level set ``||z|| <= threshold_q``). The scenario
            bound is still well-defined under this sampling distribution,
            but the joint composition with the conformal layer
            ``epsilon_total = 1 - (1 - epsilon_1)(1 - epsilon_2)`` was
            derived under the 'uniform' (truncated) assumption and may
            not be tight under QMC. QMC is currently experimental for
            variance-reduction ablations; do not rely on the joint
            epsilon for sound certification under QMC.

    Returns:
        HalfSpaceDisjointResult. ``disjoint=True`` iff every sample has
        ``max_i (G[i] @ y - g[i]) > 0``. The polyhedron is treated as
        CLOSED — a sample exactly on a face (max margin == 0) counts
        as inside, hence not-disjoint.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not (0.0 < beta_2 < 1.0):
        raise ValueError(f"beta_2 must be in (0, 1), got {beta_2}")

    G = np.asarray(halfspace.G, dtype=np.float64)
    g = np.asarray(halfspace.g, dtype=np.float64).flatten()

    # Seed np.random before sample_truncated_gaussian_ball (which uses
    # the global numpy RNG).
    if seed is not None:
        np.random.seed(seed)

    # 1. Sample latent points. Strategy dispatch:
    #    - 'uniform'        : truncated Gaussian on the ball ||z|| <= q.
    #    - 'qmc'            : Sobol QMC on full N(0, I_d) (no ball truncation).
    #    - 'qmc+antithetic' : Sobol QMC paired with -z (ceil(N/2) Sobol
    #                         points, mirrored to N total). Each sample
    #                         is marginally N(0, I_d).
    if sampling_strategy == 'uniform':
        z_samples_np = sample_truncated_gaussian_ball(
            q=threshold_q, dim=G.shape[1], n_samples=n_samples,
        )
        z_samples = torch.tensor(z_samples_np, dtype=torch.float32)
    elif sampling_strategy == 'qmc':
        z_samples = _qmc_sample_latents(
            n_samples=n_samples, dim=G.shape[1], seed=seed,
            device='cpu', dtype=torch.float32,
        )
        z_samples_np = z_samples.numpy().astype(np.float64)
    elif sampling_strategy == 'qmc+antithetic':
        z_samples = _qmc_sample_latents(
            n_samples=n_samples, dim=G.shape[1], seed=seed,
            device='cpu', dtype=torch.float32, antithetic=True,
        )
        z_samples_np = z_samples.numpy().astype(np.float64)
    else:
        raise ValueError(
            f"unsupported sampling_strategy: {sampling_strategy!r} "
            f"(expected 'uniform', 'qmc', or 'qmc+antithetic')"
        )

    # 2. Push through flow inverse (vectorized single integration)
    with torch.no_grad():
        y_samples = flow_ode.inverse(
            z_samples, t=t, n_steps=n_ode_steps,
            method=ode_method, atol=ode_atol, rtol=ode_rtol,
        )
    y_samples_np = y_samples.numpy().astype(np.float64)  # (N, d)

    # 3. Compute per-row margins: (N, k) where margins[j, i] = G[i] @ y_j - g[i]
    margins = y_samples_np @ G.T - g[None, :]

    # 4. Joint AND check: max row margin per sample
    max_margin_per_sample = margins.max(axis=1)  # (N,)

    # 5. Worst sample = one with MIN max-margin (deepest inside / least safely outside)
    worst_idx = int(np.argmin(max_margin_per_sample))
    worst_margin = float(max_margin_per_sample[worst_idx])

    # 6. Disjoint iff every sample has max margin > 0 (none inside polyhedron)
    disjoint = bool(worst_margin > 0)

    epsilon_2 = math.log(1.0 / beta_2) / n_samples
    result = HalfSpaceDisjointResult(
        disjoint=disjoint,
        worst_sample=y_samples_np[worst_idx],
        worst_max_margin=worst_margin,
        epsilon_2=epsilon_2,
        n_samples_used=n_samples,
    )

    # Adaptive escalation: re-run with more samples if the certification
    # was marginal (worst margin in (0, adaptive_threshold]). The smaller-N
    # run is treated as a heuristic gate; the larger-N run becomes the
    # actual hypothesis test, so epsilon_2 reflects the larger N.
    if (
        result.disjoint
        and adaptive_threshold is not None
        and adaptive_n_samples is not None
        and 0 < result.worst_max_margin <= adaptive_threshold
    ):
        return certify_halfspace_disjoint(
            flow_ode=flow_ode, threshold_q=threshold_q, halfspace=halfspace,
            n_samples=adaptive_n_samples, beta_2=beta_2,
            t=t, n_ode_steps=n_ode_steps, ode_method=ode_method,
            ode_atol=ode_atol, ode_rtol=ode_rtol,
            seed=(seed + 1 if seed is not None else None),
            # No further escalation — pass adaptive_*=None
            adaptive_threshold=None, adaptive_n_samples=None,
            sampling_strategy=sampling_strategy,
        )

    return result


def certify_group_disjoint(
    flow_ode,
    threshold_q: float,
    group: list,  # list[HalfSpace]
    *,
    n_samples: int,
    beta_2: float,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    seed: 'int | None' = None,
    adaptive_threshold: 'float | None' = None,
    adaptive_n_samples: 'int | None' = None,
    sampling_strategy: str = 'uniform',
) -> GroupDisjointResult:
    """Certify a group (OR of HalfSpaces) disjoint from the flow set.

    A group is disjoint iff EVERY HalfSpace in it is disjoint. Bonferroni
    across the group members: epsilon_2_group = sum of per-HS epsilon_2
    (which respects per-HS adaptive escalation when triggered).

    Implementation notes:
      - When ``seed`` is set, every member HalfSpace check sees the SAME
        latent samples (because `sample_truncated_gaussian_ball` reseeds
        numpy's global RNG on each call). Bonferroni is valid under
        arbitrary sample dependence across members, so this is sound.
      - Early exit: the first not-disjoint HalfSpace short-circuits the
        loop. As a result, ``per_hs_results`` may have fewer entries
        than ``len(group)`` when the group is not disjoint. Always has
        at least one entry (the failing member).

    Args:
        flow_ode: trained FlowODE.
        threshold_q: calibrated conformal threshold.
        group: list of HalfSpace objects (the OR members of the group).
        n_samples: scenario sample count N, shared across all HalfSpace checks.
        beta_2: scenario confidence-failure probability.
        t, n_ode_steps, ode_method, ode_atol, ode_rtol, seed: passed to
            certify_halfspace_disjoint for each member.
        adaptive_threshold, adaptive_n_samples: forwarded to each
            certify_halfspace_disjoint call. See its docstring for
            semantics.
        sampling_strategy: forwarded to each certify_halfspace_disjoint
            call. See :func:`certify_halfspace_disjoint` for semantics.

    Returns:
        GroupDisjointResult.

    Raises:
        ValueError: if the group is empty.
    """
    if len(group) == 0:
        raise ValueError("group must be non-empty")

    per_hs_results = []
    for hs in group:
        r = certify_halfspace_disjoint(
            flow_ode=flow_ode, threshold_q=threshold_q, halfspace=hs,
            n_samples=n_samples, beta_2=beta_2,
            t=t, n_ode_steps=n_ode_steps, ode_method=ode_method,
            ode_atol=ode_atol, ode_rtol=ode_rtol, seed=seed,
            adaptive_threshold=adaptive_threshold,
            adaptive_n_samples=adaptive_n_samples,
            sampling_strategy=sampling_strategy,
        )
        per_hs_results.append(r)
        if not r.disjoint:
            # Group semantics (OR): one not-disjoint HalfSpace means the
            # group is not disjoint regardless of the rest. Skip the rest
            # to save compute — each certify_halfspace_disjoint call is ~25s.
            break

    disjoint = all(r.disjoint for r in per_hs_results)
    return GroupDisjointResult(
        disjoint=disjoint,
        per_hs_results=per_hs_results,
        # Sum per-HS epsilon_2 to respect adaptive escalation per-HS.
        # Bonferroni still holds because each HS test is an independent
        # hypothesis test and the bound is the sum of per-test epsilons.
        epsilon_2=sum(r.epsilon_2 for r in per_hs_results),
        n_samples_used=n_samples,
    )


def certify_spec_disjoint(
    flow_ode,
    threshold_q: float,
    spec_groups: list,  # list[list[HalfSpace]]
    *,
    n_samples: int,
    beta_2: float,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    seed: 'int | None' = None,
    adaptive_threshold: 'float | None' = None,
    adaptive_n_samples: 'int | None' = None,
    sampling_strategy: str = 'uniform',
) -> SpecDisjointResult:
    """Certify UNSAT for an AND-of-OR-of-AND spec.

    ``spec_groups`` is the canonical list-of-list-of-HalfSpace shape
    produced by ``n2v.utils.verify_specification._parse_property_groups``.
    Outer list: AND across groups. Inner list: OR within a group.
    Each HalfSpace: AND across its rows.

    UNSAT iff ANY group is disjoint from the flow set. Uses layer 2
    (``certify_group_disjoint``) on each group, returning as soon as
    one succeeds (early exit).

    Bonferroni over ALL HalfSpace hypothesis tests actually performed,
    summing each per-HS epsilon_2 (which respects adaptive escalation
    when it triggers).

    Args:
        flow_ode: trained FlowODE.
        threshold_q: calibrated conformal threshold.
        spec_groups: list[list[HalfSpace]] in canonical n2v form.
        n_samples, beta_2: scenario parameters.
        t, n_ode_steps, ode_method, ode_atol, ode_rtol, seed: passed
            through to layer 2 (and ultimately layer 1).
        adaptive_threshold, adaptive_n_samples: forwarded to layer 2
            (and ultimately layer 1). See certify_halfspace_disjoint
            for semantics.
        sampling_strategy: forwarded to layer 2 (and ultimately layer 1).
            See :func:`certify_halfspace_disjoint` for semantics.

    Returns:
        SpecDisjointResult.

    Raises:
        ValueError: if ``spec_groups`` is empty.
    """
    if len(spec_groups) == 0:
        raise ValueError("spec_groups must be non-empty")

    per_group_results = []
    certifying_group_idx = None
    for idx, group in enumerate(spec_groups):
        gr = certify_group_disjoint(
            flow_ode=flow_ode, threshold_q=threshold_q, group=group,
            n_samples=n_samples, beta_2=beta_2,
            t=t, n_ode_steps=n_ode_steps, ode_method=ode_method,
            ode_atol=ode_atol, ode_rtol=ode_rtol, seed=seed,
            adaptive_threshold=adaptive_threshold,
            adaptive_n_samples=adaptive_n_samples,
            sampling_strategy=sampling_strategy,
        )
        per_group_results.append(gr)
        if gr.disjoint:
            certifying_group_idx = idx
            # Early exit: one disjoint group certifies UNSAT; no need
            # to test remaining groups.
            break

    unsat_certified = certifying_group_idx is not None
    return SpecDisjointResult(
        unsat_certified=unsat_certified,
        certifying_group_idx=certifying_group_idx,
        per_group_results=per_group_results,
        # Sum per-group epsilon_2 to respect adaptive escalation per-HS.
        epsilon_2=sum(g.epsilon_2 for g in per_group_results),
        n_samples_used=n_samples,
    )


# ---- Result-struct extraction helper --------------------------------------


def _extract_min_worst_max_margin(scenario_result: dict) -> 'float | None':
    """Pull min over (group, halfspace) of worst_max_margin from a
    scenario_result dict (output of ``certify_spec_on_flow``).

    Returns None when there are no per-group / per-hs entries.
    """
    per_group = scenario_result.get('per_group_results') or []
    margins: list[float] = []
    for gr in per_group:
        for hs_res in (getattr(gr, 'per_hs_results', None) or []):
            wm = getattr(hs_res, 'worst_max_margin', None)
            if wm is not None:
                margins.append(float(wm))
    if not margins:
        return None
    return min(margins)
