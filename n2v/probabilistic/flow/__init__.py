"""
Public API for flow-based probabilistic verification.

This module exposes the verification entry points that combine
flow-matching velocity fields with conformal / Monte Carlo / adaptive
multilevel splitting (AMLS) certification procedures, alongside the
underlying primitives (scores, calibration, flow model, ODE integrator,
training loop, latent samplers, and visualization helpers) that those
entry points are built from.

Top-level certification entry points:

* :func:`amls_certify_spec` — AMLS certification of a half-space spec.
* :func:`amls_bounded_certify_spec` / :func:`amls_bounded_certify_spec_union`
  — bounded-input AMLS variants (single spec / union of specs).
* :func:`is_tilted_certify_spec` — tilted importance-sampling certification.
* :func:`raw_mc_certify_spec` — raw Monte Carlo baseline certification.

Supporting primitives:

* Nonconformity scores (:class:`NonconformityScore`, :class:`HyperrectScore`,
  :class:`EllipsoidScore`, :class:`BallScore`, :class:`FlowScore`,
  :class:`GMMScore`).
* Conformal calibration (:func:`calibrate`, :func:`compute_guarantee`).
* Probabilistic reachable set container (:class:`ProbabilisticSet`).
* Flow model + integrator (:class:`VelocityField`, :class:`DiTLiteVelocityField`,
  :class:`FlowODE`) and training (:func:`train_flow`).
* Latent-ball samplers used by the certification routines
  (:func:`sample_truncated_gaussian_ball`,
  :func:`sample_empirical_latent_ball`).
* Star-set visualization helpers.
"""

from n2v.probabilistic.flow.scores import (
    NonconformityScore,
    HyperrectScore,
    EllipsoidScore,
    BallScore,
    FlowScore,
    GMMScore,
)
from n2v.probabilistic.flow.calibrate import calibrate, compute_guarantee
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.probabilistic.flow.model import VelocityField, DiTLiteVelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.train import train_flow
from n2v.probabilistic.flow.scenario_verify import (
    sample_truncated_gaussian_ball,
    sample_empirical_latent_ball,
)
from n2v.probabilistic.flow.amls import amls_certify_spec
from n2v.probabilistic.flow.amls_bounded import (
    amls_bounded_certify_spec,
    amls_bounded_certify_spec_union,
)
from n2v.probabilistic.flow.importance_sampling import is_tilted_certify_spec
from n2v.probabilistic.flow.raw_mc_uniform import raw_mc_certify_spec
from n2v.probabilistic.flow.star_viz import (
    render_star_union_3d,
    render_star_convex_hull_3d,
    render_star_union_isosurface_3d,
    render_probabilistic_set_isosurface_3d,
)

__all__ = [
    'NonconformityScore',
    'HyperrectScore',
    'EllipsoidScore',
    'BallScore',
    'FlowScore',
    'GMMScore',
    'calibrate',
    'compute_guarantee',
    'ProbabilisticSet',
    'VelocityField',
    'DiTLiteVelocityField',
    'FlowODE',
    'train_flow',
    'sample_truncated_gaussian_ball',
    'sample_empirical_latent_ball',
    'amls_certify_spec',
    'amls_bounded_certify_spec',
    'amls_bounded_certify_spec_union',
    'is_tilted_certify_spec',
    'raw_mc_certify_spec',
    'render_star_union_3d',
    'render_star_convex_hull_3d',
    'render_star_union_isosurface_3d',
    'render_probabilistic_set_isosurface_3d',
]
