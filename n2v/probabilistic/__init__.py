"""
Probabilistic verification module for n2v.

Provides two model-agnostic probabilistic reachability methods backed by
conformal inference:

  * ``conformal_reach`` — surrogate-based reach (renamed from the legacy
    ``verify`` at the post-NeurIPS cleanup refactor). Returns a
    :class:`ProbabilisticBox`.
  * ``flow_reach`` — flow-matching-based reach. Returns a
    :class:`ProbabilisticSet`.

Both accept any callable ``y = model(x)`` (PyTorch ``nn.Module``,
TensorFlow, JAX, ONNX session, ...) and are also dispatched via
:meth:`NeuralNetwork.reach` (``method='conformal'`` /
``method='flow_matching'``).

Example:
    >>> from n2v.probabilistic import conformal_reach, ConformalReachConfig
    >>> from n2v.sets import Box
    >>>
    >>> result = conformal_reach(
    ...     my_model, Box(lb, ub),
    ...     ConformalReachConfig(m=8000, epsilon=0.001),
    ... )
    >>> print(f"Coverage: {result.coverage}, Confidence: {result.confidence}")
"""

from n2v.probabilistic.conformal_reach import (
    ConformalReachConfig,
    conformal_reach,
)
from n2v.probabilistic.flow.reach import FlowReachConfig, flow_reach
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.sets.probabilistic_box import ProbabilisticBox
from n2v.probabilistic.conformal import (
    ConformalGuarantee,
    compute_confidence,
    compute_normalization,
    compute_nonconformity_scores,
    compute_threshold,
    compute_inflation,
    conformal_inference,
)

__all__ = [
    'conformal_reach',
    'ConformalReachConfig',
    'flow_reach',
    'FlowReachConfig',
    'ProbabilisticSet',
    'ProbabilisticBox',
    'ConformalGuarantee',
    'compute_confidence',
    'compute_normalization',
    'compute_nonconformity_scores',
    'compute_threshold',
    'compute_inflation',
    'conformal_inference',
]
