"""
Model-agnostic probabilistic reachability via surrogate-based conformal inference.

Public API:
    conformal_reach(model, input_box, config) -> ProbabilisticBox

Trains a surrogate (``'naive'`` or ``'clipping_block'``) on outputs sampled
from the input box, then inflates the surrogate's bounding box using the
conformal rank-``ell`` nonconformity threshold to obtain a probabilistic
reach set with the guarantee
``Pr[ Pr[f(x) in box] > 1 - epsilon ] > delta``.

Works with any model that can be called as ``y = model(x)`` on numpy
arrays — PyTorch ``nn.Module`` (auto-wrapped), TensorFlow, JAX, ONNX
sessions, or even remote HTTP-API callables. Sibling to
:func:`n2v.probabilistic.flow_reach`; both share the
:meth:`NeuralNetwork.reach` dispatch surface (``method='conformal'`` and
``method='flow_matching'`` respectively).
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from n2v.probabilistic.conformal import conformal_inference
from n2v.probabilistic.surrogates.naive import NaiveSurrogate
from n2v.probabilistic.surrogates.clipping_block import (
    BatchedClippingBlockSurrogate,
)
from n2v.sets import Box
from n2v.sets.probabilistic_box import ProbabilisticBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformalReachConfig:
    """Configuration for surrogate-based conformal probabilistic reachability.

    Used with :func:`conformal_reach` and with :meth:`NeuralNetwork.reach`
    via ``method='conformal'``. Co-located with :func:`conformal_reach`
    (mirrors the way :class:`FlowReachConfig` lives next to
    :func:`flow_reach`).

    Attributes:
        surrogate: Surrogate family — ``'clipping_block'`` (default;
            projects onto convex hull of training outputs, tighter) or
            ``'naive'`` (simple center-based, fast but looser).
        m: Calibration set size; larger ``m`` yields tighter confidence.
            Default ``8000``.
        ell: Rank parameter; the ``ell``-th smallest score is the
            calibrated threshold. ``None`` defaults to ``m - 1``. Must
            satisfy ``1 <= ell <= m``.
        epsilon: Miscoverage level for the conformal guarantee. Must be
            in ``(0, 1)``. Default ``0.001``.
        training_samples: Number of samples used to fit the surrogate.
            ``None`` defaults to ``m // 2``.
        pca_components: Reduce output dimensionality via PCA before
            surrogate fitting. ``None`` disables PCA. Useful when the
            output dim is very large (e.g. semantic segmentation).
        batch_size: Inference batch size. Default ``100``.
        seed: Master RNG seed for sampling. ``None`` leaves NumPy global
            state untouched (caller may seed externally).
        verbose: Emit progress logs. Default ``False``.
    """

    surrogate: Literal['naive', 'clipping_block'] = 'clipping_block'
    m: int = 8000
    ell: Optional[int] = None
    epsilon: float = 0.001
    training_samples: Optional[int] = None
    pca_components: Optional[int] = None
    batch_size: int = 100
    seed: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        if self.ell is None:
            object.__setattr__(self, 'ell', self.m - 1)
        if self.training_samples is None:
            object.__setattr__(self, 'training_samples', self.m // 2)
        if self.surrogate not in ('naive', 'clipping_block'):
            raise ValueError(
                f"ConformalReachConfig.surrogate must be 'naive' or "
                f"'clipping_block', got {self.surrogate!r}"
            )
        if self.m < 1:
            raise ValueError(
                f"ConformalReachConfig.m must be >= 1, got {self.m}"
            )
        if not 1 <= self.ell <= self.m:
            raise ValueError(
                f"ConformalReachConfig.ell must be in [1, {self.m}], "
                f"got {self.ell}"
            )
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError(
                f"ConformalReachConfig.epsilon must be in (0, 1), "
                f"got {self.epsilon}"
            )
        if self.training_samples < 1:
            raise ValueError(
                f"ConformalReachConfig.training_samples must be >= 1, "
                f"got {self.training_samples}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"ConformalReachConfig.batch_size must be >= 1, "
                f"got {self.batch_size}"
            )
        if self.pca_components is not None and self.pca_components < 1:
            raise ValueError(
                f"ConformalReachConfig.pca_components must be >= 1 or "
                f"None, got {self.pca_components}"
            )


def _inverse_transform_bounds(pca: object, lb_reduced: np.ndarray,
                              ub_reduced: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Map reduced-space box bounds to original space via interval arithmetic.

    For the linear map ``y[k] = mean[k] + sum_j x[j] * A[j, k]``:

        lb[k] = mean[k] + sum_j min(lb_reduced[j] * A[j, k],
                                    ub_reduced[j] * A[j, k])
        ub[k] = mean[k] + sum_j max(lb_reduced[j] * A[j, k],
                                    ub_reduced[j] * A[j, k])

    Args:
        pca: Fitted DeflationPCA with ``components_`` ``(N, n)`` and
            ``mean_`` ``(n,)``.
        lb_reduced: Lower bounds in reduced space, shape ``(N,)``.
        ub_reduced: Upper bounds in reduced space, shape ``(N,)``.

    Returns:
        Tuple of ``(lb_original, ub_original)``, each of shape ``(n,)``.
    """
    A = pca.components_  # (N, n)

    products1 = lb_reduced[:, np.newaxis] * A  # (N, n)
    products2 = ub_reduced[:, np.newaxis] * A  # (N, n)

    mins = np.minimum(products1, products2)
    maxs = np.maximum(products1, products2)

    lb = pca.mean_ + np.sum(mins, axis=0)
    ub = pca.mean_ + np.sum(maxs, axis=0)
    return lb, ub


def _module_as_numpy_callable(model: nn.Module) -> Callable:
    """Wrap an ``nn.Module`` so it accepts and returns numpy arrays.

    Device-aware: the wrapped callable forwards on the module's device
    (preserving GPU acceleration) and converts the output back to a CPU
    numpy array. Local to this module; promote to a shared helper if a
    third consumer ever appears.
    """
    def fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                try:
                    target_device = next(model.buffers()).device
                except StopIteration:
                    target_device = torch.device('cpu')
            x_t = torch.as_tensor(x_np, dtype=torch.float32).to(target_device)
            y_t = model(x_t)
            return y_t.detach().cpu().numpy()
    return fn


def conformal_reach(
    model: Union[Callable[[np.ndarray], np.ndarray], nn.Module],
    input_box: Box,
    config: Optional[ConformalReachConfig] = None,
    **kwargs,
) -> ProbabilisticBox:
    """Probabilistic reachability via surrogate + conformal calibration.

    Computes a probabilistic reach set with the guarantee
    ``Pr[ Pr[f(x) in box] > 1 - epsilon ] > delta``, where ``delta`` is
    determined by the rank parameter ``ell`` and the calibration set
    size ``m`` via Beta-binomial conformal inference.

    Works with ANY model that can be called as ``y = model(x)``.

    Args:
        model: Either a numpy callable (``y_np = model(x_np)`` — any
            framework: PyTorch via wrapper, TensorFlow, JAX, ONNX
            session, remote API, ...) or a PyTorch ``nn.Module``.
            ``nn.Module`` keeps its device for the forward pass (GPU
            acceleration preserved).
        input_box: :class:`n2v.sets.Box` defining the input region;
            samples are drawn uniformly from it.
        config: :class:`ConformalReachConfig` with hyperparameters.
            ``None`` (the default) builds a config from any ``**kwargs``
            passed, falling back to all-defaults when no kwargs are
            given. The two styles are mutually exclusive — passing both
            ``config=`` and overlapping kwargs raises ``TypeError``.
        **kwargs: Equivalent to constructing
            ``config=ConformalReachConfig(**kwargs)``. Supports the
            call-style ``conformal_reach(model, box, m=8000,
            epsilon=0.001, ...)``.

    Returns:
        :class:`ProbabilisticBox` containing:
        - Lower and upper bounds on the reachable set.
        - Guarantee parameters (``m``, ``ell``, ``epsilon``).
        - Computed coverage and confidence levels.

    Notes:
        - ``'clipping_block'`` surrogate is generally tighter than
          ``'naive'`` but costs more (LP per calibration sample).
        - Use ``pca_components`` for very high-dim outputs (e.g.,
          semantic segmentation).
        - When ``seed`` is provided the global NumPy RNG state is set,
          which gives reproducibility but may surprise callers with
          their own RNG sequence. To avoid that, seed externally and
          leave ``seed=None``.

    Example:
        >>> import torch
        >>> from n2v.probabilistic import conformal_reach, ConformalReachConfig
        >>> from n2v.sets import Box
        >>>
        >>> # With an nn.Module (auto-wrapped):
        >>> result = conformal_reach(my_torch_model, Box(lb, ub),
        ...                          ConformalReachConfig(m=8000, epsilon=0.001))
        >>>
        >>> # With any callable:
        >>> model_fn = lambda x: my_jax_model(params, x)
        >>> result = conformal_reach(model_fn, Box(lb, ub), m=8000, seed=42)
        >>>
        >>> print(result.get_guarantee_string())

    See Also:
        :meth:`n2v.nn.NeuralNetwork.reach` — the OO entry point that
        wraps this function. Use it when you already have a
        ``NeuralNetwork`` instance and want to share the dispatch
        surface with sound reach methods:

            net = NeuralNetwork(model)
            pbox = net.reach(box, method='conformal',
                             config=ConformalReachConfig(m=8000))

        The two surfaces are equivalent and produce identical results
        on the same ``(model, box, config)``; choose based on whether
        your callsite is OO-style or free-function-style.
    """
    if config is not None and kwargs:
        raise TypeError(
            "pass either config= or ConformalReachConfig kwargs, not both "
            f"(got config={type(config).__name__} and kwargs={list(kwargs)})"
        )
    if config is None:
        config = ConformalReachConfig(**kwargs)

    if not isinstance(input_box, Box):
        raise TypeError(
            f"input_box must be a Box, got {type(input_box).__name__}"
        )

    if isinstance(model, nn.Module):
        model_fn = _module_as_numpy_callable(model)
    else:
        model_fn = model

    if config.seed is not None:
        np.random.seed(config.seed)

    input_dim = input_box.dim

    if config.verbose:
        logger.info("Probabilistic Verification (conformal_reach)")
        logger.debug(f"  Input dimension: {input_dim}")
        logger.info(f"  Calibration size m: {config.m}")
        logger.info(f"  Rank ell: {config.ell}")
        logger.info(f"  Miscoverage epsilon: {config.epsilon}")
        logger.info(f"  Surrogate: {config.surrogate}")

    # ----- Step 1: training samples + outputs -----
    if config.verbose:
        logger.info(
            f"Step 1: Generating {config.training_samples} training samples..."
        )
    training_inputs = _sample_from_box(input_box, config.training_samples)
    training_outputs = _batched_inference(model_fn, training_inputs,
                                          config.batch_size)
    output_dim = training_outputs.shape[1]
    if config.verbose:
        logger.debug(f"  Output dimension: {output_dim}")

    # ----- Step 2: optional PCA -----
    pca = None
    if (config.pca_components is not None
            and config.pca_components < output_dim):
        if config.verbose:
            logger.info(
                f"Step 2: Reducing dimension {output_dim} -> "
                f"{config.pca_components} via PCA..."
            )
        from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA
        pca = DeflationPCA(n_components=config.pca_components,
                           verbose=config.verbose)
        training_outputs_reduced = pca.fit_transform(training_outputs)
    else:
        training_outputs_reduced = training_outputs

    # ----- Step 3: surrogate fit -----
    if config.verbose:
        logger.info(f"Step 3: Fitting {config.surrogate} surrogate...")
    if config.surrogate == 'naive':
        surr = NaiveSurrogate()
    else:
        surr = BatchedClippingBlockSurrogate(batch_size=1000,
                                             verbose=config.verbose)
    surr.fit(training_outputs_reduced)
    surrogate_lb_reduced, surrogate_ub_reduced = surr.get_bounds()

    # Lift the surrogate bounding box to full output space. Without PCA
    # the reduced and full spaces are the same and no lift is needed.
    if pca is not None:
        surrogate_lb_full, surrogate_ub_full = _inverse_transform_bounds(
            pca, surrogate_lb_reduced, surrogate_ub_reduced
        )
    else:
        surrogate_lb_full = surrogate_lb_reduced
        surrogate_ub_full = surrogate_ub_reduced
    if config.verbose:
        logger.debug("  Surrogate bounds computed")

    # ----- Step 4: calibration samples + outputs -----
    if config.verbose:
        logger.info(f"Step 4: Generating {config.m} calibration samples...")
    calibration_inputs = _sample_from_box(input_box, config.m)
    calibration_outputs = _batched_inference(model_fn, calibration_inputs,
                                             config.batch_size)
    if pca is not None:
        calibration_outputs_reduced = pca.transform(calibration_outputs)
    else:
        calibration_outputs_reduced = calibration_outputs

    # ----- Step 5: errors in full output space -----
    # Faithful to Paper 2 §3.2: q(x) = f(x) - g(x), g(x) = A · CLP(A^T · f(x))
    # lives in full output space. Computing errors in reduced space would
    # silently drop the PCA residual (I - A A^T) f(x), breaking coverage.
    if config.verbose:
        logger.info("Step 5: Computing calibration errors (full space)...")
    calibration_projections_reduced = surr.predict(calibration_outputs_reduced)
    if pca is not None:
        calibration_projections_full = pca.inverse_transform(
            calibration_projections_reduced
        )
    else:
        calibration_projections_full = calibration_projections_reduced
    calibration_errors = calibration_outputs - calibration_projections_full

    # Fast path: clipping_block without PCA — training outputs are convex
    # hull vertices, project to themselves, errors are zero.
    if config.surrogate == 'clipping_block' and pca is None:
        training_errors = np.zeros_like(training_outputs)
    else:
        training_projections_reduced = surr.predict(training_outputs_reduced)
        if pca is not None:
            training_projections_full = pca.inverse_transform(
                training_projections_reduced
            )
        else:
            training_projections_full = training_projections_reduced
        training_errors = training_outputs - training_projections_full

    # ----- Step 6: conformal inference -----
    if config.verbose:
        logger.info("Step 6: Running conformal inference...")
    guarantee = conformal_inference(
        training_errors=training_errors,
        calibration_errors=calibration_errors,
        m=config.m,
        ell=config.ell,
        epsilon=config.epsilon,
    )
    if config.verbose:
        logger.info(f"  Coverage: {guarantee.coverage:.4f}")
        logger.info(f"  Confidence: {guarantee.confidence:.4f}")
        logger.info(f"  Threshold R_ell: {guarantee.threshold:.4f}")

    # ----- Step 7: final bounds -----
    if config.verbose:
        logger.info("Step 7: Computing final bounds...")
    final_lb = surrogate_lb_full - guarantee.inflation
    final_ub = surrogate_ub_full + guarantee.inflation

    # ----- Step 8: build ProbabilisticBox -----
    result = ProbabilisticBox(
        lb=final_lb, ub=final_ub,
        m=config.m, ell=config.ell, epsilon=config.epsilon,
    )
    if config.verbose:
        logger.info(f"Result: {result}")
        logger.info(result.get_guarantee_string())
    return result


def _sample_from_box(box: Box, n_samples: int) -> np.ndarray:
    """Sample uniformly from a Box. Returns ``(n_samples, dim)`` float32."""
    lb = box.lb.flatten()
    ub = box.ub.flatten()
    dim = box.dim
    samples = np.random.uniform(lb, ub, size=(n_samples, dim))
    return samples.astype(np.float32)


def _batched_inference(
    model: Callable,
    inputs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Run model inference in batches. Flattens non-2D outputs.

    Args:
        model: Numpy-callable model.
        inputs: ``(n, input_dim)`` array.
        batch_size: Batch size.

    Returns:
        ``(n, output_dim)`` array.
    """
    n = inputs.shape[0]
    outputs = []
    for i in range(0, n, batch_size):
        batch = inputs[i:i + batch_size]
        batch_output = model(batch)
        if batch_output.ndim > 2:
            batch_output = batch_output.reshape(batch_output.shape[0], -1)
        outputs.append(batch_output)
    return np.vstack(outputs)
