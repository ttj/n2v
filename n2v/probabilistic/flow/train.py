"""
OT-CFM training loop for flow matching.

Trains a velocity field to transport data toward a standard Gaussian.
Pairs (x0 ~ N(0,I), x1 = data), interpolates x_t = (1-t)*x0 + t*x1,
and regresses v_theta(t, x_t) against the target velocity x1 - x0.

Supports two OT coupling methods:
  - Hungarian (exact, O(n^3), CPU-only via scipy)
  - Sinkhorn (approximate, GPU-friendly, pure tensor ops)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Union

from n2v.probabilistic.flow.model import VelocityField


def compute_adaptive_sinkhorn_reg(
    training_outputs: torch.Tensor,
    alpha: float = 0.1,
    n_probe: int = 256,
    floor: float = 1e-6,
) -> float:
    """
    Compute an adaptive Sinkhorn regularization strength based on data scale.

    The Sinkhorn kernel is K_ij = exp(-||x0_i - x1_j||^2 / reg). For numerical
    stability in float32 we need cost/reg to stay below roughly 50-80, so K
    doesn't underflow. This helper probes the actual cost by sampling a batch
    of Gaussian noise and computing pairwise squared distances against the
    training data — this matches the cost Sinkhorn actually sees during
    training, making the resulting reg numerically stable at any data scale.

    Args:
        training_outputs: (n, d) tensor of training samples.
        alpha: Proportionality constant. Smaller alpha = sharper coupling
            but closer to numerical underflow. Default 0.1 (tuned via the
            alpha sweep in docs/audits/2026-04-14-adaptive-sinkhorn-alpha-sweep.md).
        n_probe: Number of probe samples for the distance estimation.
        floor: Minimum reg value to prevent returning zero on degenerate data.

    Returns:
        A scalar float suitable for passing as `reg` to sinkhorn_coupling.
    """
    with torch.no_grad():
        n = training_outputs.shape[0]
        probe_data = training_outputs[: min(n_probe, n)]
        probe_noise = torch.randn_like(probe_data)
        probe_cost_sq = (
            torch.cdist(probe_noise, probe_data, p=2) ** 2
        ).median().item()
    return max(floor, probe_cost_sq * alpha)


def ot_coupling(
    x0: torch.Tensor, x1: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minibatch OT coupling via the Hungarian algorithm.

    Finds the permutation of (x0, x1) that minimizes total L2 cost.
    Requires CPU (uses scipy). O(n^3) per batch.

    Args:
        x0: (batch, dim) source samples.
        x1: (batch, dim) target samples.

    Returns:
        (x0_permuted, x1_permuted) with optimal coupling.
    """
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(x0, x1, p=2).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return x0[row_ind], x1[col_ind]


def sinkhorn_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float,
    max_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minibatch OT coupling via the Sinkhorn algorithm.

    Approximates optimal transport via entropic regularization.
    Pure tensor ops — works on CPU and GPU. O(n^2) per iteration.

    Args:
        x0: (batch, dim) source samples.
        x1: (batch, dim) target samples.
        reg: Entropic regularization strength (smaller = closer to exact OT).
        max_iters: Number of Sinkhorn iterations.

    Returns:
        (x0_permuted, x1_permuted) with approximate optimal coupling.
    """
    with torch.no_grad():
        cost = torch.cdist(x0, x1, p=2)
        K = torch.exp(-cost / reg)

        # Sinkhorn iterations (row/column normalization)
        u = torch.ones(x0.shape[0], device=x0.device)
        for _ in range(max_iters):
            v = 1.0 / (K.T @ u + 1e-8)
            u = 1.0 / (K @ v + 1e-8)

        # Transport plan
        plan = torch.diag(u) @ K @ torch.diag(v)

        # Extract hard assignment via argmax per row
        col_ind = plan.argmax(dim=1)
        row_ind = torch.arange(x0.shape[0], device=x0.device)

    return x0[row_ind], x1[col_ind]


def train_flow(
    velocity_field: VelocityField,
    training_outputs: torch.Tensor,
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    coupling: str = 'hungarian',
    sinkhorn_reg: Union[float, str] = 'auto',
    sinkhorn_iters: int = 50,
) -> Tuple[VelocityField, List[float]]:
    """
    Train the velocity field using OT-CFM.

    Args:
        velocity_field: VelocityField module to train.
        training_outputs: (n, d) tensor of training data points.
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        coupling: OT coupling method. One of:
            'none' — random pairing (no OT)
            'hungarian' — exact OT via Hungarian algorithm (CPU-only)
            'sinkhorn' — approximate OT via Sinkhorn (GPU-friendly)
        sinkhorn_reg: Entropic regularization strength for Sinkhorn coupling.
            If ``'auto'`` (the default), it is computed adaptively from the
            training data via :func:`compute_adaptive_sinkhorn_reg`, which
            scales ``reg`` with the data's typical cost magnitude so the
            Sinkhorn kernel stays numerically stable regardless of output
            scale. Pass a numeric value to override (e.g. ``0.05`` to
            reproduce the pre-2026-04-14 hardcoded behavior). Smaller values
            are closer to exact OT but numerically less stable. Ignored for
            non-sinkhorn couplings.
        sinkhorn_iters: Number of Sinkhorn iterations when
            ``coupling='sinkhorn'``. Ignored for other coupling methods.
            Defaults to 50 (the previously hardcoded value).

    Returns:
        (velocity_field, losses) — the trained model and per-epoch losses.

    Raises:
        ValueError: If coupling is not one of the valid options.
    """
    valid_couplings = ('none', 'hungarian', 'sinkhorn')
    if coupling not in valid_couplings:
        raise ValueError(
            f"coupling must be one of {valid_couplings}, got '{coupling}'"
        )

    # Resolve adaptive reg if requested
    if coupling == 'sinkhorn' and sinkhorn_reg == 'auto':
        sinkhorn_reg = compute_adaptive_sinkhorn_reg(training_outputs)

    optimizer = torch.optim.Adam(velocity_field.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    dataset = torch.utils.data.TensorDataset(training_outputs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (x1_batch,) in loader:
            # Sample source noise
            x0_batch = torch.randn_like(x1_batch)

            # OT coupling
            if coupling == 'hungarian':
                x0_batch, x1_batch = ot_coupling(x0_batch, x1_batch)
            elif coupling == 'sinkhorn':
                x0_batch, x1_batch = sinkhorn_coupling(
                    x0_batch, x1_batch, reg=sinkhorn_reg, max_iters=sinkhorn_iters
                )

            # Sample time uniformly
            t = torch.rand(x1_batch.shape[0], device=x1_batch.device)

            # Interpolate
            x_t = (1 - t.unsqueeze(1)) * x0_batch + t.unsqueeze(1) * x1_batch

            # Target velocity
            target_v = x1_batch - x0_batch

            # Predicted velocity
            pred_v = velocity_field(t, x_t)

            loss = F.mse_loss(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        losses.append(epoch_loss / max(n_batches, 1))

    return velocity_field, losses
