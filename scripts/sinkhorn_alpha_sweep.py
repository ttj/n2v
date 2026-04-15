"""
Short sweep to pick the adaptive Sinkhorn reg alpha constant.

Runs alpha in {0.03, 0.05, 0.1} on two probe configs and reports final flow
volumes. Used to lock the default in compute_adaptive_sinkhorn_reg before
the v2 Hashemi comparison run.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from n2v.probabilistic.flow import (
    VelocityField, FlowODE, FlowScore, calibrate, ProbabilisticSet, train_flow,
)
from n2v.probabilistic.flow.train import compute_adaptive_sinkhorn_reg
from FlowConformal.networks import RotatedBananaNet, ThreeBlobClassifier


class CenteredFlowScore:
    def __init__(self, fs, c):
        self.fs = fs
        self.c = c

    def __call__(self, y):
        return self.fs(y - self.c)


def run_one(network, x_center, radius, alpha, output_dim, seed=5000):
    # Sample train + calib
    gen = torch.Generator().manual_seed(seed)
    dim_in = x_center.shape[0]
    x_train = x_center + (torch.rand(8000, dim_in, generator=gen) * 2 - 1) * radius
    x_calib = x_center + (torch.rand(
        8000, dim_in, generator=torch.Generator().manual_seed(seed + 1)
    ) * 2 - 1) * radius
    with torch.no_grad():
        y_train = network(x_train)
        y_calib = network(x_calib)
    center = y_train.mean(dim=0)
    y_train_c = y_train - center

    # Compute the adaptive reg with this alpha
    reg = compute_adaptive_sinkhorn_reg(y_train_c, alpha=alpha)

    # Train the flow
    torch.manual_seed(seed)
    vf = VelocityField(dim=output_dim, hidden=64, n_layers=4)
    t0 = time.time()
    train_flow(
        vf, y_train_c,
        n_epochs=100, batch_size=256, lr=1e-3, coupling='sinkhorn',
        sinkhorn_reg=reg, sinkhorn_iters=50,
    )
    train_time = time.time() - t0

    # Calibrate
    fs = CenteredFlowScore(FlowScore(FlowODE(vf), t=1.0), center)
    with torch.no_grad():
        scores = fs(y_calib)
    threshold = calibrate(scores, 7999).item()

    # Volume MC
    pset = ProbabilisticSet(
        score_fn=fs, threshold=threshold,
        m=8000, ell=7999, epsilon=0.001, dim=output_dim,
    )
    ytm = y_train.min(dim=0).values.numpy()
    ytM = y_train.max(dim=0).values.numpy()
    pad = 0.1 if output_dim == 2 else 0.5
    lb = torch.tensor([ytm[k] - pad for k in range(output_dim)], dtype=torch.float32)
    ub = torch.tensor([ytM[k] + pad for k in range(output_dim)], dtype=torch.float32)
    volume, _ = pset.estimate_volume(n_samples=100_000, bounding_box=(lb, ub))

    return {
        'alpha': alpha,
        'reg': reg,
        'threshold': threshold,
        'volume': volume,
        'train_time': train_time,
    }


def main():
    torch.manual_seed(0)
    banana = RotatedBananaNet()
    torch.manual_seed(0)
    classifier = ThreeBlobClassifier()

    probes = [
        ('banana r=0.05', banana,
         torch.tensor([0.2, 0.2], dtype=torch.float32), 0.05, 2),
        ('classifier r=1.0', classifier,
         torch.tensor([-1.0, 1.0], dtype=torch.float32), 1.0, 3),
    ]
    alphas = [0.03, 0.05, 0.1]

    print(f"{'config':25s} {'alpha':>8s} {'reg':>10s} {'thr':>10s} {'volume':>12s} {'time':>8s}")
    print('-' * 80)
    for config_name, network, x_center, radius, output_dim in probes:
        for alpha in alphas:
            r = run_one(network, x_center, radius, alpha, output_dim)
            print(
                f"{config_name:25s} {r['alpha']:>8.3f} {r['reg']:>10.4f} "
                f"{r['threshold']:>10.4f} {r['volume']:>12.5f} {r['train_time']:>7.1f}s",
                flush=True,
            )


if __name__ == '__main__':
    main()
