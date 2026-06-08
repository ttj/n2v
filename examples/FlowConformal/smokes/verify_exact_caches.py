"""Verify cached 'exact' volume claims for flow-conformal benchmarks.

Re-runs Star propagation + ``star_union_volume_mc`` on the banana and
three-blob-3D benchmarks and prints the volume with a 99% Hoeffding
confidence interval. Output is compared to any cached value.

Context:
* ``ThreeBlobClassifier3D`` has a cached pkl volume at
  ``_exact_volume_cache/ThreeBlobClassifier3D__center_0.000_0.000_0.000__radius_1.000.pkl``.
* ``RotatedBananaNet`` does not have a comparable cached volume; the
  ratio CSV in ``exp_baseline_comparison_3d.csv`` uses per-method MC
  against a reference in other scripts. We just report the MC value here.

This script is slow (several minutes per benchmark): it runs
``Star.contains`` via LP on every sampled point inside any Star's bbox.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from examples.FlowConformal.networks import RotatedBananaNet, ThreeBlobClassifier3D  # noqa: E402
from examples.FlowConformal.utils import compute_exact_reach  # noqa: E402
from n2v.sets.volume import star_union_volume_mc  # noqa: E402


def verify_three_blob_3d():
    print('=' * 60)
    print('ThreeBlobClassifier3D, center=(0,0,0), radius=1.0')
    print('=' * 60)
    torch.manual_seed(0)
    net = ThreeBlobClassifier3D()
    net.eval()
    reach = compute_exact_reach(net, np.zeros(3), 1.0, output_dim=3)
    stars = reach['stars']
    print(f'  n_stars: {len(stars)}')
    v = star_union_volume_mc(
        stars, n_samples=500_000, batch_size=25_000, seed=42,
        contains_method='algebraic',  # square full-rank basis -> fast path
    )
    print(f'  MC mean       : {v.mean:.3f}')
    print(f'  SE            : {v.se:.3f}')
    print(f'  99% CI        : [{v.ci_low:.3f}, {v.ci_high:.3f}]')
    print(f'  n_lp_calls    : {v.meta["n_lp_calls"]:,}')
    print(f'  cached value  : 213.72 (prior MC pipeline)')
    if v.ci_low <= 213.72 <= v.ci_high:
        print('  VERDICT       : consistent with cached value.')
    else:
        print('  VERDICT       : INCONSISTENT with cached value.')


def verify_banana_2d():
    print('=' * 60)
    print('RotatedBananaNet, center=(0.5,0.5), radius=0.5 (2D)')
    print('=' * 60)
    torch.manual_seed(0)
    net = RotatedBananaNet()
    net.eval()
    # Banana net expects input in [0,1]^2; centre at (0.5, 0.5) with radius
    # 0.5 covers the natural domain exactly.
    reach = compute_exact_reach(
        net, np.array([0.5, 0.5]), 0.5, output_dim=2,
    )
    stars = reach['stars']
    print(f'  n_stars: {len(stars)}')
    v = star_union_volume_mc(
        stars, n_samples=300_000, batch_size=25_000, seed=42,
        contains_method='algebraic',
    )
    print(f'  MC mean       : {v.mean:.4f}')
    print(f'  SE            : {v.se:.4f}')
    print(f'  99% CI        : [{v.ci_low:.4f}, {v.ci_high:.4f}]')
    print(f'  n_lp_calls    : {v.meta["n_lp_calls"]:,}')
    print(f'  (no cached value to compare)')


def main():
    verify_banana_2d()
    print()
    verify_three_blob_3d()


if __name__ == '__main__':
    main()
