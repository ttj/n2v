"""Figure 5 — Exp 3 volume-comparison.

Reads the per-(benchmark, config, method) CSVs from
``examples/FlowConformal/experiments/exp3_synthetic/outputs/`` and
plots the volume ratio (predicted_vol / true_reach_vol) per benchmark.
One trace per (method, sample-budget config); y-axis is log-scale
because hashemi/starset blow up exponentially with input dim.

Run with::

    python -m \\
        examples.FlowConformal.paper.figures.fig5_exp3_volume_comparison
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    METHOD_COLORS,
    apply_paper_style,
    save_figure,
)

import matplotlib.pyplot as plt  # noqa: E402

# Benchmark order: by intrinsic dimensionality so the x-axis traces a
# clean d=2 → d=20 progression. The two banana benchmarks are
# nonlinear classifiers; synth_<N>d are 1-Lipschitz identity-activation
# linear nets.
BENCHES = [
    ('2d_banana', 2, 'nonlinear'),
    ('synth_2d',  2, 'linear'),
    ('3d_banana', 3, 'nonlinear'),
    ('synth_3d',  3, 'linear'),
    ('synth_5d',  5, 'linear'),
    ('synth_10d', 10, 'linear'),
    ('synth_20d', 20, 'linear'),
]

# Compact display names for the x-axis tick labels. Short enough that
# they sit flat (no rotation) within the 7-slot horizontal budget.
BENCH_DISPLAY = {
    '2d_banana': 'Banana 2D',
    '3d_banana': 'Banana 3D',
    'synth_2d':  'Synth 2D',
    'synth_3d':  'Synth 3D',
    'synth_5d':  'Synth 5D',
    'synth_10d': 'Synth 10D',
    'synth_20d': 'Synth 20D',
}

CONFIGS = ('small', 'default', 'large')

EXP3_OUTPUTS = (
    Path(__file__).resolve().parents[2]
    / 'experiments' / 'exp3_synthetic' / 'outputs'
)

# Per-method visual conventions. starset is the sound deterministic
# baseline (red palette to match the Exp 1/2 sound-verifier colors);
# hashemi-clipping reuses its blue; ours stays green.
METHOD_STYLES: dict[str, dict] = {
    'ours':             dict(color=METHOD_COLORS['ours'],
                             marker='D', label='Ours'),
    'hashemi_clipping': dict(color=METHOD_COLORS['hashemi_clipping'],
                             marker='s', label='Clipping-Block'),
    'starset':          dict(color='#a50f15',
                             marker='^', label='Star (approx)'),
}

# Config -> linestyle for the line plot, and -> hatch for the bar plot.
CONFIG_LINESTYLE = {
    'small':   ':',
    'default': '--',
    'large':   '-',
}
CONFIG_HATCH = {
    'small':   '....',
    'default': '////',
    'large':   '',
}
CONFIG_LABEL = {
    'small':   'small',
    'default': 'medium',
    'large':   'large',
}


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, '').strip()
        if not v:
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if np.isfinite(x):
            out.append(x)
    return out


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _ratio_summary(bench: str, method: str, config: str) -> tuple[float, float]:
    """Return (mean_ratio, std_ratio) over seeds. NaN if no data."""
    if method == 'ours':
        path = EXP3_OUTPUTS / f'exp3_{bench}_flow_unsat_ours_{config}.csv'
        ratio_col = 'volume_ratio'
    elif method == 'hashemi_clipping':
        path = EXP3_OUTPUTS / f'exp3_{bench}_unsat_hashemi_clipping_{config}.csv'
        ratio_col = 'volume_ratio_closedform'
    elif method == 'starset':
        # Star-set approx has no config axis; reuse the same value
        # across small/default/large bars/markers.
        path = EXP3_OUTPUTS / f'exp3_{bench}_unsat_starset_approx.csv'
        ratio_col = 'volume_ratio'
    else:
        raise ValueError(f'unknown method: {method}')
    rows = _read(path)
    ratios = _floats(rows, ratio_col)
    if not ratios:
        return float('nan'), float('nan')
    if len(ratios) == 1:
        return ratios[0], 0.0
    return statistics.mean(ratios), statistics.stdev(ratios)


def _draw_lines(ax) -> None:
    """Panel A: ratio vs network input dim. One trace per (method, config).

    Two benchmarks share each input dim 2 and 3 (banana vs. synth_*).
    We use one categorical x-slot per benchmark (in BENCHES order) so
    points don't overlap; the dim is annotated in the tick label.
    """
    n_bench = len(BENCHES)
    x_slots = list(range(n_bench))
    slot_labels = [BENCH_DISPLAY[b] for (b, _d, _) in BENCHES]

    for method in ('ours', 'hashemi_clipping', 'starset'):
        style = METHOD_STYLES[method]
        for config in CONFIGS:
            if method == 'starset' and config != 'default':
                continue
            xs, ys = [], []
            for slot, (bench, _dim, _kind) in zip(x_slots, BENCHES):
                m, _s = _ratio_summary(bench, method, config)
                if not np.isfinite(m):
                    continue
                jitter = (
                    {'small': -0.20, 'default': 0.0, 'large': 0.20}[config]
                    + {'ours': -0.05, 'hashemi_clipping': 0.05,
                       'starset': 0.0}[method]
                )
                xs.append(slot + jitter)
                ys.append(m)
            if not xs:
                continue
            label_suffix = (
                '' if method == 'starset' else f' ({CONFIG_LABEL[config]})'
            )
            # No error bars: seed-to-seed stdev frequently exceeds the
            # mean on log scale (the ratio distribution is right-skewed
            # by orders of magnitude on high-d cells), which would
            # render the lower whisker as a tall vertical artifact
            # clipped to the axis bottom. The table reports the
            # numerical (mean, stdev) for each cell.
            ax.plot(
                xs, ys,
                color=style['color'],
                linestyle=CONFIG_LINESTYLE[config],
                marker=style['marker'], markersize=6,
                markeredgecolor='white', markeredgewidth=0.6,
                label=style['label'] + label_suffix,
                linewidth=1.5,
                alpha=0.9 if method != 'starset' else 1.0,
            )

    ax.set_yscale('log')
    ax.set_ylabel('Volume ratio')
    ax.axhline(1.0, color='gray', linestyle='-', linewidth=0.6, alpha=0.5,
               zorder=0)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_xticks(x_slots)
    ax.set_xticklabels(slot_labels, rotation=0, ha='center', fontsize=8)
    ax.legend(loc='upper left', fontsize=8, ncol=2,
              framealpha=0.9, columnspacing=1.0)




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path, default=None,
        help='PNG output path. Default: <this dir>/fig5_exp3_volume_comparison.png',
    )
    args = parser.parse_args()
    output = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parent / 'fig5_exp3_volume_comparison.png'
    )

    apply_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 3.2))
    _draw_lines(ax)
    fig.tight_layout()
    save_figure(fig, output)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
