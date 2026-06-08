"""Figure 4 — Exp 4 scaling vs network depth.

Two stacked subplots sharing the depth x-axis:

* top: \\% correctly solved per method per depth (verdict matches
  ``ground_truth`` column in the CSV; UNKNOWN / TIMEOUT / ERROR /
  sound-violation rows count as NOT solved).
* bottom: mean wall-clock per instance (s, log-scale).

Reads ``exp4_d<D>_<method>.csv`` from ``--csv-dir`` (REQUIRED — no
fake-data fallback). Depths where any instance hit the per-instance
shell timeout are annotated with ``N/M TO`` on the wall-clock axis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    METHOD_COLORS,
    METHOD_DISPLAY,
    add_common_args,
    apply_paper_style,
    read_csv_rows,
    save_figure,
)

import matplotlib.pyplot as plt  # noqa: E402

EXP4_DEPTHS = (2, 4, 8, 16, 24, 32, 40)
EXP4_METHODS = ("alpha_beta_crown", "neuralsat", "hashemi_clipping", "ours")

# Local label overrides for this figure's legend. Keeps the change
# scoped to fig4 instead of touching ``METHOD_DISPLAY`` in
# ``_common.py``, which other figures still read.
LEGEND_LABEL_OVERRIDES: dict[str, str] = {
    "hashemi_clipping": "Clipping-Block",
    "ours":             "Ours",
}

# Parameter count for each depth in the synthetic 1-Lipschitz family
# (width=512). Sourced from the ``n_params`` column of any
# ``exp4_d<D>_<method>.csv`` (constant per depth since the architecture
# is fixed). Used as the x-axis label so the figure communicates
# scale-of-network rather than just layer count.
DEPTH_TO_N_PARAMS: dict[int, int] = {
    2:  3_585,
    4:  528_897,
    8:  1_579_521,
    16: 3_680_769,
    24: 5_782_017,
    32: 7_883_265,
    40: 9_984_513,
}


def _fmt_params(n: int) -> str:
    """Compact human-readable parameter count: ``3.6K`` / ``1.6M``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)

# Per-method visual style: distinct marker + linestyle so overlapping
# lines (e.g., Ours and Hashemi both at 100% on the accuracy plot)
# stay distinguishable. Markers are large enough to read through any
# overlap; tiny x-jitter is also applied so coincident points don't
# stack into a single dot.
METHOD_STYLE: dict[str, dict] = {
    "alpha_beta_crown":  dict(marker="o", linestyle="-",  markersize=7),
    "neuralsat":         dict(marker="^", linestyle="--", markersize=7),
    "hashemi_clipping":  dict(marker="s", linestyle=":",  markersize=7),
    "ours":              dict(marker="D", linestyle="-",  markersize=7),
}
# Horizontal jitter applied per method so equal-value points are not
# perfectly stacked. Units are CATEGORICAL slot indices (depths are
# placed at integer positions 0..N-1 below); ±0.3 spans 60% of the
# inter-tick gap, big enough to clearly separate the four methods
# without bleeding into adjacent depths.
METHOD_X_JITTER: dict[str, float] = {
    "alpha_beta_crown":  -0.30,
    "neuralsat":         -0.10,
    "hashemi_clipping":  +0.10,
    "ours":              +0.30,
}


def _depth_stats(rows: list[dict[str, str]]) -> tuple[float, int, int, float]:
    """Return ``(mean_wall_s, n_total, n_timeout, pct_correctly_solved)``.

    A row counts as ``correctly solved`` when its ``verdict`` matches
    its ``ground_truth`` field (case-insensitive). UNKNOWN / TIMEOUT /
    ERROR / NOT_APPLICABLE / sound-violation rows count as NOT solved.
    Rows missing a ground_truth value (rare; shouldn't happen on the
    synthetic family) are excluded from the % solved denominator.
    """
    walls: list[float] = []
    n_timeout = 0
    n_correct = 0
    n_with_gt = 0
    for r in rows:
        v = r.get("verdict", "").strip().upper()
        gt = r.get("ground_truth", "").strip().lower()
        s = r.get("wall_s", "").strip()
        if v == "TIMEOUT":
            n_timeout += 1
            try:
                walls.append(float(r.get("timeout_s", "0") or 0.0))
            except ValueError:
                pass
        elif s:
            try:
                walls.append(float(s))
            except ValueError:
                pass
        if gt in ("sat", "unsat"):
            n_with_gt += 1
            if v == gt.upper():
                n_correct += 1
    mean_wall = float(np.mean(walls)) if walls else 0.0
    pct_solved = (100.0 * n_correct / n_with_gt) if n_with_gt > 0 else 0.0
    return mean_wall, len(rows), n_timeout, pct_solved


def _load(csv_dir: Path, depth: int, method: str) -> tuple[float, int, int, float]:
    rows = read_csv_rows(csv_dir / f"exp4_d{depth}_{method}.csv")
    return _depth_stats(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output="fig4_exp4_scaling.png")
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parent / "fig4_exp4_scaling.png")

    apply_paper_style()
    fig, (ax_pct, ax_wall) = plt.subplots(
        1, 2, figsize=(12, 3.2), sharex=True,
    )

    # Depth → categorical slot index. All ticks evenly spaced so the
    # per-method x-jitter looks consistent across the whole axis,
    # regardless of the underlying depth gaps (2, 4, 8, 8, ...).
    depth_to_slot = {d: i for i, d in enumerate(EXP4_DEPTHS)}

    for method in EXP4_METHODS:
        slots, walls, pcts, timeouts = [], [], [], []
        for depth in EXP4_DEPTHS:
            mean_wall, n_total, n_to, pct = _load(args.csv_dir, depth, method)
            if n_total == 0:
                continue
            slots.append(depth_to_slot[depth])
            walls.append(mean_wall)
            pcts.append(pct)
            timeouts.append((n_to, n_total))

        if not slots:
            continue
        color = METHOD_COLORS.get(method, "#888")
        label = LEGEND_LABEL_OVERRIDES.get(method,
                                           METHOD_DISPLAY.get(method, method))
        style = METHOD_STYLE.get(method, dict(marker="o", linestyle="-",
                                              markersize=7))
        jitter = METHOD_X_JITTER.get(method, 0.0)
        xs_j = [x + jitter for x in slots]

        # Left: % correctly solved
        ax_pct.plot(
            xs_j, pcts,
            color=color, label=label,
            markeredgecolor="white", markeredgewidth=0.8,
            alpha=0.95,
            **style,
        )

        # Right: wall-clock with TO annotations
        ax_wall.plot(
            xs_j, walls,
            color=color, label=label,
            markeredgecolor="white", markeredgewidth=0.8,
            alpha=0.95,
            **style,
        )
        for x, y, (nt, n) in zip(xs_j, walls, timeouts):
            if nt > 0:
                ax_wall.annotate(
                    f"{nt}/{n} TO",
                    xy=(x, y), xytext=(0, 8),
                    textcoords="offset points",
                    color=color, fontsize=7, ha="center",
                )

    # Categorical x-axis: position 0..N-1, labelled with parameter
    # counts (compact format). Same on both subplots.
    slot_positions = list(range(len(EXP4_DEPTHS)))
    slot_labels = [_fmt_params(DEPTH_TO_N_PARAMS[d]) for d in EXP4_DEPTHS]

    ax_pct.set_ylim(-5, 105)
    ax_pct.set_xlabel("Neural network size (# of parameters)")
    ax_pct.set_ylabel("% correctly solved")
    ax_pct.set_title("Accuracy vs. network size")
    ax_pct.set_xticks(slot_positions)
    ax_pct.set_xticklabels(slot_labels)
    ax_pct.grid(True, which="both", alpha=0.3)
    ax_pct.legend(loc="lower left", fontsize=8)

    ax_wall.set_xscale("linear")
    ax_wall.set_xticks(slot_positions)
    ax_wall.set_xticklabels(slot_labels)
    ax_wall.set_xlabel("Neural network size (# of parameters)")
    ax_wall.set_ylabel("Mean wall-clock per instance (s)")
    ax_wall.set_title("Wall-clock vs. network size")
    ax_wall.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
