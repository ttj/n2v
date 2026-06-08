"""Matplotlib styling + figure-level helpers shared across paper figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow `from .._common import ...` when running scripts directly
PAPER_DIR = Path(__file__).resolve().parent.parent
if str(PAPER_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR.parent.parent))

from FlowConformal.paper._common import (  # noqa: F401  (re-exported)
    BENCHMARK_DISPLAY,
    EXP1_BENCHMARKS,
    EXP1_METHODS,
    EXP1_SOUND_VERIFIERS,
    EXP2_BENCHMARKS,
    EXP2_METHODS,
    EXP2_SOUND_VERIFIERS,
    METHOD_COLORS,
    METHOD_DISPLAY,
    PAPER_DIR,
    SOLVED_VERDICTS,
    VERDICT_ORDER,
    add_common_args,
    count_verdicts,
    mean_wall_clock,
    normalize_verdict,
    percent_solved,
    read_csv_no_header,
    read_csv_rows,
)


def apply_paper_style() -> None:
    """Apply a light paper-friendly Matplotlib style."""
    plt.rcParams.update({
        "font.family":         "serif",
        "font.size":           10,
        "axes.titlesize":      10,
        "axes.labelsize":      10,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.25,
        "grid.linewidth":      0.5,
        "legend.frameon":      False,
        "legend.fontsize":     9,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "lines.linewidth":     1.6,
        "lines.markersize":    4,
        "figure.dpi":          120,
        "savefig.dpi":         200,
        "savefig.bbox":        "tight",
    })


def save_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
