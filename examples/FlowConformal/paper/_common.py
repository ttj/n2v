"""Shared helpers for paper figure / table generation scripts.

All scripts require an explicit `--csv-dir` pointing at real
experiment outputs. There is no fallback — invoking a figure or table
script without `--csv-dir` raises a hard error to prevent any chance
of rendering paper artifacts from fake / synthetic / stale data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

# Resolve paper/ root regardless of where a script is invoked from.
PAPER_DIR = Path(__file__).resolve().parent

# ---- Verdict / benchmark ordering ----

EXP1_BENCHMARKS = [
    "acasxu_2023",
    "collins_rul_cnn_2022",
    "dist_shift_2023",
    "linearizenn_2024",
    "tllverify_2023",
    "malbeware",
    "metaroom_2023",
]

EXP2_BENCHMARKS = [
    "vit_2023",
    "tinyimagenet_2024",
    "cifar100_2024",
    "cifar10_resnet110",
]

VERDICT_ORDER = ["UNSAT", "SAT", "UNKNOWN", "TIMEOUT", "ERROR", "SKIPPED", "NOT_APPLICABLE"]
SOLVED_VERDICTS = {"UNSAT", "SAT"}

# ---- Sound-verifier ordering (Exp 1 = 4 sound verifiers; Exp 2 = αβ-CROWN only) ----

# Marabou intentionally dropped from this list (per 2026-04 paper revision).
EXP1_SOUND_VERIFIERS = ["alpha_beta_crown", "neuralsat", "pyrat", "cora"]
EXP2_SOUND_VERIFIERS = ["alpha_beta_crown"]

# ---- Method ordering (consistent legend across figures) ----

EXP1_METHODS = [
    "alpha_beta_crown",   # sound (Exp 1)
    "neuralsat",          # sound (Exp 1)
    "pyrat",              # sound (Exp 1)
    "cora",               # sound (Exp 1)
    "hashemi_clipping",
    "rs",
    "saver",
    "probstar",
    "ours",
]

EXP2_METHODS = [
    "alpha_beta_crown",   # sound but TIMEOUT-heavy (only sound verifier in Exp 2)
    "hashemi_clipping",
    "rs",
    "saver",
    "probstar",
    "ours",
]

# ---- Color palette ----
# ours = green; sound verifiers = red/orange; probabilistic baselines = blue tones.
METHOD_COLORS = {
    "ours":              "#1b9e3a",   # green
    "alpha_beta_crown":  "#e6550d",   # orange-red
    "neuralsat":         "#d62728",   # red
    "pyrat":             "#a63603",   # dark red-brown
    "cora":              "#fdae6b",   # light orange
    "hashemi_clipping":  "#1f77b4",   # blue
    "rs":                "#5ab4d3",   # cyan-ish
    "saver":             "#5e3c99",   # purple
    "probstar":          "#3690c0",   # mid-blue
}

METHOD_DISPLAY = {
    "ours":              "Ours (flow-conformal)",
    "alpha_beta_crown":  r"$\alpha,\!\beta$-CROWN",
    "neuralsat":         "NeuralSAT",
    "pyrat":             "PyRAT",
    "cora":              "CORA",
    "hashemi_clipping":  "Hashemi (clip)",
    "rs":                "RS",
    "saver":             "SaVer",
    "probstar":          "ProbStar",
}

BENCHMARK_DISPLAY = {
    "acasxu_2023":          "ACAS Xu",
    "collins_rul_cnn_2022": "Collins-RUL",
    "dist_shift_2023":      "DistShift",
    "linearizenn_2024":     "LinearizeNN",
    "tllverify_2023":       "TLLVerify",
    "malbeware":            "Malbeware",
    "metaroom_2023":        "MetaRoom",
    "vit_2023":             "ViT-2023",
    "tinyimagenet_2024":    "TinyImageNet-2024",
    "cifar100_2024":        "CIFAR100-2024",
    "cifar10_resnet110":    "CIFAR-ResNet110",
}


# ---- CSV loading ----

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV with header into a list of dicts."""
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def read_csv_no_header(path: Path) -> list[list[str]]:
    """Read a CSV without header (e.g. VNN-COMP results.csv)."""
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        return [list(row) for row in reader]


def normalize_verdict(v: str) -> str:
    """Uppercase and stripped verdict (handles VNN-COMP lowercase rows)."""
    return v.strip().upper()


def count_verdicts(rows: Iterable[dict[str, str]], verdict_key: str = "verdict") -> dict[str, int]:
    counts = {v: 0 for v in VERDICT_ORDER}
    for r in rows:
        v = normalize_verdict(r.get(verdict_key, ""))
        if v not in counts:
            counts[v] = 0
        counts[v] += 1
    return counts


def percent_solved(counts: dict[str, int]) -> float:
    total_applicable = sum(c for k, c in counts.items() if k != "NOT_APPLICABLE")
    if total_applicable == 0:
        return 0.0
    solved = sum(counts.get(v, 0) for v in SOLVED_VERDICTS)
    return 100.0 * solved / total_applicable


def mean_wall_clock(rows: Iterable[dict[str, str]], wall_key: str) -> float:
    vals = []
    for r in rows:
        s = r.get(wall_key, "").strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except ValueError:
            continue
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# ---- CLI helpers ----

def add_common_args(parser: argparse.ArgumentParser, default_output: str) -> argparse.ArgumentParser:
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory holding the input CSVs. REQUIRED — no default fallback "
             "exists. Point this at a real experiment outputs directory "
             "(e.g. examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output path (default: {default_output} alongside the script).",
    )
    return parser
