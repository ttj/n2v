"""Main results LaTeX table — methods (rows) × benchmarks (columns).

Reads result CSVs from the canonical experiment outputs directories
plus the local VNN-COMP'25 results dump, computes per-(method,
benchmark) ``% Solved`` and ``(# Verified / # Falsified)`` numbers,
flags sound-bound violations against ground truth, and emits the
LaTeX table at ``outputs/main_table.tex``.

Per-cell format::

    \\makecell{<%>\\%\\\\(<V>\\,/\\,<F>)}

where ``%``, ``V``, ``F`` are the % solved (V+F over total), # verified
(UNSAT verdicts), and # falsified (SAT verdicts). Sound violations
(verdict disagreeing with VNN-COMP'25 ground-truth consensus) appear
as a ``\\textsuperscript{\\dagger N}`` after the percentage.

Inapplicable cells (verifier loader can't handle the network, or
benchmark not run) emit ``$-$``.

Usage::

    python -m examples.FlowConformal.paper.tables.main_table

By default the script reads from the canonical experiment outputs
directories under ``examples/FlowConformal/experiments/``. Override
via ``--exp1-csv-dir``, ``--exp2-csv-dir``, and ``--vnncomp-dir``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

# Some Exp 2 CSVs encode flattened counterexample tensors (cex_x /
# cex_y) as long single-cell strings that exceed Python's default
# 131072-byte field limit. Bump it so DictReader can parse them.
csv.field_size_limit(sys.maxsize)

# Allow `from _common import ...` when invoked as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    write_table,
)


# ----- benchmark / method ordering -----

# Final paper benchmark set (cut: lsnc_relu, relusplitter, malbeware,
# cifar10_resnet110).
BENCHMARKS = [
    "acasxu_2023",
    "collins_rul_cnn_2022",
    "dist_shift_2023",
    "linearizenn_2024",
    "tllverify_2023",
    "metaroom_2023",
    "vit_2023",
    "tinyimagenet_2024",
    "cifar100_2024",
]

# Display headers (with line breaks for narrow columns; matches the
# Koller/Ladner/Althoff (CORA) Table-1 hyphenation pattern).
BENCHMARK_HEADER = {
    "acasxu_2023":          ("ACAS",       "Xu"),
    "collins_rul_cnn_2022": ("collins",    "rul-cnn"),
    "dist_shift_2023":      ("dist",       "shift"),
    "linearizenn_2024":     ("linear-",    "izenn"),
    "tllverify_2023":       ("tll-",       "verify"),
    "metaroom_2023":        ("meta-",      "room"),
    "vit_2023":             ("vit",        ""),
    "tinyimagenet_2024":    ("tiny",       "imagenet"),
    "cifar100_2024":        ("cifar",      "100"),
}

# Method rows in display order. Each entry is (method_id, display_name,
# group). Group ∈ {"ours", "sound", "prob"} — controls placement of
# the section separators.
METHOD_ROWS: list[tuple[str, str, str]] = [
    ("ours",             r"\textbf{Ours}",                   "ours"),
    ("alpha_beta_crown", r"$\alpha\beta$-CROWN",             "sound"),
    ("neuralsat",        "NeuralSAT",                        "sound"),
    ("pyrat",            "PyRAT",                            "sound"),
    ("cora",             "CORA",                             "sound"),
    ("hashemi_clipping", r"CLIP\,",                          "prob"),
    ("saver",            "SaVer",                            "prob"),
    ("probstar",         "ProbStar",                         "prob"),
    ("rs",               "RS",                               "prob"),
]

# Sound-verifier method IDs are also the directory names under
# ``vnncomp2025_results/``.
SOUND_VERIFIERS = {"alpha_beta_crown", "neuralsat", "pyrat", "cora"}

# Map our benchmark IDs to the VNN-COMP results directory names.
# Mostly ``2025_<benchmark>``, except tllverify (where VNN-COMP uses
# the longer ``tllverifybench_2023`` name).
VNNCOMP_BENCH_DIR = {b: f"2025_{b}" for b in BENCHMARKS}
VNNCOMP_BENCH_DIR["tllverify_2023"] = "2025_tllverifybench_2023"


# ----- CSV path resolution -----

REPO = Path(__file__).resolve().parents[4]
DEFAULT_EXP1 = REPO / "examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs"
DEFAULT_EXP2 = REPO / "examples/FlowConformal/experiments/exp2_prob_scale/outputs"
DEFAULT_VNNCOMP = Path.home() / "v" / "other" / "VNNCOMP" / "vnncomp2025_results"
EXP2_BENCHMARKS_SET = {"vit_2023", "tinyimagenet_2024", "cifar100_2024"}


def csv_path(method: str, bench: str, exp1_dir: Path, exp2_dir: Path,
             vnncomp_dir: Path) -> Optional[Path]:
    """Return the CSV path for (method, bench), or None when not applicable."""
    if method in SOUND_VERIFIERS:
        sub = VNNCOMP_BENCH_DIR.get(bench)
        if sub is None:
            return None
        return vnncomp_dir / method / sub / "results.csv"

    is_exp2 = bench in EXP2_BENCHMARKS_SET

    if method == "ours":
        if is_exp2:
            return exp2_dir / f"exp2_{bench}_ours.csv"
        return exp1_dir / f"exp1_{bench}_ours.csv"

    if method == "hashemi_clipping":
        # metaroom uses Hashemi-PCA per Phase 6; Exp 2 also uses PCA
        # (Phase B); other Exp 1 cells use basic Hashemi-clipping.
        if bench == "metaroom_2023":
            return exp1_dir / f"exp1_{bench}_hashemi_clipping_pca.csv"
        if is_exp2:
            return exp2_dir / f"exp2_{bench}_hashemi_clipping_pca.csv"
        return exp1_dir / f"exp1_{bench}_hashemi_clipping.csv"

    if method == "saver":
        if is_exp2:
            return exp2_dir / f"exp2_{bench}_saver.csv"
        return exp1_dir / f"exp1_{bench}_saver.csv"

    if method == "probstar":
        if is_exp2:
            return exp2_dir / f"exp2_{bench}_probstar.csv"
        return exp1_dir / f"exp1_{bench}_probstar.csv"

    if method == "rs":
        # Cohen-style randomized smoothing: classifier-only, Exp 2 only.
        if is_exp2:
            return exp2_dir / f"exp2_{bench}_rs.csv"
        return None

    raise ValueError(f"unknown method id: {method}")


# ----- ground truth -----

def load_ground_truth(repo: Path) -> dict[tuple[str, str, str], str]:
    """Load (benchmark, onnx_basename, vnn_basename) -> 'sat'|'unsat'.

    Keying by (benchmark, onnx, vnn) is REQUIRED for ACAS Xu, where 45
    different ONNX networks share each prop_<n>.vnnlib file. Keying by
    (benchmark, vnn) alone produces silently wrong sound-violation
    counts on acasxu (e.g., 5 'false UNSATs' that are actually correct
    UNSATs against the right network).
    """
    out: dict[tuple[str, str, str], str] = {}
    for path in [
        repo / "examples/FlowConformal/experiments/exp1_vnncomp_subset/ground_truth.csv",
        repo / "examples/FlowConformal/experiments/exp2_prob_scale/ground_truth.csv",
    ]:
        if not path.exists():
            continue
        for r in csv.DictReader(open(path)):
            b = r.get("benchmark", "")
            o = os.path.basename(r.get("onnx_file", ""))
            v = os.path.basename(r.get("vnnlib_file", ""))
            g = (r.get("ground_truth") or "").strip().lower()
            if g in ("sat", "unsat"):
                out[(b, o, v)] = g
    return out


# ----- per-cell metric extraction -----

def _normalize_verdict(v: str) -> str:
    return (v or "").strip().upper()


def _read_internal_csv(path: Path) -> list[dict[str, str]]:
    """Read one of our experiment CSVs (with header) and normalize to
    a list of {onnx_file, vnnlib_file, verdict} dicts."""
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            verdict = _normalize_verdict(r.get("verdict", ""))
            onnx = os.path.basename(r.get("onnx_file", ""))
            vnn = os.path.basename(r.get("vnnlib_file", ""))
            # SaVer (Exp 1 + Exp 2) and RS (Exp 2) emit a single
            # 'instance' column with format "<onnx>+<vnn>" or
            # "<onnx>|<vnn>". Parse both separators so the ground-truth
            # cross-reference picks up sound violations instead of
            # silently dropping to 0 (which is what happened pre-fix —
            # SaVer's 39 sound-bound violations on acasxu were invisible
            # because we only checked the '|' separator).
            if not onnx and not vnn:
                inst = r.get("instance", "")
                sep = "+" if "+" in inst else ("|" if "|" in inst else None)
                if sep is not None:
                    parts = inst.split(sep, 1)
                    onnx = os.path.basename(parts[0])
                    vnn = os.path.basename(parts[1]) if len(parts) > 1 else ""
                else:
                    vnn = os.path.basename(inst)
            rows.append({
                "onnx_file": onnx,
                "vnnlib_file": vnn,
                "verdict": verdict,
            })
    return rows


def _read_vnncomp_csv(path: Path) -> list[dict[str, str]]:
    """Read a VNN-COMP'25 results.csv (no header). Columns are::

        benchmark, onnx_path, vnnlib_path, prep_time, verdict, verify_time

    Returns list of {onnx_file, vnnlib_file, verdict} dicts (basenames).
    """
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        for r in csv.reader(f):
            if len(r) < 5:
                continue
            verdict = _normalize_verdict(r[4])
            onnx = os.path.basename(r[1])
            vnn = os.path.basename(r[2])
            rows.append({
                "onnx_file": onnx,
                "vnnlib_file": vnn,
                "verdict": verdict,
            })
    return rows


def cell_metrics(method: str, bench: str, gt: dict[tuple[str, str, str], str],
                 exp1_dir: Path, exp2_dir: Path, vnncomp_dir: Path
                 ) -> Optional[dict]:
    """Return dict with keys total, n_v, n_f, n_sound or None if N/A.

    ``n_sound`` = # rows whose verdict disagrees with the ground-truth
    consensus (FN: said UNSAT but GT=sat; FP: said SAT but GT=unsat).
    Rows without a known ground truth (e.g., undecided-by-all-tools)
    do not contribute to n_sound.
    """
    path = csv_path(method, bench, exp1_dir, exp2_dir, vnncomp_dir)
    if path is None:
        return None
    if method in SOUND_VERIFIERS:
        rows = _read_vnncomp_csv(path)
    else:
        rows = _read_internal_csv(path)
    if not rows:
        return None

    # If every verdict is ERROR / NOT_APPLICABLE / SKIPPED (i.e., the
    # tool couldn't even attempt verification on any instance), treat
    # the cell as "not applicable" and render em-dash. This is
    # different from all-UNKNOWN (which is a real abstention result
    # and should report 0%). Distinguishes ProbStar's all-ERROR rows
    # (StarV loader rejects the network) from SaVer's all-UNKNOWN
    # (SaVer ran every instance, abstained on all of them).
    INAPPLICABLE_VERDICTS = {"ERROR", "NOT_APPLICABLE", "SKIPPED"}
    if all(r["verdict"] in INAPPLICABLE_VERDICTS for r in rows):
        return None

    total = len(rows)
    n_v = n_f = n_sound = 0
    for r in rows:
        v = r["verdict"]
        g = gt.get((bench, r["onnx_file"], r["vnnlib_file"]), "")
        if v == "UNSAT":
            n_v += 1
            if g == "sat":
                n_sound += 1
        elif v == "SAT":
            n_f += 1
            if g == "unsat":
                n_sound += 1
    return {"total": total, "n_v": n_v, "n_f": n_f, "n_sound": n_sound}


# ----- LaTeX cell + table assembly -----

NA_CELL = r"\multicolumn{1}{c}{$-$}"


def fmt_cell(metrics: Optional[dict]) -> str:
    if metrics is None or metrics["total"] == 0:
        return NA_CELL
    total = metrics["total"]
    v = metrics["n_v"]
    f = metrics["n_f"]
    sound = metrics["n_sound"]
    # ``% Solved`` = fraction of instances CORRECTLY decided. Sound
    # violations (verdicts disagreeing with VNN-COMP'25 ground-truth
    # consensus) are subtracted from the numerator — a false UNSAT or
    # false SAT is not a "solved" instance, it is an error.
    pct = 100.0 * (v + f - sound) / total
    pct_str = f"{pct:.1f}\\%"
    if sound > 0:
        pct_str = pct_str + r"\textsuperscript{$\dagger$" + str(sound) + "}"
    return r"\makecell{" + pct_str + r"\\(" + str(v) + r"\,/\," + str(f) + ")}"


def build_table(exp1_dir: Path, exp2_dir: Path, vnncomp_dir: Path,
                gt: dict[tuple[str, str, str], str]) -> str:
    """Return the full LaTeX table source (no document wrapper)."""
    L: list[str] = []
    L.append(r"% Auto-generated by examples/FlowConformal/paper/tables/main_table.py")
    L.append(r"% Required LaTeX packages: booktabs, makecell.")
    L.append(r"\begin{table*}[!t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Verification results on VNN-COMP'25 benchmarks. Each cell "
        r"reports \emph{\% Solved}\,$\uparrow$ over the breakdown "
        r"$(\#\,\mathrm{Verified}\,/\,\#\,\mathrm{Falsified})$. \% Solved is "
        r"the fraction of instances \emph{correctly} decided "
        r"(Verified UNSAT or Falsified SAT, in agreement with the "
        r"VNN-COMP'25 ground-truth consensus); the remainder are abstentions "
        r"or sound errors. A dagger ($\dagger N$) marks cells with $N$ "
        r"verdicts that disagree with the consensus (sound violation); these "
        r"$N$ verdicts are excluded from the \% Solved numerator. An em-dash "
        r"($-$) indicates the verifier's loader does not support the network "
        r"or the benchmark was not run. ``Ours'' is highlighted at the top; "
        r"the middle block lists deterministic sound verifiers; the bottom "
        r"block lists probabilistic verifiers.}"
    )
    L.append(r"\label{tab:main_results}")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\renewcommand{\arraystretch}{1.05}")
    L.append(r"\footnotesize")
    L.append(r"\begin{tabular}{l ccccccccc}")
    L.append(r"\toprule")

    # Header row: method | benchmarks (each as makecell with two lines).
    header_cells = [r"\textbf{Method}"]
    for b in BENCHMARKS:
        top, bot = BENCHMARK_HEADER[b]
        header_cells.append(r"\makecell{" + top + r"\\" + bot + "}")
    L.append(" & ".join(header_cells) + r" \\")
    L.append(r"\midrule")

    n_cols = 1 + len(BENCHMARKS)

    last_group = None
    for method_id, display_name, group in METHOD_ROWS:
        # Section separator between groups (Ours / Sound / Prob).
        if last_group is not None and group != last_group:
            L.append(r"\midrule[\heavyrulewidth]")
            if group == "sound":
                L.append(r"\multicolumn{" + str(n_cols)
                         + r"}{l}{\textit{Sound verifiers (from VNN-COMP'25)}} \\")
                L.append(r"\cmidrule(lr){1-" + str(n_cols) + "}")
            elif group == "prob":
                L.append(r"\multicolumn{" + str(n_cols)
                         + r"}{l}{\textit{Probabilistic verifiers}} \\")
                L.append(r"\cmidrule(lr){1-" + str(n_cols) + "}")

        cells = [display_name]
        for bench in BENCHMARKS:
            metrics = cell_metrics(
                method_id, bench, gt, exp1_dir, exp2_dir, vnncomp_dir
            )
            cells.append(fmt_cell(metrics))
        L.append(" & ".join(cells) + r" \\")
        last_group = group

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


# ----- CLI -----

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp1-csv-dir", type=Path, default=DEFAULT_EXP1,
        help="Directory holding Exp 1 result CSVs.",
    )
    parser.add_argument(
        "--exp2-csv-dir", type=Path, default=DEFAULT_EXP2,
        help="Directory holding Exp 2 result CSVs.",
    )
    parser.add_argument(
        "--vnncomp-dir", type=Path, default=DEFAULT_VNNCOMP,
        help="Root of the local VNN-COMP'25 results dump (per-tool subdirs).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "main_table.tex",
        help="Output .tex path.",
    )
    args = parser.parse_args()

    gt = load_ground_truth(REPO)
    table_tex = build_table(
        args.exp1_csv_dir, args.exp2_csv_dir, args.vnncomp_dir, gt
    )
    write_table(args.output, table_tex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
