"""UNSAT-recall LaTeX table — methods (rows) × benchmarks (columns).

Identical layout to ``main_table.py`` but the per-cell metric is
\\emph{UNSAT-recall} instead of \\% Solved.

Per-cell format::

    \\makecell{<recall>\\%\\\\(<correct UNSAT>\\,/\\,<total GT UNSAT>)}

where:

* ``correct UNSAT`` = # instances where the tool emitted UNSAT AND the
  VNN-COMP'25 ground-truth consensus says UNSAT.
* ``total GT UNSAT`` = total # instances in the benchmark with
  ground-truth-consensus = UNSAT (regardless of whether the tool ran
  that instance — keeps the denominator constant across methods so
  cross-method comparison is fair).
* ``recall`` = ``correct UNSAT / total GT UNSAT``.

Instances without a ground-truth consensus (undecided by all tools)
are excluded from both numerator and denominator (they cannot be
"truly UNSAT" by definition).

Sound violations (tool emitted UNSAT but GT = SAT) are EXCLUDED from
the numerator by construction — only verdicts that match the
GT-UNSAT instance set count as correct UNSAT predictions.

Inapplicable cells (tool can't load the network for ANY instance,
or no data) emit ``$-$``.

Usage::

    python -m examples.FlowConformal.paper.tables.main_table_recall

Reads from the same canonical experiment-output paths as
``main_table.py``; same ``--exp1-csv-dir`` / ``--exp2-csv-dir`` /
``--vnncomp-dir`` / ``--output`` overrides.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Bump CSV field limit (Exp 2 cex_x / cex_y can exceed default 128KB).
csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    write_table,
)

# Reuse the path / metadata constants from main_table.py to keep the
# two tables consistent. If you add a new benchmark or method to
# main_table.py, both tables update together.
from main_table import (  # noqa: E402
    BENCHMARKS,
    BENCHMARK_HEADER,
    DEFAULT_EXP1,
    DEFAULT_EXP2,
    DEFAULT_VNNCOMP,
    METHOD_ROWS,
    REPO,
    SOUND_VERIFIERS,
    _read_internal_csv,
    _read_vnncomp_csv,
    csv_path,
    load_ground_truth,
)


# ----- per-cell metric extraction (UNSAT-recall) -----

def _effective_gt(repo: Path) -> tuple[dict[tuple[str, str, str], str], dict[str, int]]:
    """Build the *effective* ground truth used by UNSAT-recall.

    An instance is treated as effectively UNSAT if either:
      (a) the VNN-COMP'25 consensus is ``unsat``, OR
      (b) the consensus is ``unknown`` AND no tool emitted a SAT
          verdict on it (``n_sat == 0`` per the ground-truth CSV).

    Rationale: case (b) instances are ones where every sound tool
    timed out or abstained, but \\emph{no} sound tool found a
    counterexample. Per VNN-COMP scoring convention, the absence of a
    SAT verdict means the tools failed to falsify, so the most
    informative interpretation is "no counterexample exists" \\,---{}
    i.e., effective UNSAT. This penalizes tools that timed out on
    these instances and credits tools (e.g., ours) that did emit an
    UNSAT verdict on them.

    Returns:
        ``(effective_gt, gt_unsat_total)`` where:
          - ``effective_gt[(bench, onnx, vnn)]`` is ``"unsat"`` or
            ``"sat"`` (instances treated as effectively decided).
          - ``gt_unsat_total[bench]`` is the count of effective-UNSAT
            instances per benchmark (the recall denominator).
    """
    eff_gt: dict[tuple[str, str, str], str] = {}
    counts: dict[str, int] = defaultdict(int)
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
            try:
                n_sat = int(r.get("n_sat", "0") or 0)
            except ValueError:
                n_sat = 0
            if g == "unsat":
                eff_gt[(b, o, v)] = "unsat"
                counts[b] += 1
            elif g == "sat":
                eff_gt[(b, o, v)] = "sat"
            elif n_sat == 0:
                # Consensus undecided AND no tool found a counter-
                # example: treat as effectively UNSAT.
                eff_gt[(b, o, v)] = "unsat"
                counts[b] += 1
            # else: consensus undecided AND some tool found SAT but
            # consensus didn't ratify it → leave out of GT entirely.
    return eff_gt, dict(counts)


def cell_metrics(method: str, bench: str,
                 effective_gt: dict[tuple[str, str, str], str],
                 gt_unsat_total: dict[str, int],
                 exp1_dir: Path, exp2_dir: Path, vnncomp_dir: Path
                 ) -> Optional[dict]:
    """Return dict with keys correct_unsat, total_gt_unsat, n_sound or None.

    Uses the *effective* ground truth from :func:`_effective_gt`,
    where consensus-undecided instances with no SAT verdict are
    treated as effectively UNSAT.

    ``correct_unsat`` = # instances where (verdict == UNSAT AND
                       effective_gt == 'unsat')
    ``total_gt_unsat`` = effective GT-UNSAT count per benchmark
                       (denominator).
    ``n_sound``       = # sound violations (FN + FP) wrt effective GT.
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

    INAPPLICABLE_VERDICTS = {"ERROR", "NOT_APPLICABLE", "SKIPPED"}
    if all(r["verdict"] in INAPPLICABLE_VERDICTS for r in rows):
        return None

    correct_unsat = 0
    n_sound = 0
    for r in rows:
        v = r["verdict"]
        g = effective_gt.get((bench, r["onnx_file"], r["vnnlib_file"]), "")
        if v == "UNSAT":
            if g == "unsat":
                correct_unsat += 1
            elif g == "sat":
                n_sound += 1
        elif v == "SAT":
            if g == "unsat":
                n_sound += 1

    total_gt_unsat = gt_unsat_total.get(bench, 0)
    return {
        "correct_unsat": correct_unsat,
        "total_gt_unsat": total_gt_unsat,
        "n_sound": n_sound,
    }


# ----- LaTeX cell + table assembly -----

NA_CELL = r"\multicolumn{1}{c}{$-$}"

# (method, benchmark) cells whose numeric content should be rendered
# in \textbf{} (best per column / highlight). User-curated; update to
# match the visual emphasis they want in the paper.
BOLD_CELLS: set[tuple[str, str]] = {
    # Ours: every column except linearizenn.
    ("ours", "acasxu_2023"),
    ("ours", "collins_rul_cnn_2022"),
    ("ours", "dist_shift_2023"),
    ("ours", "tllverify_2023"),
    ("ours", "metaroom_2023"),
    ("ours", "vit_2023"),
    ("ours", "tinyimagenet_2024"),
    ("ours", "cifar100_2024"),
    # αβ-CROWN: every column.
    ("alpha_beta_crown", "acasxu_2023"),
    ("alpha_beta_crown", "collins_rul_cnn_2022"),
    ("alpha_beta_crown", "dist_shift_2023"),
    ("alpha_beta_crown", "linearizenn_2024"),
    ("alpha_beta_crown", "tllverify_2023"),
    ("alpha_beta_crown", "metaroom_2023"),
    ("alpha_beta_crown", "vit_2023"),
    ("alpha_beta_crown", "tinyimagenet_2024"),
    ("alpha_beta_crown", "cifar100_2024"),
    # CLIP (Hashemi): linearizenn, tllverify, vit, tinyimagenet, cifar100.
    ("hashemi_clipping", "linearizenn_2024"),
    ("hashemi_clipping", "tllverify_2023"),
    ("hashemi_clipping", "vit_2023"),
    ("hashemi_clipping", "tinyimagenet_2024"),
    ("hashemi_clipping", "cifar100_2024"),
}


def fmt_cell(metrics: Optional[dict], bold: bool = False) -> str:
    if metrics is None:
        return NA_CELL
    correct = metrics["correct_unsat"]
    total = metrics["total_gt_unsat"]
    sound = metrics.get("n_sound", 0)
    if total == 0:
        return NA_CELL
    pct = 100.0 * correct / total
    # Bold the percentage only (per user direction). Daggers and the
    # (V/total) breakdown stay un-bolded so the highlight reads cleanly.
    pct_num = f"{pct:.1f}\\%"
    if bold:
        pct_num = r"\textbf{" + pct_num + "}"
    pct_str = pct_num + (r"\textsuperscript{$\dagger$" + str(sound) + "}"
                         if sound > 0 else "")
    pair_str = "(" + str(correct) + r"\,/\," + str(total) + ")"
    return r"\makecell{" + pct_str + r"\\" + pair_str + "}"


def build_table(exp1_dir: Path, exp2_dir: Path, vnncomp_dir: Path,
                effective_gt: dict[tuple[str, str, str], str],
                gt_unsat_total: dict[str, int]) -> str:
    L: list[str] = []
    L.append(r"% Auto-generated by examples/FlowConformal/paper/tables/main_table_recall.py")
    L.append(r"% Required LaTeX packages: booktabs, makecell.")
    L.append(r"\begin{table*}[!t]")
    L.append(r"\centering")
    L.append(
        r"\caption{UNSAT-recall on VNN-COMP'25 benchmarks. Each cell "
        r"reports \emph{UNSAT-recall}\,$\uparrow$ over the breakdown "
        r"$(\#\,\mathrm{correct}\,\mathrm{UNSAT}\,/\,\#\,\mathrm{effective}\,\mathrm{UNSAT})$. "
        r"An instance counts as \emph{effective UNSAT} if the VNN-COMP'25 "
        r"ground-truth consensus is UNSAT or if every sound tool timed out / "
        r"abstained \emph{and} no tool emitted a SAT counterexample (i.e., "
        r"no falsifier found one\,---\,treated as effective UNSAT here). "
        r"This denominator is the same across methods, so tools that timed "
        r"out on the harder instances are correctly penalized rather than "
        r"silently excluded. The numerator counts instances where the tool "
        r"emitted UNSAT and the effective ground truth is UNSAT (sound "
        r"violations are excluded from the numerator by construction). A "
        r"dagger ($\dagger N$) annotates cells where the tool committed $N$ "
        r"sound violations\,---\,$N$ matches the dagger count in the "
        r"companion \% Solved table. An em-dash ($-$) indicates the "
        r"verifier's loader does not support the network, the benchmark was "
        r"not run, or the benchmark has no effective-UNSAT instances. "
        r"\textbf{Bold} percentages indicate per-column highlights. "
        r"``Ours'' is highlighted at the top; the middle block lists "
        r"deterministic sound verifiers; the bottom block lists probabilistic "
        r"verifiers.}"
    )
    L.append(r"\label{tab:main_results_recall}")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\renewcommand{\arraystretch}{1.05}")
    L.append(r"\footnotesize")
    L.append(r"\begin{tabular}{l ccccccccc}")
    L.append(r"\toprule")

    header_cells = [r"\textbf{Method}"]
    for b in BENCHMARKS:
        top, bot = BENCHMARK_HEADER[b]
        header_cells.append(r"\makecell{" + top + r"\\" + bot + "}")
    L.append(" & ".join(header_cells) + r" \\")
    L.append(r"\midrule")

    n_cols = 1 + len(BENCHMARKS)

    last_group = None
    for method_id, display_name, group in METHOD_ROWS:
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
                method_id, bench, effective_gt, gt_unsat_total,
                exp1_dir, exp2_dir, vnncomp_dir,
            )
            cells.append(fmt_cell(metrics, bold=(method_id, bench) in BOLD_CELLS))
        L.append(" & ".join(cells) + r" \\")
        last_group = group

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


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
        help="Root of the local VNN-COMP'25 results dump.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "main_table_recall.tex",
        help="Output .tex path.",
    )
    args = parser.parse_args()

    effective_gt, gt_unsat_total = _effective_gt(REPO)
    table_tex = build_table(
        args.exp1_csv_dir, args.exp2_csv_dir, args.vnncomp_dir,
        effective_gt, gt_unsat_total,
    )
    write_table(args.output, table_tex)
    print(f"Wrote {args.output}")
    # Also print the GT-UNSAT denominators so the user can sanity-check.
    print()
    print("Per-benchmark GT-UNSAT denominators (universe size):")
    for b in BENCHMARKS:
        print(f"  {b:30s} {gt_unsat_total.get(b, 0)}")


if __name__ == "__main__":
    main()
