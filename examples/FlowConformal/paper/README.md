# Paper figures and tables

This directory holds every figure and table that ships with the
flow-conformal probabilistic verifier paper. Each artefact is generated
by a single Python script that reads CSV inputs from a configurable
directory and writes a paper-ready output (`.png` for figures, `.tex`
for tables).

## Methodology change (2026-04 paper revision)

- **Marabou is dropped from all references.** Sound verifiers for
  Exp 1 are now: αβ-CROWN, NeuralSAT, PyRAT, CORA (4 total). For
  Exp 2 only αβ-CROWN is shown (NeuralSAT, PyRAT, NNV, Rover are
  dropped, because they are TIMEOUT-heavy at Exp 2 scale and add no
  signal).
- These canonical lists live in `_common.py` as
  `EXP1_SOUND_VERIFIERS` and `EXP2_SOUND_VERIFIERS`.

## Regenerating a single artefact

Each script accepts the same two flags:

```
--csv-dir <path>   directory containing the input CSVs (REQUIRED — no default)
--output  <path>   output file (default: <script_name>.{png,tex})
```

There is **no fake-data fallback**. Every invocation must point
`--csv-dir` at a real experiment outputs directory; running a script
without `--csv-dir` raises a hard error to make sure paper artifacts
can never be silently rendered from synthetic / stale data.

Examples:

```bash
# Render a figure from real ablation results:
python examples/FlowConformal/paper/figures/fig4a_score_vs_dim_linear.py \
    --csv-dir examples/FlowConformal/experiments/exp_ablation/outputs

# Override the output path:
python examples/FlowConformal/paper/tables/tab1_exp1_verdict_matrix.py \
    --csv-dir examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs \
    --output /tmp/tab1.tex
```

The scripts can also be invoked as modules:

```bash
python -m examples.FlowConformal.paper.figures.fig4a_score_vs_dim_linear \
    --csv-dir examples/FlowConformal/experiments/exp_ablation/outputs
```

## Regenerating everything

Paper scripts do NOT share a single flag interface: most table scripts
read CSVs from multiple experiments (`--exp1-csv-dir`, `--exp2-csv-dir`,
`--vnncomp-dir`), `fig4_exp4_scaling.py` takes a single `--csv-dir`, and
`fig5_exp3_volume_comparison.py` / `tab5_shared_flow_ablation.py` take
no CSV flag at all (they read hard-coded experiment-output paths). Run
each script with `--help` to see its specific flags; the table below
maps each one to its canonical experiment outputs.

A bulk-regen helper used to live here (`regenerate_all.py`) but assumed
a single shared `--csv-dir`, which is wrong for 7 of 8 scripts. It was
removed; invoke scripts individually.

## Real-data paths

When the real experiments land, point each script at the canonical
output directory documented in [`../CSV_SCHEMAS.md`](../CSV_SCHEMAS.md):

| Script                                 | Real `--csv-dir`                                                |
|----------------------------------------|-----------------------------------------------------------------|
| tab1, fig2, tab_exp1_runtime (Exp 1)   | `examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs`|
| tab2, fig3, tab_exp2_runtime (Exp 2)   | `examples/FlowConformal/experiments/exp2_prob_scale/outputs`    |
| fig4(a), fig5(a/b), fig6(a/b/c), tab3, | `examples/FlowConformal/experiments/exp_ablation/outputs`       |
| tab_score_vs_dim, tab_amls_hparam,     |                                                                 |
| tab_conformal_params, tab_flow_training|                                                                 |

`fig1_flow_training_progression.py` is a thin wrapper around the
existing `flow_matching_training/fig_training_progression.py` script.
By default it copies the cached `overlay.png`; pass `--regenerate` to
re-train the snapshot flows from scratch (slow; needs torch).

`fig7_banana_score_geometries.py` is standalone — no CSV input. It
trains a small flow on `RotatedBananaNet` outputs every run (~30 s on
CPU). Pass `--epochs N` to override the default 100.

## LaTeX integration

The `.tex` outputs are self-contained `\begin{table}...\end{table}`
fragments. They depend on a small set of LaTeX packages:

- ``booktabs`` — toprule / midrule / bottomrule (every table)
- ``multirow`` — `\multirow{N}{*}{...}` cells (tab1, tab2,
  tab_conformal_params)
- ``rotating`` — `\rotatebox{60}{...}` column headers (tab_exp1_runtime,
  tab_exp2_runtime)

Either:

- `\input{tables/tab1_exp1_verdict_matrix.tex}` directly into your
  paper, or
- compile a quick standalone preview:

  ```latex
  \documentclass{article}
  \usepackage{booktabs}
  \usepackage{multirow}
  \usepackage{rotating}
  \begin{document}
  \input{tab1_exp1_verdict_matrix.tex}
  \end{document}
  ```

  All tables compile cleanly with `pdflatex` (one harmless "overfull
  hbox" warning on Table 1 — fix later by tightening column widths).

## Saved-data audit

CSV outputs of the runners include the ``amls_levels_used`` column
(number of adaptive AMLS levels actually run; blank when
``verification_method != 'amls*'``). The value originates from
``AMLSResult.levels_used`` and is exposed as
``VerificationResult.amls_levels_used`` by
``verify_specification``'s probabilistic dispatch (after the
post-NeurIPS cleanup refactor) and as a same-named CSV column by the
shared runner helper at
``examples/FlowConformal/experiments/_shared_flow_runner.py``.

## Style conventions

- **Colour scheme** (`_common.METHOD_COLORS`):
  ours = green (`#1b9e3a`), αβ-CROWN = orange-red (`#e6550d`),
  NeuralSAT = red, PyRAT = dark red-brown, CORA = light orange,
  Hashemi = mid-blue, RS = light blue, SaVer = purple,
  ProbStar = teal-blue.
- **Bold rows** in LaTeX tables denote *ours*; **italic rows** denote
  sound (read-only) verifiers.
- **Em-dashes (`---`)** in tables mark tool/benchmark combinations
  that are NOT_APPLICABLE (e.g. RS / SaVer / Hashemi-clipping on
  ACAS Xu, where the spec is not classification-robustness).
- **Indeterminate column** in Tables 1/2 aggregates UNKNOWN, TIMEOUT,
  ERROR, NOT_APPLICABLE, SKIPPED.
- **Log-y** is used only when dynamic range > 50× (figs 2, 3, 5b).
- **Linear-y** for verdict counts, percentages, and figs 5a, 4a.
- **Sans-serif disabled** — figures use Computer-Modern serif via
  `apply_paper_style()`.

## Pick-one decisions for the user

The following pairs are produced as alternatives so the user can pick
one before submission:

| What                | Figure version                       | Table version              |
|---------------------|--------------------------------------|----------------------------|
| Score × dim         | `fig4_score_vs_dim` (log-y), `fig4a_score_vs_dim_linear` | `tab_score_vs_dim` |
| Scaling             | `fig5a_scaling_linear`, `fig5b_scaling_semilog` (replaces `fig5_scaling`) | — |
| Exp 1 runtime       | `fig2_exp1_runtime`                  | `tab_exp1_runtime`         |
| Exp 2 runtime       | `fig3_exp2_runtime`                  | `tab_exp2_runtime`         |
| Ablations (3 panels)| `fig6a/b/c_ablation_*` (split)       | `tab_amls_hparam`, `tab_conformal_params`, `tab_flow_training` |
