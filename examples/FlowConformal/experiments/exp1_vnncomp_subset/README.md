# Exp 1: Sound-Verifier Comparison on VNN-COMP 2025

Run our flow-conformal bounded-AMLS pipeline on a curated VNN-COMP 2025
subset, compare verdict counts against the 8-tool sound-verifier
consensus (αβ-CROWN, NeuralSAT, PyRAT, NNV, NNEnum, CORA, ROVER,
sobolbox), and quantify the false-UNSAT rate (FUR) and false-SAT rate
(FSR) of ours vs. Hashemi-clipping at VNN-COMP-budget per-row timeouts.

We do NOT integrate any third-party verifier into n2v. Sound-verifier
verdicts are pre-aggregated into [`ground_truth.csv`](ground_truth.csv)
by [`build_ground_truth.py`](../build_ground_truth.py) using the
SAT-wins rule (any single SAT vote → SAT; otherwise any UNSAT → UNSAT;
otherwise unknown).

## Roster (7 benchmarks)

| short name             | VNN-COMP dir (under `vnncomp2025_benchmarks/benchmarks/`) | per-row budget |
|------------------------|------------------------------------------------------------|----------------|
| `acasxu_2023`          | `acasxu_2023/`                                             | varies (per-row) |
| `collins_rul_cnn_2022` | `collins_rul_cnn_2022/`                                    | varies         |
| `dist_shift_2023`      | `dist_shift_2023/`                                         | varies         |
| `linearizenn_2024`     | `linearizenn_2024/`                                        | varies         |
| `tllverify_2023`       | `tllverifybench_2023/`                                     | varies         |
| `malbeware`            | `malbeware/`                                               | 100 s          |
| `metaroom_2023`        | `metaroom_2023/`                                           | 210 s          |

What's NOT here (per design):

* `vit_2023` — same network appears in Exp 2 with identical hparams; running
  it twice was pure compute duplication.
* `cora_2024`, `safenlp_2024` — calibration-miss-infeasible at their tight
  per-row budgets (30 s and 20 s respectively).
* `cifar100_2024` — appears in Exp 2 (with disjunctive `amls_bounded_union`).

The three image-classification benchmarks (`malbeware`, `metaroom_2023`,
and Exp 2's `cifar100_2024`) all have **multi-class disjunctive specs**
(cora-style nested OR-of-HalfSpaces). They use
`verification_method='amls_bounded_union'` — folds K disjuncts into a
single AMLS chain via `phi_union(y) = min_k phi_k(y)`. The default
`amls_bounded` (one chain per disjunct) TIMEOUTs at any reasonable
budget on these specs.

## Layout

```
exp1_vnncomp_subset/
├── README.md                              this file
├── _common.py                             generic ONNX wrapper + spec extractor
├── _benchmarks.py                         per-benchmark loader dispatch + hparam config
├── exp1_run_ours.py                       single ours runner (--benchmark X)
├── exp1_run_hashemi_clipping.py           single Hashemi runner (--benchmark X)
├── exp1_run_hashemi_clipping_pca.py       PCA-projected Hashemi clipping variant
├── exp1_run_probstar.py                   ProbStar baseline runner
├── exp1_run_saver.py                      SAVER baseline runner
├── ground_truth.csv                       pre-computed VNN-COMP consensus (run-once)
└── outputs/                               per-(benchmark, tool) CSVs land here
```

## Per-benchmark hparam locks

The locked production config per benchmark (committed in
`_benchmarks.py:PER_BENCHMARK_CONFIG`):

| benchmark              | flow_config | n_train | flow_epochs | scenario_n_samples | max_levels |
|------------------------|-------------|---------|-------------|---------------------|------------|
| `acasxu_2023`          | base        | 5 000   | 2 000       | 2 000               | 30         |
| `collins_rul_cnn_2022` | small       | 1 000   | 1 000       | 500                 | 30         |
| `dist_shift_2023`      | mega        | 10 000  | 2 000       | 2 000               | 30         |
| `linearizenn_2024`     | mega        | 10 000  | 2 000       | 2 000               | 30         |
| `tllverify_2023`       | mega        | 10 000  | 2 000       | 2 000               | 30         |

The two new additions (`malbeware`, `metaroom_2023`) start at `mega`
and back off via the lock probe; the locked config gets written into
`PER_BENCHMARK_CONFIG` in [`_benchmarks.py`](_benchmarks.py) once the
probe completes. Both use `verification_method='amls_bounded_union'`
to handle their multi-class disjunctive specs (24 and 19 disjuncts
respectively).

## Smoke

```bash
cd /path/to/n2v
PY=python
for bench in acasxu_2023 collins_rul_cnn_2022 dist_shift_2023 \
             linearizenn_2024 tllverify_2023 malbeware \
             metaroom_2023; do
  for tool in ours hashemi_clipping; do
    $PY -m examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_$tool \
        --benchmark $bench --smoke
  done
done
```

## Full sweep

The canonical entry point is the project-level launcher:

```bash
bash examples/FlowConformal/experiments/run_paper_sweeps.sh --phase exp1
```

It dispatches each `(benchmark, tool)` cell through
[`run_cell.sh`](../run_cell.sh), which wraps every instance in a shell
`timeout` and appends a `TIMEOUT` row via `--write-timeout-row` on
exit 124. A single hung instance never kills the rest of the cell.

## Aggregation

The headline `% Solved` and `UNSAT-recall` numbers are computed by the
paper-side scripts under `examples/FlowConformal/paper/tables/`
(`main_table.py`, `main_table_recall.py`, `main_table_recall_compact.py`),
which read every `outputs/exp1_<bench>_<tool>.csv` directly and join
against [`ground_truth.csv`](ground_truth.csv). No intermediate
aggregate step is needed.
