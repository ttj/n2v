# FlowConformal — Paper Experiments

Authoritative description of the four paper experiments + the
verification-method ablation, including setup, headline results, and
execution order. Each section includes a paper-ready summary suitable
for the writeup.

CSV column-level schema for every output is at
[`../CSV_SCHEMAS.md`](../CSV_SCHEMAS.md). The single canonical sweep
launcher is
[`run_paper_sweeps.sh`](run_paper_sweeps.sh)
(use `--phase exp1|exp2|exp3|exp4|ablation|all`).

---

## Common conventions

### Seeding (cross-experiment, cross-tool, order-independent)

**Single global seed: `SEED = 47`.** Every per-(benchmark, tool)
runner resets the RNG at the start of each instance's pipeline:

```python
SEED = 47
for onnx_rel, vnn_rel in instances:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    row = run_flow_pipeline(network, lb, ub, spec, seed=SEED, ...)
```

Guarantees: (1) bit-identical CSVs on rerun, (2) order-independence
(reordering instances doesn't change rows), (3) tool-independence at
the instance setup level (ours / Hashemi / αβ-CROWN see the same
input box, calibration data, and spec for any given instance), and
(4) one number to remember.

For synthetic benchmarks (Exp 3, Exp 4), the *instance* itself is
deterministically generated via a separate per-instance hash seed
(`hash((depth, instance_idx)) & 0x7FFFFFFF` for Exp 4, the random
weight init for synth_<N>d networks for Exp 3). Once the instance is
produced, the verification pipeline uses `SEED=47` exactly as for
VNN-COMP benchmarks.

### Output paths

Per-experiment outputs go under
`examples/FlowConformal/experiments/<exp_dir>/outputs/<filename>.csv`
with a method tag in the filename (e.g.
`exp1_acasxu_2023_ours.csv`). The paper-table / paper-figure
generators under `examples/FlowConformal/paper/` read these CSVs
directly — no intermediate aggregate step.

### Verdicts

`verdict ∈ {UNSAT, SAT, UNKNOWN, TIMEOUT, ERROR, SKIPPED, NOT_APPLICABLE}`.

### Ground truth

For Exp 1 / Exp 2: the SAT-wins consensus across the eight VNN-COMP
2025 sound verifiers, pre-computed by `build_ground_truth.py` and
committed at `exp{1,2}_*/ground_truth.csv`.

For Exp 3: closed-form `|det(W_total)| · prod(ub-lb)` for
identity-activation 1-Lipschitz nets; cached MC reach-set volume
(`exact_volumes.py`) for the bananas.

For Exp 4: UNSAT-by-construction (the spec threshold is set to
`empirical_max(y) + 0.1` where the max is taken over a 100K-sample MC
of the input box; for a 1-Lipschitz net this is reachable only with
vanishing probability).

---

## Experiment 1 — VNN-COMP sound-verifier comparison

| | |
|---|---|
| **Goal** | Probabilistic flow-conformal vs the eight published VNN-COMP'25 sound verifiers on six standard benchmarks. Demonstrate parity with sound verifiers on benchmarks they handle well, and competitive recall where they time out. |
| **Benchmarks (6)** | `acasxu_2023`, `collins_rul_cnn_2022`, `dist_shift_2023`, `linearizenn_2024`, `tllverify_2023`, `metaroom_2023` |
| **Methods** | ours (bounded AMLS), Hashemi-clipping (m=8000), Hashemi-clipping-PCA (metaroom only, m=8000, K=10 components), SaVer, ProbStar. Sound-verifier rows (αβ-CROWN, NeuralSAT, PyRAT, CORA) read from VNN-COMP 2025 `results.csv` — no compute by us. |
| **Per-benchmark hparams** | Locked in `exp1_vnncomp_subset/_benchmarks.py:PER_BENCHMARK_CONFIG`. Most use `mega` (n_train=10K, flow_epochs=2K, scenario_n=2K); ACAS Xu uses `base` (5K/2K/2K), collins_rul_cnn uses `small` (1K/1K/500). Falsifier ON for ours and Hashemi with identical APGD budgets. |
| **N instances per benchmark** | full per-benchmark VNN-COMP `instances.csv` (range: 33–186) |
| **Per-row timeout** | from each `instances.csv` column 3 (VNN-COMP 2025 budget) |
| **Outputs** | `exp1_<benchmark>_<method>.csv` |
| **Wall (full sweep)** | ~2-3 hours |

### Description

Six benchmarks from VNN-COMP 2025 span the standard verifier-comparison
roster: ACAS Xu policy networks (5-input MLPs with K-disjunct
classification specs); RUL-prediction regression
(`collins_rul_cnn_2022`); input-distribution-shift specs
(`dist_shift_2023`); small linearised networks
(`linearizenn_2024`); deep ReLU MLPs from the TLL-verify generator
(`tllverify_2023`); and the metaroom indoor-scene CNN
(`metaroom_2023`, 19-class disjunctive output spec).

Each benchmark uses the VNN-COMP 2025 published `instances.csv` to
fix the per-instance input box, the unsafe halfspace, and the
per-instance shell timeout. We run our method and three probabilistic
baselines (Hashemi-clipping, SaVer, ProbStar) on every instance under
that timeout. The four sound verifiers from VNN-COMP 2025
(αβ-CROWN, NeuralSAT, PyRAT, CORA) are *not* re-run — we read their
published `results.csv` directly so the comparison is against
exactly the numbers the verifier authors reported.

Ground truth for each instance is the SAT-wins consensus across the
eight VNN-COMP 2025 sound tools; instances where every sound tool
timed out *without* a SAT counter-example are folded into "effective
UNSAT" so a method that abstains on the hardest instances is
correctly penalised in the recall denominator.

Two headline tables are produced:

- `paper/tables/main_table.tex` — `% Solved` per (method, benchmark),
  i.e. fraction of instances correctly decided (Verified UNSAT or
  Falsified SAT) under the SAT-wins ground truth.
- `paper/tables/main_table_recall_compact.tex` — UNSAT-recall (the
  fraction of effective-UNSAT instances correctly certified UNSAT),
  with sound-violation counts annotated as daggers.

This experiment establishes the headline soundness-and-recall claim
of the paper: probabilistic flow-conformal certification is
competitive with sound verification on standard VNN-COMP benchmarks
under matched per-instance budgets.

---

## Experiment 2 — Probabilistic-scale comparison

| | |
|---|---|
| **Goal** | Three image-classification benchmarks (ViT, ResNet-medium) where every published sound verifier hits its 100s budget on a substantial fraction of instances. Show that probabilistic flow-conformal scales when sound verification doesn't. |
| **Benchmarks (3)** | `vit_2023` (76K-param 10-class ViT, 200 instances), `tinyimagenet_2024` (2.5M-param ResNet-medium, 200-class, 200 instances), `cifar100_2024` (2.5M-param ResNet-medium, 100-class, 200 instances) |
| **Methods** | ours (mega + `amls_bounded_union`), Hashemi-clipping-PCA (m=2500, K=32 PCA components), SaVer, ProbStar, RS (cifar100 only). Sound-verifier rows again from VNN-COMP 2025 `results.csv`. |
| **Per-benchmark hparams** | All three use the same locked config: `mega` (n_train=10K, flow_epochs=2K, scenario_n=2K) + `verification_method='amls_bounded_union'` + `amls_max_levels=30`. The cora-style nested-OR specs (9 / 99 / 199 disjuncts respectively) are folded into a single AMLS chain via the union variant. |
| **N instances per benchmark** | 200 (full benchmark) |
| **Per-row timeout** | 100s (matches VNN-COMP 2025) |
| **Outputs** | `exp2_<benchmark>_<method>.csv` |
| **Wall (full sweep)** | ~3-4 hours |

### Description

The three benchmarks (ViT, ResNet-medium-100, ResNet-medium-200) are
the high-output-dim subset of VNN-COMP 2025 and share an
architectural property that stresses sound verification: every
published sound verifier hits its 100s budget on a substantial
fraction of instances. The specs are cora-style nested ORs (9, 99,
and 199 disjuncts respectively) — a K-class robustness predicate
that asks "is the predicted class robust to all `K-1` competing
classes within the L∞ ball".

All three benchmarks share the same locked configuration: `mega`
training (n_train=10K, flow_epochs=2K, scenario_n=2K) with
`verification_method='amls_bounded_union'`. The union variant folds
the K-disjunct OR into a single AMLS chain on
`phi_union(y) = min_j phi_halfspace_j(y)` instead of K parallel
chains, giving a tight K× speedup at no soundness cost.

The Hashemi-clipping baseline at m=8000 fails on these benchmarks —
every UNSAT instance times out at 100s under raw `clipping_block`
because the surrogate solves K LPs per sample. We instead report the
PCA-projected variant (K=32 components, m=2500) which is
wall-matched to ours at ~50 s/instance and is the variant the
paper table reports as the "CLIP" row.

Per-row timeout = 100s (matches VNN-COMP 2025). We run all 200
instances per benchmark. Ground truth and the headline tables are
produced via the same pipeline as Exp 1 — these three rows complete
the 9-benchmark table in `main_table*.tex`.

This experiment isolates the *probabilistic-scale* story: when sound
verification hits its budget on large image-classification networks,
how do probabilistic methods compare against each other and against
the published sound-verifier numbers under matched compute?

---

## Experiment 3 — Synthetic volume comparison

| | |
|---|---|
| **Goal** | Quantify the *tightness* of each method's predicted reach set against an analytically- or MC-known ground-truth volume. Demonstrates the geometry advantage of flow-conformal that doesn't show up in coarse verdict statistics on far-away halfspace specs. |
| **Benchmarks (7)** | `2d_banana` (RotatedBananaNet on `[0, 1]²`), `3d_banana` (ThreeBlobClassifier3D on `[-1, 1]³`), `synth_2d`, `synth_3d`, `synth_5d`, `synth_10d`, `synth_20d` (identity-activation 1-Lipschitz nets) |
| **Methods (3)** | ours (flow score with bounded AMLS), Hashemi-clipping (axis-aligned pbox), and `n2v.nn.reach.reach_pytorch_model(method='approx')` as the sound deterministic baseline (Star-set bbox) |
| **Sample-budget configs (3)** | small (m=1K), default (m=8K), large (m=16K) |
| **K seeds** | 5 |
| **Spec** | `unsat`: `y_0 ≥ 1e6` for synth_<N>d (UNSAT by 1-Lipschitz construction) and far halfspaces for the bananas. Falsifier OFF. |
| **Outputs** | `exp3_<bench>_flow_unsat_ours_<config>.csv`, `exp3_<bench>_unsat_hashemi_clipping_<config>.csv`, `exp3_<bench>_unsat_starset_approx.csv` |
| **Wall (full sweep)** | ~3-4 hours (synth_20d at m=16K is the long pole) |

### Description

Verdict statistics on far-away halfspace benchmarks compress two
distinct properties of a predicted reach set into a single bit:
*tightness* (how closely the predicted set wraps the true reach
set) and *disjointness with the unsafe halfspace*. A loose predicted
set can still prove disjointness with a far-away spec by 1D
projection; the verdict alone tells the reader nothing about the
geometric quality of the prediction.

This experiment measures tightness directly. The volume ratio
`vol(R_predicted) / vol(R_true)` quantifies how much each method
over-approximates the true reach set; lower is tighter, 1.00 is
exact, and the spread across methods is the geometric advantage
that hides behind verdict-only comparisons.

**Ground-truth volume.** For `synth_<N>d` (identity-activation
1-Lipschitz nets), the reach set is the parallelotope image with
volume `|det(W_total)| · prod(ub - lb)` — closed-form. For the two
banana benchmarks (`RotatedBananaNet`, `ThreeBlobClassifier3D`) the
reach set is non-convex; we use a cached N=10⁷ MC estimate via
`exact_star_union_volume` (in `exact_volumes.py`).

**Predicted-volume estimators.** Three methods are compared:

- **Ours** (flow score with bounded AMLS): `R_q = {y : ‖φ⁻¹(y)‖ ≤ q}`
  where `φ` is the calibrated flow. Volume estimated by Monte Carlo
  on a bounding box of network outputs.
- **Hashemi-clipping**: axis-aligned pbox volume = `prod(ub - lb)`,
  closed-form. We additionally compute an MC sanity estimate by
  sampling from a 1.1×-padded bbox; the closed form and MC agree
  modulo MC noise.
- **Starset-approx baseline** (`n2v.nn.reach.reach_pytorch_model
  (method='approx')`): the sound deterministic over-approximation —
  one Star polytope per output region, axis-aligned bbox of the
  union. This is the natural sound-baseline comparator.

**Sample-budget axis.** Three configs (`small` m=1K, `default`
m=8K, `large` m=16K) per (benchmark, method) sweep over the
calibration / MC budget. The axis controls how budget translates
into tightness — a useful headline because sample-based methods
trade compute for accuracy. 5 seeds per cell.

**Specification.** All cells use `unsat`-type specs (far-away
halfspaces, UNSAT by construction). Falsifier OFF. The verdict is
not the metric here — the metric is the volume ratio.

The figure is `paper/figures/fig5_exp3_volume_comparison.png` —
volume ratio vs benchmark, log-y, one trace per (method, config).
This experiment is what justifies the geometry-aware nonconformity
score in the paper: it exposes a 2-7-order-of-magnitude gap that's
invisible at the verdict level.

---

## Experiment 4 — Controlled scaling on 1-Lipschitz family

| | |
|---|---|
| **Goal** | Demonstrate sub-exponential scaling of ours vs the exponential blow-up of sound verifiers on a controlled deep-MLP family. Headline plot: accuracy (% correctly solved) and mean wall-clock per instance vs network size. |
| **Network family** | 7 spectrally-normalized ReLU MLPs with random Gaussian weights, **fixed width W=512**, depths `D ∈ {2, 4, 8, 16, 24, 32, 40}`. Param range **3.6K → 10M** (4 orders of magnitude). One network per depth. |
| **Verification problems per network** | 10 per depth. Each instance: pick a random `x_0 ~ Uniform([-1, 1]^5)`; the input box is the L∞ ball `[x_0 ± 0.1]`; the spec is `y_0 ≥ empirical_max(y) + 0.1` — UNSAT by 1-Lipschitz construction. |
| **Methods (4)** | ours (bounded AMLS), αβ-CROWN, NeuralSAT, Hashemi-clipping (m=8000) |
| **Total runs** | 7 × 10 × 4 = **280** |
| **Per-row timeout** | 300 s |
| **Outputs** | `exp4_d<D>_<method>.csv` |
| **Wall (full sweep)** | ~3-4 hours (αβ-CROWN dominates at high depth) |

### Description

VNN-COMP benchmarks vary multiple network properties at once
(depth, width, activation, training regime, spec shape), so
benchmark-to-benchmark differences in any verifier's wall-clock
conflate algorithmic scaling with benchmark idiosyncrasy. This
experiment isolates the *scaling* axis on a controlled synthetic
family.

**Network family.** Seven spectrally-normalised ReLU MLPs share a
fixed width W=512 and identical Gaussian-initialised weight
distributions; only depth varies (`D ∈ {2, 4, 8, 16, 24, 32, 40}`).
The parameter count grows from ~3.6K to ~10M (4 orders of
magnitude) while every other architectural choice is held constant.
Each network is generated programmatically with a deterministic
weight-init seed and exported to ONNX so αβ-CROWN and NeuralSAT can
ingest it via their standard CLI.

**Verification problems per network.** 10 instances per depth.
Each instance picks a random `x_0 ∼ Uniform([-1, 1]^5)` (the input
seeds are deterministic per `instance_idx`), defines an L∞ input box
`[x_0 ± 0.1]`, and constructs the spec
`y_0 ≥ empirical_max(y) + 0.1` where `empirical_max` is the maximum
network output observed over a 100K-sample MC of the input box. The
spec is **UNSAT by 1-Lipschitz construction** — a 1-Lipschitz net
cannot exceed its empirical max by more than 0.1 + L · perturbation,
which sits well below the threshold.

**Methods.** Four are run on every (depth, instance) cell:
- **Ours** (bounded AMLS, mega config) — calibrates a flow on
  100K MC samples of network outputs, then certifies via
  `amls_bounded_certify_spec`.
- **Hashemi-clipping** (m=8000) — axis-aligned pbox surrogate.
- **αβ-CROWN** — invoked via subprocess into its own conda env, with
  a config tuned for deep ReLU MLPs (`abcrown_exp4_deep_mlp.yaml`,
  bs=1024, no PGD, input-split BaB, expandable_segments to mitigate
  OOM).
- **NeuralSAT** — invoked via subprocess into its own conda env.

Per-instance shell timeout is 300s. Total runs: 7 × 10 × 4 = 280.

The figure is `paper/figures/fig4_exp4_scaling.png`: two side-by-side
panels — `% correctly solved` (left) and `mean wall-clock per
instance` (right) — both as a function of network size in
parameters. Cells where any instance hit the shell timeout are
annotated with `N/M TO` on the wall-clock axis.

This experiment is the paper's headline scaling story: with all
non-depth axes held fixed, how does each method's verdict-rate and
wall-clock scale as the network grows?

---

## Verification-method ablation

| | |
|---|---|
| **Goal** | Justify the production choice of `verification_method='amls_bounded'` on Pareto-front grounds. With calibration held fixed per instance (shared `(flow, q)`), compare 5 verifiers on (a) UNSAT-recall on GT-UNSAT instances and (b) sound violations on GT-SAT instances. |
| **Benchmarks (2)** | `acasxu_2023` (186 instances, 140 GT-UNSAT / 47 GT-SAT) and `tllverify_2023` (32 instances, 15 GT-UNSAT / 17 GT-SAT) |
| **Methods (5)** | scenario, AMLS (unbounded), AMLS-bounded, IS-tilted, raw MC uniform |
| **Setup** | `ablation_shared_flow.py` calibrates the flow ONCE per instance and runs all 5 verifiers against the same `(flow, q)`. This isolates verifier quality from flow-randomness contamination. |
| **N instances** | full benchmarks (186 + 32 = 218 instances total) |
| **Outputs** | `ablation_shared_flow_<benchmark>_<method>.csv` |
| **Wall (full sweep)** | ~4 hours (acasxu_2023 dominates) |

### Description

The flow-conformal pipeline has two stages: (a) calibrate a flow on
network outputs and pick a conformal threshold `q`, then (b) verify
that the calibrated reach set `R_q = {y : ‖φ⁻¹(y)‖ ≤ q}` is disjoint
from the unsafe halfspace. Stage (b) — the *verifier* — has multiple
candidate algorithms in the literature, each with different
soundness / recall / wall-clock trade-offs. This experiment is the
controlled comparison that justifies our production choice.

**The methodological pitfall this experiment avoids.** Naively
running each verifier in its own end-to-end pipeline retrains the
flow per (verifier, instance) pair, so cross-verifier differences
conflate verifier quality with flow-randomness noise. Empirically,
flow-randomness noise dominates verifier-quality noise on the
ACAS Xu probe — without controlling for calibration, the ablation is
uninterpretable.

**Shared calibration setup.** The runner
[`ablation_shared_flow.py`](exp_ablation/ablation_shared_flow.py)
calibrates `(flow, q)` ONCE per (instance, input-region box) and
runs all candidate verifiers against the *same* tuple. Verifier
quality is then the only varying axis. Implementation detail: the
script calls `NeuralNetwork.reach(method='flow_matching')` once
per box to obtain the calibrated `ProbabilisticSet`, then loops
over methods calling
`verify_specification(prob_set, spec, config=ProbVerifyConfig(method=...))`
— the new API natively supports the train-once-verify-many pattern.

**Five candidate verifiers.**

- **Scenario** — truncated-Gaussian Monte Carlo on `‖z‖ ≤ q` with
  Campi-Garatti scenario optimisation (the Phase 5b production
  default).
- **AMLS (unbounded)** — adaptive multi-level splitting with a full
  Gaussian-prior MCMC, asymptotic CI bound. The Phase 5d default.
- **AMLS-bounded** — the same level-splitting MCMC restricted to
  `‖z‖ ≤ q` so the rare-event search domain matches the conformal
  coverage region exactly. The Phase 5e production default and the
  candidate this experiment validates.
- **IS-tilted** — importance sampling with a flow-tilted proposal
  (alternative paradigm to MCMC).
- **Raw MC uniform** — uniform Monte Carlo on `‖z‖ ≤ q` with
  Clopper-Pearson upper bound. Brute-force baseline that demonstrates
  what's lost without rare-event estimation.

**Two benchmarks** are chosen to expose complementary failure modes:

- `acasxu_2023` (full 186 instances, 140 GT-UNSAT / 47 GT-SAT) —
  reach-set boundary close to the conformal-coverage limit;
  exposes scenario / IS's boundary-tail miss failure mode where i.i.d.
  sampling fails to find unsafe witnesses near `‖z‖ = q`.
- `tllverify_2023` (full 32 instances, 15 GT-UNSAT / 17 GT-SAT) —
  reach-set support is genuinely disjoint from unsafe but the
  flow assigns small tail mass outside the calibrated ball;
  exposes unbounded AMLS's spurious-positive failure mode where
  level-splitting MCMC drifts past `‖z‖ ≤ q` and finds unsafe
  witnesses *outside* the conformal coverage region.

The table is `paper/tables/tab5_shared_flow_ablation.tex` — five
method rows × four metric columns (UNSAT-recall and sound violations
per benchmark). Bolding marks the per-column best. The experiment
demonstrates the Pareto-front argument for `amls_bounded`: it's the
only verifier that's best-or-tied on every column.

---

## Execution order

The single canonical launcher
[`run_paper_sweeps.sh`](run_paper_sweeps.sh) accepts a `--phase` flag
that maps 1-to-1 to the experiments above:

```bash
bash run_paper_sweeps.sh --phase exp1     # ~2-3 hr
bash run_paper_sweeps.sh --phase exp2     # ~3-4 hr
bash run_paper_sweeps.sh --phase exp3     # ~3-4 hr
bash run_paper_sweeps.sh --phase exp4     # ~3-4 hr
bash run_paper_sweeps.sh --phase ablation # ~4 hr
bash run_paper_sweeps.sh --phase all      # all of the above sequentially, ~15-20 hr
```

Each cell is guarded by a `--force`-able no-overwrite check (the
launcher aborts an individual cell if its output CSV already exists,
preserving the data on disk). Use `--smoke` for a 1-instance sanity
pass per cell (~30-60 min total, hard-capped at 10 min per cell).

---

## Per-experiment runner convention

Every runner accepts:

```
expN_run_<method>.py --benchmark <name> [--smoke]
                     [--instance-idx <k>] [--list-instances]
                     [--write-timeout-row] [--output-csv <path>]
```

`run_cell.sh` uses `--list-instances` + `--instance-idx` to drive
per-instance shell timeouts (so a single hung instance never sinks
the rest of the cell). All Exp 1 / Exp 2 / Exp 4 runners support
this; Exp 3 and the ablation runner manage their own instance loops
in-process.

For long-running sweeps:

```bash
nohup bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    --phase exp4 \
    > /tmp/exp4.log 2>&1 &
```
