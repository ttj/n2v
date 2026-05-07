# FlowConformal — Paper Experiments

Authoritative description of the four paper experiments + ablation
suite + soundness audit, together with their setup, expected outputs,
and execution order. **Read this first before running any experiment.**
The general framework, probabilistic claims, and Phase 5e bounded-AMLS
design are documented in
[`.claude/research/flow-matching-probabilistic-reach/`](../../../.claude/research/flow-matching-probabilistic-reach/)
and [`docs/research/2026-04-28-bounded-amls-design.md`](../../../docs/research/2026-04-28-bounded-amls-design.md).

CSV column-level schema for every output CSV is at
[`CSV_SCHEMAS.md`](../CSV_SCHEMAS.md). Schemas in this
README are summary snapshots; the canonical column list lives there
and is what the figure / table generators consume.

---

## Common conventions

### Seeding (cross-experiment, cross-tool, order-independent reproducibility)

**Single global seed: `SEED = 47`.** Every per-(benchmark, tool) script
resets the RNG at the start of *each instance's pipeline*:

```python
SEED = 47
for onnx_rel, vnn_rel in instances:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    row = run_pipeline(network, lb, ub, spec, seed=SEED, ...)
```

This gives us four guarantees:

1. **Reproducibility** — same script + `SEED=47` → bit-identical CSV.
2. **Order-independence** — the RNG is reset before each instance's
   pipeline, so reordering instances within the sweep doesn't change
   any row.
3. **Tool-independence for the instance setup** — ours,
   Hashemi-clipping, αβ-CROWN all start each instance from the same
   RNG state, so they see identical input boxes, calibration data,
   and spec.
4. **Maximum simplicity** — one number to remember. "I used `SEED=47`
   for everything."

**For VNN-COMP benchmarks** (Exp 1, Exp 2): the instance is fully
identified by `(onnx_rel, vnn_rel)` from each benchmark's
`instances.csv`; pipeline RNG is reset to `SEED=47` per instance.

**For synthetic benchmarks** (Exp 3, Exp 4): the instances themselves
have to be deterministically *different* across `instance_idx`, so
their *generation* uses a separate RNG seeded as
`np.random.RandomState(hash((depth, instance_idx)) & 0x7FFFFFFF)` —
this produces a unique `x_0` for each Exp 4 instance, and a unique
geometric transform for each Exp 3 instance. Once the instance is
generated, the verification pipeline for it uses `SEED=47` exactly
as for VNN-COMP.

**Historical Phase 5e ACAS Xu sweep — superseded.** The earlier
ACAS Xu CSV (run with `seed = hash((onnx_rel, vnn_rel)) & 0x7FFFFFFF`,
the older per-instance scheme) has been archived to
`.claude/research/flow-matching-probabilistic-reach/_archive/acasxu_phase5_sweeps/`.
Under the new SEED=47 convention, ACAS Xu is now produced as a
first-class Phase 1 cell (`exp1_acasxu_2023_ours.csv`) — wall ~3 hr,
single source of truth shared with the other six Exp 1 benchmarks.

We pay one cost: for ours, the flow is retrained per script invocation
(since each tool has its own script). That overhead is modest (~30s
of ~80s total per instance) and the architectural simplification is
worth it.

### Output paths

All per-experiment outputs go under
`examples/FlowConformal/experiments/<exp_dir>/outputs/<filename>.csv`
with a method tag in the filename, e.g.
`exp1_collins_rul_cnn_2022_ours.csv`,
`exp1_collins_rul_cnn_2022_hashemi_clipping.csv`.

Aggregate / comparison tables live at the same level with names like
`exp1_comparison_table.csv`.

### Verdicts

`verdict` ∈ `{UNSAT, SAT, UNKNOWN, TIMEOUT, ERROR, SKIPPED, NOT_APPLICABLE}`.

### Ground truth

For Exp 1 and Exp 2, ground truth comes from VNN-COMP 2025 sound-
verifier consensus (αβ-CROWN ∪ NeuralSAT ∪ PyRAT ∪ NNV ∪ NNEnum) read
from `~/v/other/VNNCOMP/vnncomp2025_results/<tool>/<bench_dir>/results.csv`.
A verdict is `sat` iff *any* sound verifier returned sat; same for
`unsat`; `conflict` if both appear; `unknown` if neither.

For Exp 3 ground truth is computed analytically (1-Lipschitz
networks with closed-form reach sets) or via αβ-CROWN on the synthetic
networks.

For Exp 4 ground truth is **UNSAT-by-construction** (Lipschitz-bounded
networks with specs at empirical-max + 0.1).

---

## Experiment 1 — Sound-Verifier Comparison

| | |
|---|---|
| **Goal** | Demonstrate ours is sound (0% FUR) where Hashemi-clipping isn't (~25–28% FUR at small `m`) at VNN-COMP-budget. |
| **Benchmarks** | `acasxu_2023`, `collins_rul_cnn_2022`, `dist_shift_2023`, `linearizenn_2024`, `tllverify_2023`, `malbeware`, `metaroom_2023`. (`vit_2023` was dropped — it appears in Exp 2 with the same hparams. `malbeware` (malware classification, 25 classes, 4096-d, 100s/row) and `metaroom_2023` (indoor-scene CNN, 20 classes, 5376-d, 210s/row) added for image-classification breadth; both use `verification_method='amls_bounded_union'` to fold their 19–24 disjuncts into a single AMLS chain — the same trick cifar100 uses in Exp 2.) |
| **Methods** | ours (bounded AMLS, locked per-benchmark configs), **Hashemi-clipping** (m=8000). Sound verifiers (αβ-CROWN, NeuralSAT, PyRAT, NNV, NNEnum, CORA, ROVER, sobolbox) read from VNN-COMP CSVs — no compute by us. Ground truth uses the SAT-wins rule across all 8 sound tools (see ``ground_truth.csv``). |
| **Per-benchmark hparam overrides** | Each benchmark's locked config lives in `exp1_vnncomp_subset/_benchmarks.py:PER_BENCHMARK_CONFIG`. **acasxu_2023**: `base` (n_train=5K, flow_epochs=2K, scenario_n=2K, max_levels=30). **collins_rul_cnn_2022**: `small` (1K/1K/500). **dist_shift_2023, linearizenn_2024, tllverify_2023**: `mega` (10K/2K/2K, max_levels=30). **metaroom_2023**: `mega` + `verification_method='amls_bounded_union'` (folds the 19-disjunct K-class spec into a single AMLS chain — same trick cifar100 uses in Exp 2). |
| **Falsifier** | **ON** for both ours and Hashemi-clipping with **identical** per-benchmark APGD-only budgets (most benchmarks `(n_restarts=3, n_steps=25)`, `dist_shift_2023` `(5, 50)`). Both call the same [`n2v.utils.falsify.falsify`](../../../n2v/utils/falsify.py) entrypoint with the same kwargs, so the only methodological difference between ours and Hashemi is the score function and calibrated set's geometry. |
| **K seeds** | K=1 |
| **N instances per benchmark** | full per-benchmark VNN-COMP `instances.csv` (range: ~30–186) |
| **Per-row VNN-COMP timeout** | yes — read from each `instances.csv` column 3 |
| **CSV outputs** | `exp1_<benchmark>_ours.csv`, `exp1_<benchmark>_hashemi_clipping.csv`, `exp1_<benchmark>_hashemi_naive.csv` (optional) |
| **Aggregate** | `exp1_comparison_table.csv` — one row per (benchmark, method) summarising verdict counts, walls, FUR |
| **Wall (compute by us)** | ours ~6–8 hr · Hashemi-clipping ~3–4 hr · *Hashemi-naive ~2–3 hr if run* |

### Expected per-instance CSV schema (columns)
```
benchmark, instance, method, verdict, wall_s, vnncomp_timeout_s,
coverage_empirical, coverage_n_test, q, epsilon_total, delta_total,
amls_bounded_eps_2_upper, amls_bounded_detected_unsafe,
amls_levels_used, ground_truth, ground_truth_source,
soundness_flag, error, timestamp
```
See [`CSV_SCHEMAS.md` §1](../CSV_SCHEMAS.md#1-experiment-1--vnn-comp-subset-sound-verifier-comparison) for full column defs.

### What's NOT in Exp 1 (per design)
- `cora_2024` — calibration-miss-infeasible at 30s budget
- `safenlp_2024` — calibration-miss-infeasible at 20s budget
- `vit_2023` — dropped from Exp 1 (lives in Exp 2 with identical hparams; no point running twice)
- `cifar100_2024` — too large; appears in Exp 2
- `sat_relu` — initially considered as cora replacement, rejected (focused on stable benchmark set)

---

## Experiment 2 — Probabilistic-Scale Comparison

| | |
|---|---|
| **Goal** | Probabilistic methods at scale — ours vs Hashemi-clipping + RS + SAVER + ProbStar where applicable. αβ-CROWN as sole sound reference. |
| **Benchmarks** | `cifar10_resnet110` (1.7M params, Cohen RS pretrained), `cifar100_2024` (2.5M params, ResNet-medium), `vit_2023` (76K params, 10-class ViT), `tinyimagenet_2024` (2.5M params, ResNet-medium, 200 classes) |
| **Methods (Tier-A — main results)** | ours (bounded AMLS, locked configs), **Hashemi-clipping** (m=8000), αβ-CROWN (computed by us this time, since some Exp 2 configs differ from VNN-COMP), **RS** (image classification only — applies to cifar10_resnet110, cifar100_2024) |
| **Methods (Tier-B — appendix, run if time)** | **ProbStar** (now that Gurobi is installed, run on benchmarks where the network is piecewise-linear — likely cifar10_resnet110, vit_2023; not cifar100/tinyimagenet's ResNet-medium with bn), **SAVER** (single-disjunct only — currently no Exp 2 benchmark with a 1-disjunct classification spec) |
| **Per-benchmark hparam overrides** | All four benchmarks use the same config: `mega` (n_train=10K, flow_epochs=2K, scenario_n=2K) + `verification_method='amls_bounded_union'` + `amls_max_levels=30`. The four specs are all multi-class disjunctive (cora-style nested OR — 9, 99, 199, 9 disjuncts respectively), so the union method handles them uniformly. cifar10_resnet110 uses 300s timeout (no VNN-COMP equivalent), the others use the canonical 100s VNN-COMP per-row budget. |
| **Falsifier** | **ON** for both ours and Hashemi-clipping with the same per-benchmark APGD budgets as Exp 1. Three of the four Exp 2 benchmarks are VNN-COMP and the fourth (`cifar10_resnet110`) is the same architecture family; keeping the falsify-first scaffold is consistent with the sound-verifier comparison. αβ-CROWN's intrinsic PGD-warmstart is inherent to that tool. RS / ProbStar / SaVer run on their own terms (no external falsification). |
| **K seeds** | K=1 |
| **N instances per benchmark** | 100 |
| **Timeouts** | **100s** per-row for vit_2023 / cifar100_2024 / tinyimagenet_2024 (matches VNN-COMP 2023/2024); **300s** for cifar10_resnet110 (no published VNN-COMP timeout) |
| **CSV outputs** | `exp2_<benchmark>_ours.csv`, `exp2_<benchmark>_hashemi_clipping.csv`, `exp2_<benchmark>_alpha_beta_crown.csv`, `exp2_<benchmark>_rs.csv`, *(later)* `exp2_<benchmark>_probstar.csv`, `exp2_<benchmark>_saver.csv` |
| **Aggregate** | `exp2_comparison_table.csv` |
| **Wall** | Tier-A: ours ~10–12 hr · Hashemi-clipping ~4 hr · αβ-CROWN ~6–10 hr · RS ~2 hr | Tier-B (later): ProbStar ~3–5 hr · SAVER ~1 hr |

### Expected per-instance CSV schema
Same as Exp 1, plus method-specific columns:
- RS: `sigma`, `n0`, `n_certify`, `alpha`, `pred_class`, `true_class`, `l2_radius`, `eps_linf_threshold_l2`
- ProbStar: `n_pieces`, `prob_unsafe`, `confidence`
- SAVER: `n_samples`, `epsilon`, `confidence`

See [`CSV_SCHEMAS.md` §2](../CSV_SCHEMAS.md#2-experiment-2--probabilistic-scale-comparison).

---

## Experiment 3 — Synthetic Geometric Validation

| | |
|---|---|
| **Goal** | Demonstrate the flow-shape advantage on **non-axis-aligned** output distributions where Hashemi's hyperrect bbox is intrinsically loose. Volume-comparable comparison. |
| **Benchmarks** | (a) **3D banana classifier** (`ThreeBlobClassifier3D` — three multimodal blobs with curved separators); (b) **5D / 10D / 20D 1-Lipschitz networks** with identity activation (closed-form exact volume); (c) **Geometric-transformation suite** — axis-aligned / rotated / translated / nonlinear input-set transforms applied to a base network, to stress geometry-awareness against fixed-form scores. |
| **Methods (combined with score-function ablation)** | ours with **4 score families**: hyperrect, ellipsoid (Mahalanobis), GMM (k=3), flow (FlowScore — naive). Plus Hashemi-naive, Hashemi-clipping. |
| **Spec types per benchmark** | (a) **trivially-UNSAT** spec (unsafe far from data) — should always be UNSAT; (b) **reachable-SAT** spec (unsafe inside reach support) — falsifier OFF, so ours should output UNKNOWN (validates honest-abstention behavior). |
| **K seeds** | K=5 |
| **CSV outputs** | `exp3_<benchmark>_<score>.csv` (one per score family per benchmark) |
| **Aggregate** | `exp3_volume_comparison.csv`, `exp3_geo_transforms.csv` |
| **Wall** | ~6–10 hr |

### Expected per-instance CSV schema
```
benchmark, score_function, seed, verdict,
volume_estimate, volume_exact, volume_ratio,
coverage_empirical, q, wall_s, error
```
Volume measurements are in the network's output space; `volume_exact`
is computable analytically for the 1-Lipschitz cells and via fine-grid
exhaustive sampling for the banana.

See [`CSV_SCHEMAS.md` §3](../CSV_SCHEMAS.md#3-experiment-3--synthetic-volume-comparison).

---

## Experiment 4 — Controlled Scaling **(NEW)**

| | |
|---|---|
| **Goal** | Demonstrate sub-exponential scaling of ours vs the exponential blow-up of sound verifiers on a controlled network family. The headline plot is wall-time vs network size, with TIMEOUT counts. |
| **Network family** | 7 spectrally-normalized ReLU MLPs with random Gaussian weights, **fixed width W=512**, depths `D ∈ {2, 4, 8, 16, 24, 32, 40}`. Param range **4K → 10M** (4 orders of magnitude). One network per depth (no init-seed variation). |
| **Verification problems per network** | **10 instances** generated as follows: pick a random "starting sample" `x_0 ~ Uniform([-1, 1]^5)`; the input box is the L∞ ball `[x_0 - eps, x_0 + eps]` (eps fixed at 0.1); the spec is `y > C` where `C = empirical_max(y over input box) + 0.1` — UNSAT by 1-Lipschitz construction. Each of the 10 instances has a different randomly-sampled `x_0`. |
| **Methods** | **ours (bounded AMLS, mega), αβ-CROWN, NeuralSAT, Hashemi-clipping (m=8000)** — all 4 are run *by us* on the synthetic networks (no VNN-COMP CSVs since the networks aren't standard VNN-COMP benchmarks). αβ-CROWN uses a custom config ([`abcrown_exp4_deep_mlp.yaml`](exp4_scaling/abcrown_exp4_deep_mlp.yaml)) tuned for deep ReLU MLPs (bs=1024, no PGD, input-split BaB) plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for OOM mitigation. **Expected sound-verifier failure modes** at deeper depths are documented in [`.claude/research/flow-matching-probabilistic-reach/sound-verifier-limitations.md`](../../../.claude/research/flow-matching-probabilistic-reach/sound-verifier-limitations.md): αβ-CROWN exhibits genuine exponential BaB-tree blowup (TIMEOUT at d ≥ 24); NeuralSAT exhibits abstractor saturation (UNKNOWN/`early_stop` at d ≥ 16). |
| **Total runs** | 7 × 10 × 4 = **280** |
| **Timeout** | 600s per (network, instance, method) |
| **CSV outputs** | `exp4_scaling_ours.csv`, `exp4_scaling_alpha_beta_crown.csv`, `exp4_scaling_neuralsat.csv`, `exp4_scaling_hashemi_clipping.csv` |
| **Aggregate** | `exp4_scaling_summary.csv` — one row per (depth, method) with median wall, p10/p90, TIMEOUT count, FALSE_UNSAT count |
| **Wall** | ~12–16 hr (TIMEOUT-dominated for sound verifiers at D≥16) |

### Expected per-instance CSV schema (NEW)
```
depth, width, n_params, instance_idx, method, verdict, wall_s,
timeout_s, x_0_seed, eps, spec_threshold, ground_truth,
amls_bounded_eps_2_upper, amls_levels_used, error, timestamp
```

`ground_truth = unsat` always (UNSAT-by-construction). A `verdict =
UNSAT` is correct; `UNKNOWN` is honest abstention; `SAT` would be a
soundness violation (should never occur for our method).

### Aggregate CSV schema
```
depth, width, n_params, method,
n_instances, n_unsat, n_unknown, n_sat, n_timeout, n_error,
median_wall_s, p10_wall_s, p90_wall_s,
n_false_unsat
```

This is what the heatmap and scaling-curve figures consume.

### Implementation notes for Exp 4
- Synthetic networks are generated programmatically with a deterministic
  seed per `(depth, width)` and saved as ONNX so αβ-CROWN / NeuralSAT
  can ingest them. The vnnlib spec files are also generated per
  instance.
- Network family file: `examples/FlowConformal/experiments/exp4_scaling/networks.py`.
- Per-method runner pattern: `exp4_run_<method>.py --depth <D> --instance <idx>`.

---

## Ablation Studies

| sweep | varying | range | wall |
|---|---|---|---|
| **Score function** | score family | naive flow / hyperrect / ellipsoid / GMM | combined with Exp 3 |
| **Verification method** | verification primitive | scenario / amls (unbounded) / **amls_bounded** / scenario_v2 / IS / Langevin | ~3 hr |
| **AMLS hyperparameters** | ρ, mcmc_steps | ρ ∈ {0.05, 0.1, 0.2}, mcmc ∈ {5, 10, 20, 40} | ~2 hr |
| **Conformal parameters** | α, m, ell, β_2 | α ∈ {0.001, 0.01, 0.05, 0.1}, m ∈ {500, 2000, 8000}, β_2 ∈ {0.001, 0.01, 0.1}, ell offset {0, 1, 5} | ~2 hr |
| **Flow training** | n_train, flow_epochs | n_train ∈ {1K, 2K, 5K, 10K, 20K, 50K}, flow_epochs ∈ {500, 1K, 2K, 5K} | ~3 hr |

| | |
|---|---|
| **Probe** | 10-instance ACAS Xu probe (4 persistent failures + 6 controls) |
| **K seeds** | K=1 |
| **CSV outputs** | `ablation_<sweep_tag>.csv` per sweep × value |
| **Aggregate** | `ablation_summary.csv` |
| **Wall** | ~10 hr total |

See [`CSV_SCHEMAS.md` §5](../CSV_SCHEMAS.md#5-ablation-suite).

---

## Soundness Audit

| | |
|---|---|
| **Goal** | Post-hoc sound verification of every UNSAT verdict from any method by AutoAttack + 5K-restart PGD on the original network. Catches false UNSATs that VNN-COMP ground truth might also have missed. |
| **Scope** | every UNSAT row from Exp 1, Exp 2, Phase 5e ACAS Xu, and Exp 4 |
| **Tools** | AutoAttack ([fra31/auto-attack](https://github.com/fra31/auto-attack), parameter-free APGD-CE + APGD-T + FAB-T + Square ensemble) and 5000-restart PGD with 100 steps |
| **Prerequisite** | AutoAttack: `pip install git+https://github.com/fra31/auto-attack.git`; timm: ✅ already installed |
| **CSV output** | `soundness_audit.csv` |
| **Wall** | ~3–5 hr |

Per-row schema:
```
benchmark, instance, method, claimed_verdict,
audit_method, audit_verdict, witness_x, witness_y,
audit_wall_s, audit_n_restarts, audit_pgd_steps
```

`audit_verdict = sat` on a row where `claimed_verdict = UNSAT` flags
a false UNSAT. We expect 0 such rows for ours; up to ~25-28% for
Hashemi-clipping at small `m` per probe v2.

---

## Execution order (priority hierarchy)

Running everything sequentially. Earlier items have higher priority —
if compute time runs out before reaching the lower items, the paper
remains coherent.

| order | what | wall | rationale |
|---:|---|---:|---|
| **1** | **Exp 1** — ours + Hashemi-clipping + SaVer on the 7 benchmarks (21 cells) | ~10-12 hr | core soundness story; all paper headline numbers come from here. SaVer (Tier-B) added so every Exp 1 row carries a probabilistic-baseline column. |
| **2** | **Exp 2 Tier-A** — ours + Hashemi-clipping + αβ-CROWN on the 4 benchmarks | ~17 hr | scaling story for probabilistic methods (RS moved to phase 5) |
| **3** | **Exp 3** — ours (4 score families) + Hashemi on synthetic | ~8 hr | geometry advantage |
| **4** | **Exp 4** — ours + αβ-CROWN + NeuralSAT + Hashemi-clipping on 7 synthetic networks | ~14 hr | controlled scaling |
| **5** | **Exp 2 RS** — Cohen randomized smoothing on cifar10_resnet110 + cifar100_2024 | ~5 hr | smoothing-baseline comparison on image classification only |
| **6** | **Soundness audit** — AutoAttack + PGD on every UNSAT verdict | ~4 hr | independent verification of paper claims |
| **7** | **Ablation studies** — verification method (incl. amls_bounded / amls_bounded_union vs scenario / amls / is_tilted / derived), AMLS hp, conformal hp, flow training, score family | ~7-8 hr | one-knob-at-a-time motivations |
| **8** | **Exp 2 Tier-B** — ProbStar + SaVer on the 4 benchmarks | ~8-12 hr | additional probabilistic-baseline comparison; ProbStar uniformly emits NOT_APPLICABLE on Exp 2 (StarV's loader rejects transformer attention, residual `Add`, and `Gemm` nodes — all 4 networks hit one of these limitations). The runner records the NOT_APPLICABLE rows to substantiate the gap claim. |

**Cumulative wall (sequential):** Exp 1 → Exp 4 inclusive ~ 49 hr · everything ~ 75-95 hr · with overnight scheduling, **~7-9 calendar days of experiment compute**.

### Things to do before starting

#### Implementation work

| item | status |
|---|---|
| ✅ Bounded AMLS implementation + `verification_method='amls_bounded'` plumbing | done |
| ✅ Loader audits (no bugs; wrappers match ONNX Runtime) | done |
| ✅ Lock-in probe for Exp 1/2 configs (locked offline; canonical hparams now committed in each benchmark's `_benchmarks.py:PER_BENCHMARK_CONFIG`) | done — original 5 Exp 1 benchmarks plus vit_2023, cifar100_2024, metaroom_2023, tinyimagenet_2024 all locked at mega + amls_bounded_union where appropriate. |
| ✅ acasxu_sweep.py refactored to SEED=47 + `--vnncomp-timeouts` + `--smoke` | done — smoke run gives UNSAT on (1,1)+prop_1 in 52.2s (under 116s VNN-COMP budget) |
| ✅ Union-AMLS implementation (`amls_bounded_estimate_union_mass`, `amls_bounded_certify_spec_union`, `verification_method='amls_bounded_union'`) | done — 10 unit tests pass |
| ✅ `amls_max_levels` override threads through `run_verification_pipeline` | done — verified with `test_max_levels_caps_union_loop` |
| ⏳ ACAS Xu sweep rerun under SEED=47 (replaces Phase 5e CSV; feeds Exp 1's ACAS Xu row) | pending — ~3 hr |
| ✅ Exp 4 synthetic-network family + 4 per-(depth, tool) runners | done — `experiments/exp4_scaling/` with `_benchmarks.py`, `instance_generator.py`, `networks.py`, and 4 runners (ours, hashemi_clipping, alpha_beta_crown, neuralsat) |
| ✅ Per-(experiment, tool) runner scripts | done — 15 runners total (Exp 1 × 3 [ours + hashemi + saver], Exp 2 × 5 [ours + hashemi + αβ-CROWN + RS + probstar], Exp 3 × 2 [ours + hashemi], Exp 4 × 4 [ours + hashemi + αβ-CROWN + neuralsat], plus standalone `baselines/run_probstar.py` for the starv conda env) all exposing `--list-instances` / `--instance-idx` / `--write-timeout-row` for `run_cell.sh` |

#### Tool installations (verify before any experiment runs)

Each tool needs an importable / runnable smoke test against a tiny
ACAS Xu instance (or a generated tiny MLP for AutoAttack/RS) before we
commit any compute to a full benchmark sweep. Smoke tests are written
into the per-tool runner script so a `--smoke` flag exercises the path
end-to-end on a 1-instance run.

| tool | install / location | smoke test | status |
|---|---|---|---|
| **αβ-CROWN** | clone at `~/v/other/alpha-beta-CROWN` | run `complete_verifier/abcrown.py --config <tiny>.yaml` on `acasxu_2023/onnx/ACASXU_run2a_1_1.onnx + prop_3.vnnlib` | ⏳ install verify + smoke |
| **NeuralSAT** | clone at `~/v/other/neuralsat` | run NeuralSAT's CLI on the same ACAS Xu (1,1) instance | ⏳ install verify + smoke |
| **timm** | `pip install timm` | `import timm; timm.create_model('resnet50')` | ✅ installed |
| **AutoAttack** | `pip install git+https://github.com/fra31/auto-attack.git` (Croce & Hein 2020 — APGD-CE + APGD-T + FAB-T + Square ensemble) | run `AutoAttack(model, norm='Linf', eps=0.1).run_standard_evaluation(x, y)` on a 2-layer MLP | ⏳ install + smoke |
| **PGD-5K-restart** | local — no install (custom loop in `n2v.utils.falsify`) | already in `falsify.py`; smoke is just unit tests | ✅ in repo |
| **Gurobi WLS license** | env var `GRB_LICENSE_FILE` | `python -c 'import gurobipy as gp; gp.Model()'` | ✅ done (per prior session) |
| **ProbStar** | dispatched via subprocess into the `starv` conda env (`baselines/run_probstar.py` standalone) | exp2_run_probstar `--smoke` on cifar100_2024 → NOT_APPLICABLE (StarV's loader rejects residual Add — expected) | ✅ smoke (NOT_APPLICABLE on every Exp 2 net) |
| **SAVER** | runs **in-process inside the n2v env** (no separate conda env; `baselines/run_saver.py` adds `~/v/other/SaVer-Toolbox` to `sys.path`) | exp1_run_saver `--smoke` on dist_shift_2023 / metaroom_2023 → UNKNOWN | ✅ smoke |
| **Cohen RS** | pretrained weights present | `torch.load(<rs_ckpt>)` and run `Smooth.certify` on 1 image | ✅ weights downloaded (per prior session) |

Each per-tool runner script (`expN_run_<benchmark>_<method>.py`)
includes a `--smoke` flag that runs *one* tiny instance and asserts
the verdict matches a hand-checked expected value. CI / pre-flight is:
run every script with `--smoke`, ensure all pass, only then start the
full sweeps.

---

## Per-experiment script call convention

**One script = one `(experiment, tool)` sweep, parameterised by `--benchmark`.**
Each runner script accepts a benchmark CLI argument and runs the entire
benchmark when called. Per-benchmark hparam overrides (config name,
`max_levels`, `verification_method`, etc.) live in a hardcoded
`PER_BENCHMARK_CONFIG` dict at the top of the runner. Loader logic is
factored into a per-experiment `_benchmarks.py` helper module so each
runner stays small.

```
expN_run_<method>.py --benchmark <name> [--output-csv <path>] [--smoke]
```

For example:
```bash
# Exp 1, ours on collins_rul_cnn (full sweep)
PYTHONPATH=. python examples/FlowConformal/experiments/exp1_vnncomp_subset/exp1_run_ours.py \
  --benchmark collins_rul_cnn_2022
# Exp 1, Hashemi-clipping on dist_shift
PYTHONPATH=. python examples/FlowConformal/experiments/exp1_vnncomp_subset/exp1_run_hashemi_clipping.py \
  --benchmark dist_shift_2023
# Exp 2, αβ-CROWN on cifar100
PYTHONPATH=. python examples/FlowConformal/experiments/exp2_prob_scale/exp2_run_alpha_beta_crown.py \
  --benchmark cifar100_2024
# Exp 4, NeuralSAT — depth selects the synthetic network
PYTHONPATH=. python examples/FlowConformal/experiments/exp4_scaling/exp4_run_neuralsat.py --depth 16
# Smoke (run 1 instance, assert hand-checked verdict)
PYTHONPATH=. python examples/FlowConformal/experiments/exp1_vnncomp_subset/exp1_run_ours.py \
  --benchmark vit_2023 --smoke
```

Each runner script:
1. Loads `PER_BENCHMARK_CONFIG[benchmark]` for hparam overrides
   (config name, `max_levels`, `verification_method`, etc.).
2. Calls the experiment's `_benchmarks.load_instances(benchmark)` helper
   to get the instance list and per-instance loader.
3. For each instance, resets `torch.manual_seed(SEED)` / `np.random.seed(SEED)` with `SEED=47`.
4. Calls the tool's verifier with `seed=SEED`.
5. Honours per-row VNN-COMP timeout for Exp 1/2 or the configured Exp 4 timeout.
6. Writes one CSV row per instance to `experiments/<exp_dir>/outputs/<filename>.csv`.

The `--smoke` flag runs only the first instance and asserts a
hand-checked expected verdict (e.g. for ACAS Xu (1,1)+prop_1: UNSAT;
for cifar100: matches αβ-CROWN VNN-COMP ground truth). All scripts
must pass `--smoke` before any full sweep is launched.

For long-running sweeps, prefer launching via `nohup` with `-u` so
output is line-buffered:
```bash
PYTHONPATH=. nohup python -u examples/FlowConformal/experiments/<exp>/<script>.py \
  --benchmark <name> > examples/FlowConformal/experiments/<exp>/outputs/<log>.log 2>&1 &
```

### Old prototype outputs

The CSVs currently sitting in `experiments/exp{1,2,3,_ablation}/outputs/`
are all `_smoke`-suffixed prototype runs from earlier figure-prototyping
work (~200 B–30 KB each). They are **safe to delete** — real experiment
runs use the canonical filenames (no `_smoke` suffix) per
`exp<N>_<benchmark>_<method>.csv` and will not collide.

`experiments/exp4_scaling/` does not exist yet; it will be created when
the synthetic-network family and Exp 4 runners are built.
