# Reproducing the FlowConformal experiments

End-to-end recipe to reproduce every paper experiment from a clean
checkout, including conda-env creation, pretrained-weight setup,
smoke validation, full sweep execution, and figure generation.

**Audience:** anyone re-running the experiments — the paper authors on
a fresh machine, reviewers, or future maintainers. Assume a Linux box
with an NVIDIA GPU (CUDA 12.x driver) and conda installed.

**Total wall time:** roughly **15-20 hours of GPU compute** for the
full paper sweep (Exp 1, Exp 2, Exp 3, Exp 4, and the
verification-method ablation). Multi-GPU parallelisation across
phases brings it down to a single overnight run.

---

## 0. Prerequisites

```bash
# clone + cd (use --recurse-submodules to pull the onnx2torch fork; see below).
git clone --recurse-submodules <repo-url> n2v && cd n2v

# system libs (Ubuntu/Debian); root only.
sudo apt install libglpk-dev libgmp3-dev    # for ProbStar/StarV (Tier-B)
```

### 0a. onnx2torch submodule pin

The repo vendors a small fork of [onnx2torch](https://github.com/sammsaski/onnx2torch.git)
under `third_party/onnx2torch/`, pinned to a specific commit that contains
three local patches required for VNN-COMP 2025 networks:

- `batch_norm.py`: rank-agnostic BatchNorm path (needed for `vit_2023` in Exp 2).
- `clip.py`: treats empty-string input slot as missing optional input (ONNX-spec compliance fix).
- `slice.py`: typo fix in `OnnxSliceV9.forward` + accept list/numpy `axes`/`steps`.

Fresh clones with `--recurse-submodules` get the right version automatically.
If you cloned without it:

```bash
git submodule update --init --recursive
```

The pin lives in `.gitmodules` and the supermodule commit — do not reset it
unless you mean to.

GPU driver must be **≥ 525.60** (CUDA 12.1+). Verify with `nvidia-smi`.

### 0b. Parity verification (historical)

The post-NeurIPS cleanup refactor migrated the runners from the legacy
`run_verification_pipeline` to the new three-stage public API
(`falsify` → `NeuralNetwork.reach(method='flow_matching')` →
`verify_specification`). A parity smoke test
(`examples/FlowConformal/smokes/refactor_parity.py`, deleted at Phase 17
after the migration was complete) asserted that the new API produced
bit-identical `q` / `verdict` / `epsilon_total` on a representative
instance. See [.claude/plans/neurips-cleanup-refactor.md](../../.claude/plans/neurips-cleanup-refactor.md)
for the full record. Future drift is now caught by the unit test suite
(`pytest tests/unit/probabilistic/`).

---

## 1. Conda environments

We use one env per tool to avoid dep conflicts (each verifier has
incompatible torch/numpy pins).

| env | python | torch | role |
|---|---|---|---|
| `n2v` | 3.12 | 2.7.x+cu118 | our core method, Hashemi, RS, AutoAttack, **SaVer** (imported in-process from `~/v/other/SaVer-Toolbox`) |
| `alpha-beta-crown` | 3.11 | 2.7.1+cu118 | αβ-CROWN |
| `neuralsat` | 3.10 | 2.1.2+cu118 | NeuralSAT |
| `starv` | 3.10 | 2.7.1+cu118 (numpy<=1.26.4) | ProbStar (Tier-B); subprocess-dispatched because StarV's numpy<=1.26.4 pin is incompatible with n2v |

### 1a. n2v env

```bash
# Existing env that the project uses; create or activate it.
# (If creating fresh, install n2v's own requirements first; see repo root.)
~/miniconda3/envs/n2v/bin/pip install git+https://github.com/fra31/auto-attack.git
~/miniconda3/envs/n2v/bin/pip install timm   # if not already
```

### 1b. αβ-CROWN

```bash
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git \
    ~/v/other/alpha-beta-CROWN
cd ~/v/other/alpha-beta-CROWN
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
# Their pinned torch is cu128; downgrade to cu118 for older drivers:
~/miniconda3/envs/alpha-beta-crown/bin/pip install --force-reinstall \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118
cd -
```

Quick smoke (should print `Result: unsat` in <30s):
```bash
~/miniconda3/envs/alpha-beta-crown/bin/python \
    ~/v/other/alpha-beta-CROWN/complete_verifier/abcrown.py \
    --config ~/v/other/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp21/acasxu.yaml \
    --onnx_path ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx \
    --vnnlib_path ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_1.vnnlib
```

### 1c. NeuralSAT

```bash
git clone https://github.com/dynaroars/neuralsat.git ~/v/other/neuralsat
conda create -n neuralsat python=3.10 -c conda-forge -y
~/miniconda3/envs/neuralsat/bin/pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118
conda install -n neuralsat -c gurobi gurobi -y
~/miniconda3/envs/neuralsat/bin/pip install -r ~/v/other/neuralsat/requirements.txt
~/miniconda3/envs/neuralsat/bin/pip install "setuptools<80"   # so torch.utils.cpp_extension imports
```

Quick smoke (should print `unsat,<small>` on the last line):
```bash
cd ~/v/other/neuralsat && \
~/miniconda3/envs/neuralsat/bin/python src/main.py \
    --net ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx \
    --spec ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_1.vnnlib \
    --device cuda --timeout 60
```

### 1d. ProbStar / SaVer (Tier-B; defer if not running Phase 1 SaVer cells or Phase 8)

```bash
# StarV / ProbStar (subprocess-dispatched into its own env)
conda create -n starv python=3.10 -c conda-forge -y
~/miniconda3/envs/starv/bin/pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda install -n starv -c gurobi gurobi -y
~/miniconda3/envs/starv/bin/pip install -e ~/v/other/StarV
~/miniconda3/envs/starv/bin/pip install "numpy<=1.26.4"

# SaVer (no separate env — runs in-process from n2v env via a sys.path
# insert). Just clone SaVer-Toolbox to ~/v/other/SaVer-Toolbox; the
# baseline runner picks it up automatically.
git clone https://github.com/<saver-toolbox-fork> ~/v/other/SaVer-Toolbox
```

### 1e. Cohen RS pretrained weights (for Exp 2 cifar10_resnet110)

```bash
# Follow Cohen et al. 2019 instructions to download checkpoint.pth.tar
# into ~/v/other/smoothing/models/cifar10/resnet110/noise_<sigma>/
# See https://github.com/locuslab/smoothing for the gdrive link.
```

---

## 2. VNN-COMP benchmark + results data

Clone the VNN-COMP 2025 repos under `~/v/other/VNNCOMP/`:

```bash
mkdir -p ~/v/other/VNNCOMP
cd ~/v/other/VNNCOMP
git clone https://github.com/stanleybak/vnncomp2025_benchmarks.git
git clone https://github.com/<vnncomp25-results>.git vnncomp2025_results
```

The benchmark dir layout we expect:
* `vnncomp2025_benchmarks/benchmarks/<bench>/onnx/*.onnx`
* `vnncomp2025_benchmarks/benchmarks/<bench>/vnnlib/*.vnnlib`
* `vnncomp2025_benchmarks/benchmarks/<bench>/instances.csv`
* `vnncomp2025_results/<tool>/results.csv`  (consensus ground truth)

---

## 3. Pre-flight smoke

Before committing many GPU-hours to the full sweep, run a smoke pass
to catch broken installs / config mismatches:

```bash
cd /path/to/n2v
bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    --smoke --dry-run
```

`--dry-run` prints the command list without executing; drop it to
actually run smokes (~30-60 min total, 1 instance per cell, 10 min
hard-cap each). Each cell writes a 1-row CSV and prints
`[smoke] PASS` / `[smoke] FAIL`.

---

## 4. Run experiments

Sequential execution of every cell that feeds a paper output
(~15-20 hr total):

```bash
bash examples/FlowConformal/experiments/run_paper_sweeps.sh
```

Phase-by-phase (recommended for parallelising across GPUs / nights):

```bash
# Exp 1 — 6 benchmarks × 4 methods (ours + Hashemi-clipping(-PCA on
# metaroom) + SaVer + ProbStar). ~2-3 hours.
nohup bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    --phase exp1 > /tmp/exp1.log 2>&1 &

# Exp 2 — probabilistic-scale (3 benchmarks × 4 methods + RS on
# cifar100). ~3-4 hours.
nohup bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    --phase exp2 > /tmp/exp2.log 2>&1 &

# Exp 3 — synthetic volume comparison (7 benchmarks × 3 sample-budget
# configs × {ours, Hashemi-clipping, starset-approx}). ~3-4 hours.
nohup bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    --phase exp3 > /tmp/exp3.log 2>&1 &

# Exp 4 — controlled scaling (4 methods × 7 depths). ~3-4 hours
# (αβ-CROWN dominates wall-time at high depth).
nohup bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    --phase exp4 > /tmp/exp4.log 2>&1 &

# Verification-method ablation (shared-flow on acasxu_2023 +
# tllverify_2023 × 5 verifiers). ~4 hours.
nohup bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    --phase ablation > /tmp/ablation.log 2>&1 &
```

Each cell's output CSV is guarded by `--force` — by default the
launcher aborts an individual cell if its output already exists,
preserving the data on disk.

---

## 5. Paper tables and figures

The paper-output generators read per-instance CSVs from the canonical
experiment-outputs directories and emit `.tex` / `.png` under
`examples/FlowConformal/paper/{tables,figures}/outputs/`. No
intermediate aggregate step is needed.

```bash
# Headline tables: % Solved + UNSAT-recall (compact and full).
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.tables.main_table
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.tables.main_table_recall
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.tables.main_table_recall_compact

# Shared-flow verification-method ablation (5 methods × 2 benchmarks):
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.tables.tab5_shared_flow_ablation

# Exp 4 scaling figure:
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.figures.fig4_exp4_scaling

# Exp 3 volume-comparison figure:
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.figures.fig5_exp3_volume_comparison
```

Figures and tables read directly from per-instance CSVs using the
column conventions in [`CSV_SCHEMAS.md`](CSV_SCHEMAS.md).

---

## 6. Reproducibility guarantees

* **Single global seed:** `SEED = 47`. Every per-(benchmark, tool)
  runner resets `torch.manual_seed(47)` and `np.random.seed(47)` at
  the start of each instance's pipeline. Re-running any sweep
  produces bit-identical CSVs (modulo MCMC numerical reproducibility,
  which torch generators guarantee on a fixed CUDA device).

* **Order-independent:** the per-instance reset means reordering
  instances within a benchmark doesn't change any individual row.

* **Cross-tool deterministic:** the same instance's input box,
  calibration data, and spec are bit-identical regardless of which
  tool runs (the seed is set at the runner level, not by per-tool
  sub-seeding).

* **Synthetic instance generation** (Exp 3, Exp 4) uses
  `_stable_hash` (hashlib SHA-256) for cross-process determinism;
  Python's built-in `hash()` is randomised per-process and would
  break reproducibility.

---

## 7. Files of interest

* `examples/FlowConformal/experiments/README.md` — paper-experiment
  spec (benchmark choice, hparam overrides per benchmark, expected
  CSV outputs, execution order).
* `examples/FlowConformal/experiments/run_paper_sweeps.sh` — single
  canonical launcher with `--phase`, `--smoke`, `--dry-run`,
  `--force` flags.
* `examples/FlowConformal/experiments/_external_verifiers.py` —
  subprocess wrappers for αβ-CROWN, NeuralSAT, and ProbStar
  (each dispatches into its own conda env to avoid dep conflicts).
* `examples/FlowConformal/experiments/baselines/run_probstar.py` —
  zero-n2v-dep ProbStar standalone executed inside the `starv` env.
* `examples/FlowConformal/CSV_SCHEMAS.md` — canonical CSV schema for every per-instance
  and per-aggregate output.
