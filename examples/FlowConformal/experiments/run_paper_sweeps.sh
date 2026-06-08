#!/usr/bin/env bash
# Single canonical sweep script — runs every cell that feeds one of the
# paper's six final outputs and nothing else. Replaces the previous
# stack of run_all_sweeps / run_all_remaining / run_all_extra* scripts.
#
# Paper outputs and the cells that feed them:
#
#   1. paper/tables/outputs/main_table.tex
#   2. paper/tables/outputs/main_table_recall.tex
#   3. paper/tables/outputs/main_table_recall_compact.tex
#       ↳ Phase exp1 + Phase exp2 (per-(method, benchmark) CSVs)
#       ↳ joined against the per-experiment ground_truth.csv files
#       ↳ sound-verifier rows pulled from VNN-COMP 2025 results.csv
#         (no compute by us)
#
#   4. paper/tables/outputs/tab5_shared_flow_ablation.tex
#       ↳ Phase ablation (2 benchmarks × 5 methods, shared (flow, q))
#
#   5. paper/figures/fig4_exp4_scaling.png
#       ↳ Phase exp4 (4 methods × 7 network depths)
#
#   6. paper/figures/fig5_exp3_volume_comparison.png
#       ↳ Phase exp3 (7 benchmarks × 3 sample-budget configs ×
#                      {ours, hashemi_clipping, starset_approx})
#
# Usage:
#   bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
#        [--phase exp1|exp2|exp3|exp4|ablation|all] \
#        [--smoke] [--force] [--dry-run]
#
# --force  : allow OVERWRITING existing CSVs (default: each cell aborts
#            individually if its output already exists).
# --smoke  : 1-instance / 1-seed sanity check per cell, hard-capped at
#            10 min wall.
# --dry-run: print commands; do not execute.
#
# Wall-clock estimate (full sweep, no --smoke):
#   Phase exp4    ~3-4 hours   (αβ-CROWN dominates at high depth)
#   Phase exp1    ~2-3 hours   (acasxu_2023 saver/probstar are the long pole)
#   Phase exp2    ~3-4 hours   (200 instances × 100 s budget × 4 methods)
#   Phase exp3    ~3-4 hours   (synth_20d hashemi-large is the long pole)
#   Phase ablation ~4 hours    (acasxu_2023 × 5 verifiers)
#   Total: ~15-20 hours sequentially.

set -u

# Defaults are overridable: `PY=/some/python REPO=/elsewhere bash run_paper_sweeps.sh`.
# PY defaults to whatever `python` is on PATH (i.e. the activated n2v env);
# REPO is derived from this script's own location so it works from any clone.
PY="${PY:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PHASE=all
DRY_RUN=0
FORCE=0
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"

HARDCAP_SMOKE=600   # 10 min per smoke cell

log_start() { echo "[$(date -Iseconds)] === START: $2 ===" >> "$1"; }
log_end()   {
  echo "" >> "$1"
  echo "[$(date -Iseconds)] === END:   $2 (rc=$3) ===" >> "$1"
}

# Abort the cell if any of $@ already exists and --force was not given.
guard_no_overwrite() {
  local hit=0
  for f in "$@"; do
    if [[ -e "$f" ]]; then
      echo "  ABORT: would overwrite existing file: $f" >&2
      hit=1
    fi
  done
  if [[ $hit -eq 1 && $FORCE -eq 0 ]]; then
    echo "  Pass --force to overwrite. Stopping this cell." >&2
    return 1
  fi
  return 0
}

# Run a paper-experiment runner via the run_cell.sh per-instance
# shell-timeout wrapper. The wrapper iterates over the runner's
# --list-instances output and applies per-row timeouts read from
# instances.csv. Used for Exp 1 / Exp 2 / Exp 4.
run_cell() {
  local exp_dir="$1"     # e.g. exp2_prob_scale
  local module="$2"      # e.g. exp2_run_ours
  local args="$3"        # extra runner args (--benchmark X ...)
  local logname="$4"     # e.g. exp2_cifar100_2024_ours
  local logfile="examples/FlowConformal/experiments/$exp_dir/outputs/$logname.log"
  local out_csv="examples/FlowConformal/experiments/$exp_dir/outputs/$logname.csv"

  if ! guard_no_overwrite "$out_csv"; then
    return 1
  fi
  echo "[$(date +%H:%M:%S)] >>> $logname"
  if [[ $SMOKE -eq 1 ]]; then
    local cmd="timeout --kill-after=60 ${HARDCAP_SMOKE}s $PY -u -m examples.FlowConformal.experiments.$exp_dir.$module $args --smoke"
    echo "  [smoke] $cmd > $logfile"
    [[ $DRY_RUN -eq 1 ]] && return 0
    log_start "$logfile" "$logname"
    $cmd >> "$logfile" 2>&1
  else
    local cmd="bash examples/FlowConformal/experiments/run_cell.sh $exp_dir.$module $args"
    echo "  [full] $cmd > $logfile"
    [[ $DRY_RUN -eq 1 ]] && return 0
    log_start "$logfile" "$logname"
    bash examples/FlowConformal/experiments/run_cell.sh \
      "$exp_dir.$module" $args >> "$logfile" 2>&1
  fi
  local rc=$?
  log_end "$logfile" "$logname" "$rc"
  if [[ $rc -ne 0 ]]; then
    echo "  EXIT $rc — see $logfile"
  else
    tail -3 "$logfile" | sed 's/^/    /'
  fi
  return $rc
}

# Run a Python module directly (no per-instance wrapper). Used for Exp 3
# and the ablation, which manage their own instance loops in-process.
run_direct() {
  local label="$1"
  local logfile="$2"
  shift 2
  if ! guard_no_overwrite "${1:-}"; then
    # First positional after label/logfile is conventionally the output
    # CSV path (the runner writes it). For runners where we don't pass
    # an explicit output path, the caller should pass an empty string
    # to disable the guard. The first arg after --output-csv (if any)
    # is what we want to guard. Best-effort: skip-on-conflict.
    :
  fi
  echo "[$(date +%H:%M:%S)] >>> $label"
  echo "  cmd: $*"
  echo "  log: $logfile"
  [[ $DRY_RUN -eq 1 ]] && return 0
  mkdir -p "$(dirname "$logfile")"
  log_start "$logfile" "$label"
  if [[ $SMOKE -eq 1 ]]; then
    timeout --kill-after=60 ${HARDCAP_SMOKE}s "$@" >> "$logfile" 2>&1
  else
    "$@" >> "$logfile" 2>&1
  fi
  local rc=$?
  log_end "$logfile" "$label" "$rc"
  if [[ $rc -ne 0 ]]; then
    echo "  EXIT $rc — see $logfile"
  else
    tail -3 "$logfile" | sed 's/^/    /'
  fi
  return $rc
}

# ============================================================
# Phase exp1 — Exp 1, sound-verifier comparison rows
# ============================================================
EXP1_BENCHES=(acasxu_2023 collins_rul_cnn_2022 dist_shift_2023
              linearizenn_2024 tllverify_2023 metaroom_2023)

phase_exp1() {
  echo
  echo "===================== Phase exp1: Exp 1 (6 benchmarks × 4 methods) ====================="
  for bench in "${EXP1_BENCHES[@]}"; do
    # ours
    run_cell exp1_vnncomp_subset exp1_run_ours \
             "--benchmark $bench" \
             "exp1_${bench}_ours"
    # Hashemi: PCA variant for metaroom (high output dim), basic
    # clipping_block for the others. Mirrors main_table.csv_path.
    if [[ $bench == "metaroom_2023" ]]; then
      run_cell exp1_vnncomp_subset exp1_run_hashemi_clipping_pca \
               "--benchmark $bench --pca-components 10" \
               "exp1_${bench}_hashemi_clipping_pca"
    else
      run_cell exp1_vnncomp_subset exp1_run_hashemi_clipping \
               "--benchmark $bench" \
               "exp1_${bench}_hashemi_clipping"
    fi
    # SaVer (delta=0.05 baked into run_saver.py per Phase 5b fix)
    run_cell exp1_vnncomp_subset exp1_run_saver \
             "--benchmark $bench" \
             "exp1_${bench}_saver"
    # ProbStar (StarV; emits NOT_APPLICABLE on most network ops)
    run_cell exp1_vnncomp_subset exp1_run_probstar \
             "--benchmark $bench" \
             "exp1_${bench}_probstar"
  done
}

# ============================================================
# Phase exp2 — Exp 2, probabilistic-scale comparison rows
# ============================================================
EXP2_BENCHES=(vit_2023 tinyimagenet_2024 cifar100_2024)
N_EXP2_FULL=201

phase_exp2() {
  echo
  echo "===================== Phase exp2: Exp 2 (3 benchmarks × 4 methods + RS on cifar100) ====================="
  for bench in "${EXP2_BENCHES[@]}"; do
    run_cell exp2_prob_scale exp2_run_ours \
             "--benchmark $bench --n-instances $N_EXP2_FULL" \
             "exp2_${bench}_ours"
    # Hashemi-PCA m=2500: wall-matched to ours, K=32 PCA components
    # mirrors the published clipping-block paper's high-output-dim
    # recommendation. The headline paper rows for these 3 benchmarks
    # come from the PCA variant, not raw clipping.
    run_cell exp2_prob_scale exp2_run_hashemi_clipping_pca \
             "--benchmark $bench --pca-components 32 --m 2500 --n-instances $N_EXP2_FULL" \
             "exp2_${bench}_hashemi_clipping_pca"
    run_cell exp2_prob_scale exp2_run_saver \
             "--benchmark $bench --n-instances $N_EXP2_FULL" \
             "exp2_${bench}_saver"
    run_cell exp2_prob_scale exp2_run_probstar \
             "--benchmark $bench --n-instances $N_EXP2_FULL" \
             "exp2_${bench}_probstar"
  done
  # Cohen-style randomized smoothing — only cifar100_2024 has a smoothed
  # classifier with the matching σ; vit/tinyimagenet would be
  # NOT_APPLICABLE (no pretrained smoothed weights available).
  run_cell exp2_prob_scale exp2_run_rs \
           "--benchmark cifar100_2024 --n-instances $N_EXP2_FULL" \
           "exp2_cifar100_2024_rs"
}

# ============================================================
# Phase exp3 — Exp 3, synthetic volume-comparison
# ============================================================
EXP3_BENCHES=(2d_banana 3d_banana synth_2d synth_3d
              synth_5d synth_10d synth_20d)

# Sample-budget configs. Filenames keep the legacy "default" suffix
# (rather than "medium") because the figure generator parses the
# config from the filename and the v1/v2 sweeps already produced
# CSVs with this naming.
declare -A OURS_CONFIG_ARGS=(
  [small]="--n-train 2000 --flow-epochs 500 --scenario-n-samples 1000 --volume-m 1000 --volume-n-samples 100000"
  [default]="--n-train 5000 --flow-epochs 2000 --scenario-n-samples 2000 --volume-m 8000 --volume-n-samples 200000"
  [large]="--n-train 10000 --flow-epochs 5000 --scenario-n-samples 4000 --volume-m 16000 --volume-n-samples 400000"
)
declare -A HASHEMI_CONFIG_ARGS=(
  [small]="--m 1000"
  [default]="--m 8000"
  [large]="--m 16000"
)

phase_exp3() {
  echo
  echo "===================== Phase exp3: Exp 3 (7 benchmarks × 3 configs × 2 methods + starset baseline) ====================="
  local out_dir="examples/FlowConformal/experiments/exp3_synthetic/outputs"
  local seeds_arg=""
  if [[ $SMOKE -eq 1 ]]; then
    seeds_arg="--smoke"
  else
    seeds_arg="--seeds 5"
  fi
  for bench in "${EXP3_BENCHES[@]}"; do
    for cfg in small default large; do
      local ours_csv="$out_dir/exp3_${bench}_flow_unsat_ours_${cfg}.csv"
      local ours_log="$out_dir/exp3_${bench}_flow_unsat_ours_${cfg}.log"
      if guard_no_overwrite "$ours_csv"; then
        run_direct "exp3_${bench}_ours_${cfg}" "$ours_log" \
          $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours \
          --benchmark "$bench" --score flow --spec unsat \
          $seeds_arg --output-csv "$ours_csv" \
          ${OURS_CONFIG_ARGS[$cfg]}
      fi
      local hash_csv="$out_dir/exp3_${bench}_unsat_hashemi_clipping_${cfg}.csv"
      local hash_log="$out_dir/exp3_${bench}_unsat_hashemi_clipping_${cfg}.log"
      if guard_no_overwrite "$hash_csv"; then
        run_direct "exp3_${bench}_hashemi_${cfg}" "$hash_log" \
          $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_hashemi_clipping \
          --benchmark "$bench" --spec unsat \
          $seeds_arg --output-csv "$hash_csv" \
          ${HASHEMI_CONFIG_ARGS[$cfg]}
      fi
    done
    # Sound deterministic baseline — single config (no m axis).
    local star_csv="$out_dir/exp3_${bench}_unsat_starset_approx.csv"
    local star_log="$out_dir/exp3_${bench}_unsat_starset_approx.log"
    if guard_no_overwrite "$star_csv"; then
      run_direct "exp3_${bench}_starset_approx" "$star_log" \
        $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_starset_approx \
        --benchmark "$bench" --spec unsat \
        $seeds_arg --output-csv "$star_csv"
    fi
  done
}

# ============================================================
# Phase exp4 — Exp 4, controlled scaling on 1-Lipschitz family
# ============================================================
EXP4_DEPTHS=(2 4 8 16 24 32 40)
EXP4_TOOLS=(ours hashemi_clipping alpha_beta_crown neuralsat)

phase_exp4() {
  echo
  echo "===================== Phase exp4: Exp 4 (4 methods × 7 depths) ====================="
  # Tool-major × depth-minor: each tool runs every depth before the
  # next tool starts. Lets the log reader watch a single tool's
  # full scaling curve before the next begins.
  for tool in "${EXP4_TOOLS[@]}"; do
    for depth in "${EXP4_DEPTHS[@]}"; do
      run_cell exp4_scaling "exp4_run_$tool" \
               "--depth $depth" \
               "exp4_d${depth}_${tool}"
    done
  done
}

# ============================================================
# Phase ablation — verification-method ablation
# ============================================================
ABL_METHODS=(scenario amls amls_bounded is_tilted raw_mc_uniform)
declare -A ABL_INSTANCES=(
  [acasxu_2023]=186
  [tllverify_2023]=32
)

phase_ablation() {
  echo
  echo "===================== Phase ablation: shared-flow verifier ablation (2 benchmarks × 5 methods) ====================="
  local out_dir="examples/FlowConformal/experiments/exp_ablation/outputs"
  for bench in acasxu_2023 tllverify_2023; do
    local prefix="ablation_shared_flow_${bench}"
    # Per-method CSVs the cell will write. Guard each.
    local targets=()
    for m in "${ABL_METHODS[@]}"; do
      targets+=("$out_dir/${prefix}_${m}.csv")
    done
    if ! guard_no_overwrite "${targets[@]}"; then
      continue
    fi
    local n=${ABL_INSTANCES[$bench]}
    run_direct "ablation_${bench}" "$out_dir/${prefix}.log" \
      $PY -u -m examples.FlowConformal.experiments.exp_ablation.ablation_shared_flow \
      --benchmark "$bench" --n-instances $n \
      --methods "${ABL_METHODS[@]}"
  done
}

# ============================================================
# Dispatcher
# ============================================================
case "$PHASE" in
  exp1)     phase_exp1 ;;
  exp2)     phase_exp2 ;;
  exp3)     phase_exp3 ;;
  exp4)     phase_exp4 ;;
  ablation) phase_ablation ;;
  all)
    phase_exp1
    phase_exp2
    phase_exp3
    phase_exp4
    phase_ablation
    ;;
  *)
    echo "unknown phase: $PHASE (valid: exp1, exp2, exp3, exp4, ablation, all)" >&2
    exit 2
    ;;
esac

echo
echo "[$(date +%H:%M:%S)] === run_paper_sweeps complete ==="
