#!/usr/bin/env bash
# VNN-COMP-style outer loop with per-instance shell timeout.
#
# Usage:
#   run_cell.sh <module> [runner-args...]
#
# where ``module`` is the dotted module path under
# ``examples.FlowConformal.experiments`` (e.g.
# ``exp1_vnncomp_subset.exp1_run_ours``) and ``runner-args`` are
# whatever the underlying runner accepts (``--benchmark <X>``,
# ``--depth <D>``, ``--output-csv <path>``, etc.).
#
# This wrapper:
#   1. Calls the runner with ``--list-instances`` to get a list of
#      ``"<idx> <timeout_s>"`` pairs.
#   2. For each pair, invokes the runner with ``--instance-idx <idx>``
#      under a shell ``timeout`` of ``<timeout_s>+buffer``.
#      The runner appends one CSV row per instance.
#   3. On per-instance shell-timeout (exit 124), logs the timeout and
#      moves on to the next instance — this is the key hang-resilience
#      property: a single hung instance can't hold up the rest of the
#      cell.
#
# Examples:
#   bash run_cell.sh exp1_vnncomp_subset.exp1_run_ours --benchmark acasxu_2023
#   bash run_cell.sh exp4_scaling.exp4_run_ours --depth 24
#
# The instance-level shell timeout is the production pattern used by
# run_paper_sweeps.sh for Exp 1 / Exp 2 / Exp 4. For one-off / smoke
# runs where you want fast feedback and don't worry about hangs,
# calling the runner directly (without this wrapper) is fine.
set -u

PY=${PY:-python}
TIMEOUT_BUFFER=${TIMEOUT_BUFFER:-60}    # seconds added on top of the per-instance budget
KILL_AFTER=${KILL_AFTER:-30}            # SIGKILL grace after SIGTERM
START_IDX=${START_IDX:-0}               # 0-indexed instance to start from
                                        # (skip 0..START_IDX-1; useful when
                                        # resuming a partially-done sweep
                                        # whose CSV already has the early rows)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <module> [runner-args...]" >&2
  echo "  module: dotted module path under examples.FlowConformal.experiments" >&2
  echo "  runner-args: passed verbatim to the underlying runner" >&2
  echo "" >&2
  echo "Environment overrides:" >&2
  echo "  PY              python interpreter (default: python, i.e. the activated env)" >&2
  echo "  TIMEOUT_BUFFER  seconds added to each instance's shell timeout (default: 60)" >&2
  echo "  KILL_AFTER      SIGKILL grace seconds after SIGTERM (default: 30)" >&2
  echo "  START_IDX       0-indexed instance to start from; skip earlier (default: 0)" >&2
  exit 2
fi

MODULE="$1"
shift
RUNNER_ARGS=("$@")

# Step 1: get the instance list.
LIST=$("$PY" -m "examples.FlowConformal.experiments.$MODULE" \
       "${RUNNER_ARGS[@]}" --list-instances 2>&1)
LIST_RC=$?
if [[ $LIST_RC -ne 0 ]]; then
  echo "Failed to list instances (exit $LIST_RC):" >&2
  echo "$LIST" >&2
  exit $LIST_RC
fi

TOTAL=$(echo "$LIST" | grep -c '^[0-9]')
if [[ $TOTAL -eq 0 ]]; then
  echo "No instances reported by $MODULE --list-instances" >&2
  exit 2
fi

echo "[run_cell] $MODULE ${RUNNER_ARGS[*]}: $TOTAL instances (START_IDX=$START_IDX)"

# Step 2: per-instance loop with shell timeout.
COUNT=0
SKIPPED=0
TIMEOUTS=0
ERRORS=0
START=$(date +%s)
echo "$LIST" | while IFS=' ' read -r IDX VNNCOMP_T; do
  # Skip blanks / non-numeric lines.
  [[ "$IDX" =~ ^[0-9]+$ ]] || continue
  # Skip indices below START_IDX (resume a partially-done sweep without
  # re-running / duplicating the early rows already in the CSV).
  if (( IDX < START_IDX )); then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  COUNT=$((COUNT + 1))
  HARD_T=$((VNNCOMP_T + TIMEOUT_BUFFER))

  ELAPSED=$(($(date +%s) - START))
  printf "[run_cell %4d/%4d t=%5ds] idx=%-4d budget=%ds (shell timeout=%ds)\n" \
         "$COUNT" "$TOTAL" "$ELAPSED" "$IDX" "$VNNCOMP_T" "$HARD_T"

  timeout --kill-after="${KILL_AFTER}s" "${HARD_T}s" \
    "$PY" -m "examples.FlowConformal.experiments.$MODULE" \
        "${RUNNER_ARGS[@]}" --instance-idx "$IDX"
  RC=$?

  if [[ $RC -eq 124 ]]; then
    echo "  ... idx=$IDX TIMEOUT (shell killed at ${HARD_T}s; runner did not finish)"
    TIMEOUTS=$((TIMEOUTS + 1))
    # Runner died mid-instance — invoke it again (cheaply, no-op load)
    # to append a TIMEOUT row to the CSV. Mirrors VNN-COMP's pattern of
    # writing a verdict row even when the tool was killed.
    "$PY" -m "examples.FlowConformal.experiments.$MODULE" \
        "${RUNNER_ARGS[@]}" --instance-idx "$IDX" --write-timeout-row \
        || echo "  ... idx=$IDX WARNING: --write-timeout-row failed (csv may be missing this row)"
  elif [[ $RC -eq 137 ]]; then
    echo "  ... idx=$IDX SIGKILL (likely OOM or stuck process)"
    ERRORS=$((ERRORS + 1))
  elif [[ $RC -ne 0 ]]; then
    echo "  ... idx=$IDX exit non-zero ($RC)"
    ERRORS=$((ERRORS + 1))
  fi
done

echo "[run_cell] complete. shell-killed timeouts and exits noted above; CSV row counts in the output file."
