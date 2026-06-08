"""Shared helpers consolidated from the per-experiment runners post-NeurIPS.

This module holds DRY consolidations extracted from the
``exp{1,2,3,4}_*_run_*.py`` runner scripts. Each helper here is
behaviour-preserving relative to the pre-consolidation copies it
replaces: it does exactly the same file-system, CSV-writing, and
verdict-aggregation work, just from a single source.

The consolidation done here covers only the SAFEST clusters from the
DRY audit (.claude/issues.md I-24):

- **DUP-005** — :func:`append_csv_row_with_defaults`: the file-open /
  header / row-write / flush boilerplate that was duplicated across the
  ~18 ``_write_timeout_row`` functions. Each caller still owns its own
  ``_FIELDS`` list and its own row-dict construction (which vary across
  runners); only the plumbing is shared.
- **DUP-004** — :func:`aggregate_box_verdicts`: the per-box ->
  per-instance Bonferroni aggregation used by ``exp1_run_ours`` and
  ``exp2_run_ours``. SAT short-circuits; all-UNSAT sums epsilons and
  intersects deltas; otherwise UNKNOWN.

# TODO(post-paper): consolidate remaining DUP-* clusters from I-24
#
# The decision to consolidate the rest of the duplications (DUP-001,
# DUP-002, DUP-003, DUP-006, DUP-007, DUP-008, DUP-009 per
# .claude/issues.md I-24) is DEFERRED until paper reproducibility
# freezes are no longer a concern. Those clusters touch RNG draw
# order, CSV column construction, or main() scaffolds where a careless
# refactor would shift byte-exact paper outputs. Revisit at the next
# experimental campaign.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


def append_csv_row_with_defaults(
    out_csv: Path,
    fields: Iterable[str],
    updates: Mapping[str, Any],
) -> None:
    """Append one row to ``out_csv``, writing the header iff the file
    is empty/new.

    All fields are initialised to ``''`` and then overlaid with
    ``updates``. This matches the pattern used by every per-runner
    ``_write_timeout_row`` (DUP-005) byte-for-byte: it preserves the
    existing field ordering (taken from each runner's ``_FIELDS``),
    the file-existence test (``exists() and stat().st_size > 0``), the
    ``'a' if file_exists else 'w'`` open mode with ``newline=''``, and
    the ``writeheader()`` + ``flush()`` -> ``writerow()`` + ``flush()``
    sequencing.
    """
    fieldnames = list(fields)
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    with open(out_csv, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            f.flush()
        row = {_f: '' for _f in fieldnames}
        row.update(updates)
        writer.writerow(row)
        f.flush()


def aggregate_box_verdicts(box_results: list[dict]) -> dict:
    """Aggregate per-box verdicts to a single per-instance result dict.

    Used for OR-of-input-regions (Bonferroni) aggregation across
    multiple input boxes for a single VNN-LIB instance — see
    ``exp1_run_ours._run_one_instance`` and
    ``exp2_run_ours._run_one_instance``.

    Aggregation rule:
      - any SAT box => return the first SAT result (caller already
        short-circuits the per-box loop on the first SAT, so this
        branch is reached only if a SAT exists in ``box_results``);
      - all UNSAT => return ``box_results[0]`` with ``epsilon_total``
        replaced by the sum across boxes and ``delta_total`` replaced
        by the min across boxes (Bonferroni over disjoint input
        regions); the result is copied with ``dict(result)`` so the
        original per-box record is not mutated;
      - otherwise => return the first UNKNOWN result.

    This function is byte-equivalent to the inlined block at
    ``exp1_run_ours.py:122-137`` and ``exp2_run_ours.py:152-166``
    (same iteration order, same epsilon/delta combiners, same fallback
    defaults of ``0.0`` for missing epsilons and ``1.0`` for missing
    deltas).
    """
    verdicts = [r['verdict'] for r in box_results]
    if 'SAT' in verdicts:
        result = next(r for r in box_results if r['verdict'] == 'SAT')
    elif all(v == 'UNSAT' for v in verdicts):
        result = box_results[0]
        if len(box_results) > 1:
            eps_sum = sum(
                (r.get('epsilon_total') or 0.0) for r in box_results)
            delta_min = min(
                (r.get('delta_total') or 1.0) for r in box_results)
            result = dict(result)
            result['epsilon_total'] = eps_sum
            result['delta_total'] = delta_min
    else:
        result = next(r for r in box_results if r['verdict'] == 'UNKNOWN')
    return result
