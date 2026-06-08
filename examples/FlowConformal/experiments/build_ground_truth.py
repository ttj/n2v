"""Build per-experiment ground-truth CSVs from VNN-COMP 2025 results.

Reads ``~/v/other/VNNCOMP/vnncomp2025_results/<tool>/results.csv`` for
all 8 sound verifiers (αβ-CROWN, NeuralSAT, PyRAT, NNV, NNEnum, CORA,
ROVER, sobolbox) and computes per-instance ground truth using the
SAT-wins rule:

* If any tool reports ``sat`` → ground truth = ``sat``.
  Tools that reported ``unsat`` for the same instance are recorded as
  ``dissenting_tools`` — those are soundness-violation suspects.
* Else if at least one tool reports ``unsat`` (and no tool reports
  ``sat``) → ground truth = ``unsat``.
* Else (all tools timed out / errored / unknown) → ground truth = ``unknown``.

The rationale: SAT is an existence claim verifiable by inspecting the
counterexample, so even one tool finding it is decisive. UNSAT is a
universal claim and any tool with a soundness bug can produce a
spurious UNSAT — but the SAT-wins rule still resolves correctly
because any one valid counterexample flips the answer to SAT.

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.build_ground_truth

Writes:
* ``examples/FlowConformal/experiments/exp1_vnncomp_subset/ground_truth.csv``
* ``examples/FlowConformal/experiments/exp2_prob_scale/ground_truth.csv``

Run-once and read by the aggregators. Re-run only if VNN-COMP 2025
results are updated upstream.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_HERE = Path(__file__).resolve().parent
_VNNCOMP_RESULTS = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_results'))

_SOUND_VERIFIERS: Tuple[str, ...] = (
    'alpha_beta_crown', 'neuralsat', 'pyrat', 'nnv', 'nnenum',
    'cora', 'rover', 'sobolbox',
)

# Map our local benchmark names to the row-key VNN-COMP uses in
# ``<tool>/results.csv`` column 1.
_LOCAL_TO_VNNCOMP = {
    'acasxu_2023': 'acasxu_2023',
    'collins_rul_cnn_2022': 'collins_rul_cnn_2022',
    'dist_shift_2023': 'dist_shift_2023',
    'linearizenn_2024': 'linearizenn_2024',
    'tllverify_2023': 'tllverifybench_2023',
    'vit_2023': 'vit_2023',
    'malbeware': 'malbeware',
    'metaroom_2023': 'metaroom_2023',
    'tinyimagenet_2024': 'tinyimagenet_2024',
    'cifar100_2024': 'cifar100_2024',
    # New multi-output benches (smoke-decided). VNN-COMP 2025 has
    # at least 3 sound verifiers (αβ-CROWN, PyRAT, ROVER) reporting
    # on each — enough for the SAT-wins consensus.
    'lsnc_relu': 'lsnc_relu',
    'relusplitter': 'relusplitter',
}

# Per-experiment benchmarks that need ground truth. cifar10_resnet110
# is intentionally absent — it's not a VNN-COMP benchmark.
_EXP1_BENCHMARKS: Tuple[str, ...] = (
    'acasxu_2023', 'collins_rul_cnn_2022', 'dist_shift_2023',
    'linearizenn_2024', 'tllverify_2023',
    'malbeware', 'metaroom_2023',
    'lsnc_relu', 'relusplitter',
)

_EXP2_BENCHMARKS: Tuple[str, ...] = (
    'vit_2023', 'tinyimagenet_2024', 'cifar100_2024',
)

_OUTPUT_FIELDS = [
    'benchmark', 'onnx_file', 'vnnlib_file', 'ground_truth',
    'n_sat', 'n_unsat', 'n_timeout', 'n_unknown', 'n_error',
    'dissenting_tools', 'source_tools',
]


def _scan_tool_results(
    tool: str, vnncomp_keys: Set[str],
) -> Dict[Tuple[str, str, str], str]:
    """Return ``{(vnncomp_key, onnx_basename, vnnlib_basename): verdict}``
    for the rows whose first column is one of ``vnncomp_keys``.
    """
    out: Dict[Tuple[str, str, str], str] = {}
    csv_path = _VNNCOMP_RESULTS / tool / 'results.csv'
    if not csv_path.exists():
        return out
    with open(csv_path, newline='') as f:
        for r in csv.reader(f):
            if len(r) < 5:
                continue
            key = r[0]
            if key not in vnncomp_keys:
                continue
            onnx_name = Path(r[1]).name
            vnn_name = Path(r[2]).name
            verdict = r[4].strip().lower().split(',')[0].strip()
            out[(key, onnx_name, vnn_name)] = verdict
    return out


def _resolve_ground_truth(
    instance_verdicts: Dict[str, str],
) -> Tuple[str, List[str], List[str]]:
    """Apply SAT-wins with an αβ-CROWN soundness veto.

    The original SAT-wins rule was too permissive: it took any single
    tool's SAT verdict as ground truth, even when contradicted by
    αβ-CROWN's UNSAT proof. On the lsnc_relu benchmark this caused 68
    spurious SATs from ROVER to override αβ-CROWN's UNSAT proofs,
    yielding a wrong 80-SAT/1-UNSAT distribution instead of the
    αβ-CROWN-aligned 12-SAT/69-UNSAT.

    Revised rule:

    * ``ground_truth = 'sat'`` iff ≥1 tool says SAT AND αβ-CROWN does
      NOT say UNSAT for this instance. (αβ-CROWN is the de-facto sound
      verifier; its UNSAT proof vetoes any other tool's un-validated
      SAT claim.)
    * ``ground_truth = 'unsat'`` iff at least one tool says UNSAT AND
      the SAT criterion above is not met.
    * ``ground_truth = 'unknown'`` otherwise.

    Returns ``(ground_truth, dissenting_tools, source_tools)``.

    * ``dissenting_tools``: when GT='sat', the tools that returned
      UNSAT (still useful for soundness audits even though they don't
      change the verdict). When GT='unsat', the tools whose un-validated
      SAT was vetoed by αβ-CROWN's UNSAT (the new failure-mode this
      rule catches).
    """
    sat_tools = sorted(t for t, v in instance_verdicts.items() if v == 'sat')
    unsat_tools = sorted(t for t, v in instance_verdicts.items() if v == 'unsat')
    abcrown_says_unsat = (instance_verdicts.get('alpha_beta_crown') == 'unsat')

    if sat_tools and not abcrown_says_unsat:
        return 'sat', unsat_tools, sat_tools + unsat_tools
    if unsat_tools:
        # If αβ-CROWN says UNSAT but some other tool said SAT, those
        # SATs are vetoed — record them as dissenters for audit.
        if abcrown_says_unsat and sat_tools:
            return 'unsat', sat_tools, unsat_tools
        return 'unsat', [], unsat_tools
    return 'unknown', [], []


def _count_verdicts(verdicts: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for v in verdicts.values():
        # Normalise unusual verdicts (e.g. 'sat,12.3') into the canonical
        # vocabulary; anything else lands in 'error'.
        if v in ('sat', 'unsat', 'timeout', 'unknown', 'error'):
            counts[v] += 1
        else:
            counts['error'] += 1
    return counts


def build_for_benchmarks(
    benchmarks: Tuple[str, ...],
    output_path: Path,
) -> int:
    """Write a ground-truth CSV for the given benchmarks. Returns row count."""
    vnncomp_keys = {_LOCAL_TO_VNNCOMP[b] for b in benchmarks}
    reverse_map = {_LOCAL_TO_VNNCOMP[b]: b for b in benchmarks}

    # Collect per-instance verdicts across tools.
    # key: (vnncomp_key, onnx_name, vnn_name) -> {tool -> verdict}
    per_instance: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    for tool in _SOUND_VERIFIERS:
        tool_verdicts = _scan_tool_results(tool, vnncomp_keys)
        for inst_key, verdict in tool_verdicts.items():
            per_instance[inst_key][tool] = verdict

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS)
        w.writeheader()
        for (vnncomp_key, onnx_name, vnn_name) in sorted(per_instance.keys()):
            verdicts = per_instance[(vnncomp_key, onnx_name, vnn_name)]
            counts = _count_verdicts(verdicts)
            ground_truth, dissenters, sources = _resolve_ground_truth(verdicts)
            w.writerow({
                'benchmark': reverse_map[vnncomp_key],
                'onnx_file': onnx_name,
                'vnnlib_file': vnn_name,
                'ground_truth': ground_truth,
                'n_sat': counts.get('sat', 0),
                'n_unsat': counts.get('unsat', 0),
                'n_timeout': counts.get('timeout', 0),
                'n_unknown': counts.get('unknown', 0),
                'n_error': counts.get('error', 0),
                'dissenting_tools': ','.join(dissenters),
                'source_tools': ','.join(sources),
            })
            n_rows += 1
    return n_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--exp1-out',
        type=Path,
        default=_HERE / 'exp1_vnncomp_subset' / 'ground_truth.csv',
    )
    p.add_argument(
        '--exp2-out',
        type=Path,
        default=_HERE / 'exp2_prob_scale' / 'ground_truth.csv',
    )
    args = p.parse_args()

    n1 = build_for_benchmarks(_EXP1_BENCHMARKS, args.exp1_out)
    print(f'Wrote {n1} rows to {args.exp1_out}')
    n2 = build_for_benchmarks(_EXP2_BENCHMARKS, args.exp2_out)
    print(f'Wrote {n2} rows to {args.exp2_out}')


if __name__ == '__main__':
    main()
