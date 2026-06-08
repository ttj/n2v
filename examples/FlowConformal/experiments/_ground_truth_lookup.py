"""Runtime ground-truth lookup for smoke summaries.

Reads the per-experiment ``ground_truth.csv`` (produced by
:mod:`examples.FlowConformal.experiments.build_ground_truth`) and
returns the SAT-wins consensus verdict for a given benchmark+instance
pair. Used by the per-runner smoke-summary blocks so that the
``[smoke] PASS`` line reports the actual ground truth from the
VNN-COMP 2025 sound-verifier consensus rather than a hardcoded
``smoke_expected_verdict='UNSAT'`` assumption that's wrong on
benchmarks whose first instance happens to be SAT (collins_rul_cnn_2022,
linearizenn_2024 — discovered 2026-04-30).

For benchmarks without a VNN-COMP entry (currently only
``cifar10_resnet110`` — Cohen pretrained ResNet-110, locally generated
ONNX+vnnlib) the lookup returns ``'NOT_APPLICABLE'``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

_HERE = Path(__file__).resolve().parent

_GT_CSV_BY_EXPERIMENT = {
    'exp1': _HERE / 'exp1_vnncomp_subset' / 'ground_truth.csv',
    'exp2': _HERE / 'exp2_prob_scale' / 'ground_truth.csv',
}

# Benchmarks for which no VNN-COMP ground truth exists (locally
# generated specs or outside the VNN-COMP 2025 benchmark set).
_NOT_APPLICABLE_BENCHMARKS = frozenset({'cifar10_resnet110'})


def _split_instance_name(instance_name: str) -> Tuple[str, str]:
    """Split an Exp-style ``instance_name`` into (onnx_basename, vnnlib_basename).

    Conventions in our runners:

    * Exp 1 / Exp 2 VNN-COMP-format runners write ``onnx+vnnlib`` joined
      by ``+`` (e.g. ``NN_rul_small_window_20.onnx+robustness_2perturbations_delta5_epsilon10_w20.vnnlib``).
    * cifar10_resnet110 instances are bare labels (e.g. ``cifar10_test_0000_label_3``).
    """
    if '+' in instance_name:
        onnx, vnn = instance_name.split('+', 1)
        return onnx.strip(), vnn.strip()
    return instance_name.strip(), instance_name.strip()


def lookup_ground_truth(
    experiment: str,
    benchmark: str,
    instance_name: str,
) -> str:
    """Return the ground-truth verdict for a smoke instance.

    Args:
        experiment: ``'exp1'`` or ``'exp2'`` — selects which
            ``ground_truth.csv`` to read.
        benchmark: e.g. ``'collins_rul_cnn_2022'``.
        instance_name: e.g. ``'NN_rul_small_window_20.onnx+robustness_2perturbations_delta5_epsilon10_w20.vnnlib'``.

    Returns:
        Upper-cased verdict (``'UNSAT'``, ``'SAT'``, ``'UNKNOWN'``).
        For benchmarks without a VNN-COMP entry (e.g. ``cifar10_resnet110``)
        returns ``'NOT_APPLICABLE'``.
        If the CSV exists but the specific instance isn't found,
        returns ``'NOT_FOUND'``.
    """
    if benchmark in _NOT_APPLICABLE_BENCHMARKS:
        return 'NOT_APPLICABLE'

    csv_path = _GT_CSV_BY_EXPERIMENT.get(experiment)
    if csv_path is None or not csv_path.exists():
        return 'NOT_FOUND'

    onnx, vnn = _split_instance_name(instance_name)

    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            if r['benchmark'].strip() != benchmark:
                continue
            if (r['onnx_file'].strip() == onnx
                    and r['vnnlib_file'].strip() == vnn):
                return r['ground_truth'].strip().upper()
    return 'NOT_FOUND'
