"""Per-(experiment, tool) runner shim for Exp 1.

Provides a single uniform interface that every Exp 1 runner script
(``exp1_run_<tool>.py``) calls, with the benchmark name passed via
``--benchmark``:

    from .._benchmarks import (
        BENCHMARK_ROOTS,
        PER_BENCHMARK_CONFIG,
        load_one_instance,
        parse_instances_csv,
    )

The loader dispatches on benchmark name: ACAS Xu uses its dedicated
``_ACASXuWrapper`` (input normalisation baked in); all other Exp 1
benchmarks use the generic ``_GenericONNXWrapper`` from
:mod:`._common`. Both paths return ``(network, boxes, spec)`` with
identical contract.

The ``PER_BENCHMARK_CONFIG`` dict carries the lock-probe-validated
hparam overrides per benchmark (n_train, flow_epochs, scenario_n,
verification_method, amls_max_levels, etc.). The ``malbeware`` and
``metaroom_2023`` entries default to ``mega`` with
``verification_method='amls_bounded_union'`` and rely on a
5-instance lock-probe run for validation before the full sweep.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ._common import load_instance as _generic_load_instance
from ._common import parse_instances_csv  # re-export


# ----- benchmark roots (VNN-COMP 2025) -----

# Mirror of ``baselines._common.VNNCOMP_BENCHMARK_ROOTS`` for the 6
# Exp 1 benchmarks. Kept in this module so the runner doesn't depend
# on the baselines package.
BENCHMARK_ROOTS: Dict[str, Path] = {
    'acasxu_2023': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023')),
    'collins_rul_cnn_2022': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/collins_rul_cnn_2022')),
    'dist_shift_2023': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/dist_shift_2023')),
    'linearizenn_2024': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/linearizenn_2024')),
    'tllverify_2023': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/tllverifybench_2023')),
    'malbeware': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/malbeware')),
    'metaroom_2023': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/metaroom_2023')),
    # Multi-output candidates added for the geometry-advantage story
    # (small-output-dim controllers + MNIST classifier). Smoke-tested
    # before being committed to the full sweep.
    'lsnc_relu': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/lsnc_relu')),
    'relusplitter': Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/relusplitter')),
}

EXP1_BENCHMARKS: Tuple[str, ...] = tuple(BENCHMARK_ROOTS.keys())


# ----- per-benchmark hparam overrides (locked by probe_amls_bounded_lock) -----

# All five non-vit benchmarks below have ``cell_status == 'ok'`` for 5/5
# instances at the listed config in
# ``examples/FlowConformal/probes/outputs/probe_amls_bounded_lock.csv``.
# vit_2023 needs ``amls_max_levels=5`` per ``probes/diag_max_levels.py``;
# its `--smoke` is the canonical validation before a full sweep.
#
# Smoke-checked expected verdicts mirror VNN-COMP 2025 ground truth on
# each benchmark's first ``instances.csv`` row.
# Falsifier configuration is driven per-benchmark.
#   ``use_falsifier``         : bool, gate Stage-1 falsification on/off.
#   ``falsifier_method``      : passed as ``sat_backend`` to
#                               ``run_verification_pipeline``. We use
#                               'apgd' (Auto-PGD; Croce-Hein 2020) because
#                               it subsumes vanilla PGD at equal compute
#                               and dropping the legacy 'random+pgd+apgd'
#                               cascade saves ~2/3 of falsifier wall.
#   ``falsifier_n_restarts``  : independent random inits per call.
#   ``falsifier_n_steps``     : APGD steps per restart.
#
# Tuned by ``probes/probe_falsifier_budget.py``: for each Exp 1
# benchmark, K=3 SAT-by-VNN-COMP-consensus instances were probed at
# an escalating ``(n_restarts, n_steps)`` grid. Recommendation is the
# smallest budget that captures every APGD-findable CEX (instances
# that fail at (3, 25) also fail at (20, 200) — APGD-resistant SATs;
# more compute does not help and the AMLS in-coverage detection
# handles them downstream). All Exp 1 benchmarks land at (3, 25)
# except dist_shift_2023, which needs (5, 50) for one of the probed
# instances. Per-instance falsifier wall on UNSAT instances: ~0.1-0.4s
# (down from ~47s with the legacy 30x200 cascade).
PER_BENCHMARK_CONFIG: Dict[str, Dict[str, Any]] = {
    'acasxu_2023': dict(
        flow_config='base',
        n_train=5_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded',
        amls_max_levels=30,
        use_falsifier=True,  # Stage-1 falsifier ON for VNN-COMP comparability (parity with sound-verifier cex extraction).
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        smoke_first_instance='ACASXU_run2a_1_1_batch_2000.onnx + prop_1.vnnlib',
    ),
    'collins_rul_cnn_2022': dict(
        flow_config='base',
        n_train=1_000,
        flow_epochs=1_000,
        scenario_n_samples=500,
        alpha=0.001,
        verification_method='amls_bounded',
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        smoke_first_instance=None,  # filled in by parse_instances_csv
    ),
    'dist_shift_2023': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded',
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=5,    # probe-tuned: one SAT-by-consensus instance needs (5, 50); (3, 25) misses it.
        falsifier_n_steps=50,
        smoke_first_instance=None,
    ),
    'linearizenn_2024': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded',
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        smoke_first_instance=None,
    ),
    'tllverify_2023': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded',
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        smoke_first_instance=None,
    ),
    'malbeware': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded_union',  # 24-disjunct cls spec (cora-style nested OR)
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        smoke_first_instance=None,
    ),
    'metaroom_2023': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded_union',  # 19-disjunct cls spec (cora-style nested OR)
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        smoke_first_instance=None,
    ),
    # ── New benchmarks added for the multi-output geometry story ──
    # Smoke-tested before being committed to the full sweep. Configs
    # mirror nearest established benchmark family:
    #   lsnc_relu  ← small-input control (like ACAS Xu); n_train=5K
    #   relusplitter  ← MNIST classifier with 9 disjuncts; n_train=10K
    'lsnc_relu': dict(
        flow_config='base',
        # Budget-fitted via probe: 5K/2K/1K → 25.4s on instance 0
        # (UNSAT verdict, eps_2=1.5e-4). Cuts scenario_n_samples from
        # 2K to 1K to fit the VNN-COMP 25s per-instance budget.
        n_train=5_000,
        flow_epochs=2_000,
        scenario_n_samples=1_000,
        alpha=0.001,
        verification_method='amls_bounded_union',  # multi-disjunct policy spec
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        # Hashemi-clipping m, used by exp1_run_hashemi_clipping. Reduced
        # from default 8000 → 4000 to fit the 25s VNN-COMP budget
        # (m=4000 → 24.6s, ~98% of budget).
        hashemi_m=4_000,
        smoke_first_instance=None,
    ),
    'relusplitter': dict(
        flow_config='base',
        # Budget-fitted via probe: 5K/1K/1K → 23.9s on instance 0
        # (UNSAT verdict, eps_2 at floor). Cuts both flow_epochs and
        # scenario_n_samples from defaults to fit the 30s VNN-COMP
        # per-instance budget.
        n_train=5_000,
        flow_epochs=1_000,
        scenario_n_samples=1_000,
        alpha=0.001,
        verification_method='amls_bounded_union',  # MNIST 10-class
        amls_max_levels=30,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
        # Hashemi-clipping m reduced from 8000 → 4000 to fit the 30s
        # budget (m=4000 → 28.3s).
        hashemi_m=4_000,
        smoke_first_instance=None,
    ),
}


# ----- per-benchmark loader dispatch -----

def _load_acasxu_instance(root: Path, onnx_rel: str, vnn_rel: str):
    """Use the ACAS Xu wrapper that bakes in input normalisation."""
    from examples.FlowConformal.benchmarks.demo_acasxu_single import (
        _ACASXuWrapper, _extract_spec,
    )
    from n2v.utils import load_vnnlib
    from n2v.utils.model_loader import load_onnx

    # ACAS Xu ONNX files live under ``examples/ACASXu/`` (a sibling of
    # ``examples/FlowConformal/``), not under the VNN-COMP benchmark
    # root — that root only ships the vnnlib specs and instances.csv.
    # ``_benchmarks.py`` is at examples/FlowConformal/experiments/
    # exp1_vnncomp_subset/_benchmarks.py, so ``parents[3]`` is examples/.
    acasxu_root = (Path(__file__).resolve().parents[3] / 'ACASXu')
    onnx_path = acasxu_root / onnx_rel.removeprefix('./')
    vnn_path = acasxu_root / vnn_rel.removeprefix('./')
    network = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())

    prop = load_vnnlib(str(vnn_path))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        lbs, ubs = prop['lb'], prop['ub']
        boxes = [(np.asarray(lb).flatten(), np.asarray(ub).flatten())
                 for lb, ub in zip(lbs, ubs)]
    else:
        boxes = [(np.asarray(prop['lb']).flatten(),
                  np.asarray(prop['ub']).flatten())]
    spec = _extract_spec(prop['prop'])
    return network, boxes, spec


def load_one_instance(benchmark: str, onnx_rel: str, vnn_rel: str):
    """Uniform loader: ``(network, boxes, spec)`` for one instance.

    Dispatches on benchmark name. ACAS Xu gets its dedicated wrapper;
    all others go through the generic ``_GenericONNXWrapper`` from
    :mod:`._common`.

    Raises:
        KeyError: ``benchmark`` not in :data:`EXP1_BENCHMARKS`.
        FileNotFoundError: ONNX or vnnlib path missing.
        NotImplementedError: spec shape unsupported (caller should
            translate this to ``verdict='SKIPPED'``).
    """
    if benchmark not in BENCHMARK_ROOTS:
        raise KeyError(
            f'unknown Exp 1 benchmark: {benchmark}; '
            f'expected one of {EXP1_BENCHMARKS}')
    root = BENCHMARK_ROOTS[benchmark]
    if benchmark == 'acasxu_2023':
        return _load_acasxu_instance(root, onnx_rel, vnn_rel)
    return _generic_load_instance(root, onnx_rel, vnn_rel)


def list_instances(benchmark: str) -> List[Tuple[str, str, int]]:
    """Return the parsed ``instances.csv`` rows for a benchmark.

    Each row is ``(onnx_rel, vnn_rel, vnncomp_timeout_s)``.
    """
    if benchmark not in BENCHMARK_ROOTS:
        raise KeyError(
            f'unknown Exp 1 benchmark: {benchmark}; '
            f'expected one of {EXP1_BENCHMARKS}')
    root = BENCHMARK_ROOTS[benchmark]
    csv_path = root / 'instances.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'instances.csv missing at {csv_path}')
    return parse_instances_csv(csv_path)
