"""Per-(experiment, tool) runner shim for Exp 2.

Same idea as the Exp 1 shim: factor out the per-benchmark config and
loader dispatch so each runner script (``exp2_run_<tool>.py``) stays
small and just calls ``list_instances(benchmark)`` /
``load_one_instance(benchmark, ...)``.

Exp 2 benchmark mix:

* ``vit_2023`` — VNN-COMP 2023, 76K params, 9-class spec, max_levels=5
  override (validated by ``probes/diag_max_levels.py``).
* ``tinyimagenet_2024`` — VNN-COMP 2024, ResNet-medium, 200-class
  disjunctive spec, ``mega`` config (n_train=10K, flow_epochs=2K,
  scenario_n=2K) with ``verification_method='amls_bounded_union'``
  to fold the 199 other-class disjuncts into a single AMLS chain.
* ``cifar100_2024`` — VNN-COMP 2024, 2.5M-param ResNet-medium with
  batchnorm, 99 other-class disjuncts → ``verification_method='amls_bounded_union'``
  (single chain on ``phi_union(y) = min_k phi_k(y)`` instead of 99
  per-halfspace chains). Not in the lock probe — runner ``--smoke``
  is the validation gate.
* ``cifar10_resnet110`` — Cohen RS 110-layer ResNet (1.7M params), L∞
  perturbation around CIFAR-10 test images, 300s/instance (no VNN-COMP
  per-row timeout for this benchmark — locked by Exp 2 design doc).
  Sound verifiers (αβ-CROWN/NeuralSAT) reportedly don't scale to
  this depth; runner reports whatever they output.

For the αβ-CROWN runner we use the VNN-COMP-format benchmarks (vit,
tinyimagenet, cifar100) plus cifar10_resnet110 via locally-generated
ONNX+vnnlib (see ``build_resnet110_onnx.py``). RS only applies to
image-classification specs (cifar10, cifar100).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# We re-use the deferred-evaluation loader pattern from the existing
# ``baselines/_common.py``: each entry is ``(name, loader_fn, timeout_s)``
# where ``loader_fn()`` returns ``(network, boxes, spec, name)``.
from examples.FlowConformal.experiments.baselines._common import (
    load_benchmark_instances as _baseline_load_benchmark_instances,
)

# ----- benchmark roots (VNN-COMP 2025) -----

VNNCOMP_BENCHMARKS: Tuple[str, ...] = (
    'vit_2023', 'tinyimagenet_2024', 'cifar100_2024',
)
IMAGE_BENCHMARKS: Tuple[str, ...] = (
    'cifar10_resnet110',
)
EXP2_BENCHMARKS: Tuple[str, ...] = VNNCOMP_BENCHMARKS + IMAGE_BENCHMARKS

# Subset of benchmarks that have an on-disk ONNX+vnnlib pair (so
# αβ-CROWN / NeuralSAT can ingest them via subprocess + CLI).
# cifar10_resnet110 is included here once
# ``examples/FlowConformal/experiments/exp2_prob_scale/build_resnet110_onnx.py``
# has generated the local artifacts (ONNX + 100 vnnlib + instances.csv);
# the runner skips it gracefully if those artifacts are missing.
EXP2_VNNCOMP_FORMAT: Tuple[str, ...] = (
    VNNCOMP_BENCHMARKS + ('cifar10_resnet110',)
)

# Subset that's image-classification with logit outputs (so Cohen RS
# applies). cifar100_2024's network IS a CIFAR-100 classifier, so RS
# can be applied via the same Smooth wrapper.
EXP2_RS_APPLICABLE: Tuple[str, ...] = ('cifar10_resnet110', 'cifar100_2024')


# ----- per-benchmark hparam + timeout overrides -----

# Falsifier ON for all Exp 2 benchmarks: 3 of 4 are VNN-COMP benchmarks
# (vit_2023, tinyimagenet_2024, cifar100_2024) where extracting a CEX
# matches sound-verifier behaviour, and it makes sense to keep the same
# pre-verify falsification step on cifar10_resnet110 too. Both ours and
# Hashemi run the same APGD with the same per-benchmark
# (n_restarts, n_steps) budget — see ``exp1_vnncomp_subset/_benchmarks.py``
# for the symmetric Exp 1 setup.
PER_BENCHMARK_CONFIG: Dict[str, Dict[str, Any]] = {
    'vit_2023': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded_union',  # 9-disjunct cls spec
        amls_max_levels=30,
        timeout_policy='vnncomp_per_row',
        default_timeout_s=100,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
    ),
    'tinyimagenet_2024': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded_union',  # 199-disjunct cls spec
        amls_max_levels=30,
        timeout_policy='fixed',
        default_timeout_s=100,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
    ),
    'cifar100_2024': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded_union',  # 99 disjuncts → single chain
        amls_max_levels=30,
        timeout_policy='fixed',
        default_timeout_s=100,         # matches VNN-COMP 2024
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
    ),
    'cifar10_resnet110': dict(
        flow_config='base',
        n_train=10_000,
        flow_epochs=2_000,
        scenario_n_samples=2_000,
        alpha=0.001,
        verification_method='amls_bounded',
        amls_max_levels=30,
        timeout_policy='fixed',
        default_timeout_s=300,
        use_falsifier=True,
        falsifier_method='apgd',
        falsifier_n_restarts=3,
        falsifier_n_steps=25,
    ),
}


# ----- benchmark loader dispatch -----

def list_instances(
    benchmark: str,
    n: int = 100,
) -> List[Tuple[str, Callable, int]]:
    """Return up to ``n`` instances for the named Exp 2 benchmark.

    Each entry is a ``(name, loader_fn, timeout_s)`` triple. The
    loader is deferred — call ``loader_fn()`` to get
    ``(network, boxes, spec, name)``. Useful for image classification
    benchmarks where loading the pretrained network and sampling
    test images is expensive enough to want to amortise across the
    sweep but defer per-instance work to the per-instance call.

    The returned ``timeout_s`` is the per-instance timeout to use:
    for ``timeout_policy='vnncomp_per_row'`` it's the value from
    ``instances.csv`` column 3; for ``timeout_policy='fixed'`` it's
    the ``default_timeout_s`` from :data:`PER_BENCHMARK_CONFIG`.
    """
    if benchmark not in EXP2_BENCHMARKS:
        raise KeyError(
            f'unknown Exp 2 benchmark: {benchmark}; '
            f'expected one of {EXP2_BENCHMARKS}')
    cfg = PER_BENCHMARK_CONFIG[benchmark]

    raw = _baseline_load_benchmark_instances(benchmark, n)

    # Override timeouts when the benchmark has a fixed-timeout policy
    # (cifar10_resnet110, cifar100_2024, tinyimagenet_2024); pass
    # through VNN-COMP per-row values otherwise (vit_2023).
    if cfg['timeout_policy'] == 'fixed':
        out = [(name, loader, cfg['default_timeout_s'])
               for (name, loader, _t) in raw]
    else:
        out = [(name, loader, t if t > 0 else cfg['default_timeout_s'])
               for (name, loader, t) in raw]
    return out


def load_one_instance(benchmark: str, loader: Callable):
    """Materialise an instance from its deferred loader.

    Returns ``(network, boxes, spec, name)``. Raises whatever the
    underlying loader raises (typically ``FileNotFoundError`` when
    pretrained weights are missing).
    """
    if benchmark not in EXP2_BENCHMARKS:
        raise KeyError(
            f'unknown Exp 2 benchmark: {benchmark}; '
            f'expected one of {EXP2_BENCHMARKS}')
    return loader()


# ----- VNN-COMP path lookup (for sound-verifier runners) -----

_VNNCOMP_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks'))

# cifar10_resnet110's "VNN-COMP path" is locally generated by
# ``build_resnet110_onnx.py`` (no upstream VNN-COMP equivalent).
_LOCAL_RESNET110_ROOT = (
    Path(__file__).resolve().parent / 'cifar10_resnet110_vnncomp'
)

_VNNCOMP_BENCH_DIR = {
    'vit_2023': _VNNCOMP_ROOT / 'vit_2023',
    'tinyimagenet_2024': _VNNCOMP_ROOT / 'tinyimagenet_2024',
    'cifar100_2024': _VNNCOMP_ROOT / 'cifar100_2024',
    'cifar10_resnet110': _LOCAL_RESNET110_ROOT,
}


def vnncomp_paths(benchmark: str) -> Path:
    """Root directory of VNN-COMP files for ``benchmark``.

    Used by αβ-CROWN/NeuralSAT runners to resolve absolute ``.onnx`` /
    ``.vnnlib`` paths from the relative paths in ``instances.csv``.
    """
    if benchmark not in _VNNCOMP_BENCH_DIR:
        raise KeyError(
            f'no VNN-COMP path for benchmark {benchmark}; '
            f'sound verifiers can only run on {EXP2_VNNCOMP_FORMAT}')
    return _VNNCOMP_BENCH_DIR[benchmark]


def list_vnncomp_format_instances(
    benchmark: str,
    n: int = 100,
) -> List[Tuple[Path, Path, int]]:
    """Return absolute ``(onnx_path, vnnlib_path, timeout_s)`` triples
    for a VNN-COMP-format benchmark.

    Used by ``exp2_run_alpha_beta_crown.py`` / ``exp2_run_neuralsat.py``
    runners that consume ONNX + vnnlib files via subprocess. Only
    valid for benchmarks in :data:`EXP2_VNNCOMP_FORMAT`.

    Per-instance timeout follows :data:`PER_BENCHMARK_CONFIG`'s
    ``timeout_policy`` — VNN-COMP per-row for vit/yolo, fixed for
    cifar100.
    """
    if benchmark not in EXP2_VNNCOMP_FORMAT:
        raise KeyError(
            f'list_vnncomp_format_instances only valid for '
            f'{EXP2_VNNCOMP_FORMAT}; got {benchmark}')
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
        parse_instances_csv,
    )
    cfg = PER_BENCHMARK_CONFIG[benchmark]
    root = vnncomp_paths(benchmark)
    rows = parse_instances_csv(root / 'instances.csv')[:n]
    out = []
    for (onnx_rel, vnn_rel, vnncomp_t) in rows:
        # Resolve to absolute paths; strip the ``./`` prefix some
        # benchmarks ship with.
        onnx_path = (root / onnx_rel.lstrip('./').lstrip('/')).resolve()
        vnn_path = (root / vnn_rel.lstrip('./').lstrip('/')).resolve()
        timeout_s = (
            cfg['default_timeout_s'] if cfg['timeout_policy'] == 'fixed'
            else (vnncomp_t if vnncomp_t > 0 else cfg['default_timeout_s'])
        )
        out.append((onnx_path, vnn_path, timeout_s))
    return out
