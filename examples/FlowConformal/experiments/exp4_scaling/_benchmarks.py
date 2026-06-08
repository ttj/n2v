"""Per-experiment benchmark loader for Exp 4.

Mirrors the role that ``examples/FlowConformal/benchmarks/`` plays for
VNN-COMP benchmarks: gives each per-tool runner a uniform interface for
listing instances, fetching the network, and locating the on-disk ONNX
+ vnnlib artifacts for sound-verifier tools.

The synthetic networks live in ``networks_onnx/`` and the per-instance
vnnlib files in ``vnnlib/`` — both relative to this experiment dir.
Both are generated lazily on first access (idempotent: re-runs return
identical files because the seeds are deterministic).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .instance_generator import (
    EXP4_INSTANCES_PER_DEPTH,
    generate_instance,
    write_vnnlib,
)
from .networks import (
    EXP4_DEPTHS,
    EXP4_WIDTH,
    SyntheticMLP,
    export_onnx,
    make_synthetic_network,
    n_params,
)

_THIS_DIR = Path(__file__).resolve().parent
_NETWORKS_DIR = _THIS_DIR / 'networks_onnx'
_VNNLIB_DIR = _THIS_DIR / 'vnnlib'


def onnx_path(depth: int) -> Path:
    return _NETWORKS_DIR / f'synth_d{depth}_w{EXP4_WIDTH}.onnx'


def vnnlib_path(depth: int, instance_idx: int) -> Path:
    return _VNNLIB_DIR / f'synth_d{depth}_w{EXP4_WIDTH}_i{instance_idx}.vnnlib'


def get_network(depth: int, *, force_regenerate: bool = False) -> tuple[SyntheticMLP, Path]:
    """Build (or reuse cached) synthetic network for ``depth``.

    Always returns the in-Python ``net`` object plus the on-disk ONNX
    path. The ONNX is written if absent or if ``force_regenerate=True``.
    """
    p = onnx_path(depth)
    net = make_synthetic_network(depth)
    if force_regenerate or not p.exists():
        _NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
        export_onnx(net, p)
    return net, p


def get_instance(
    depth: int,
    instance_idx: int,
    *,
    net: SyntheticMLP | None = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Build (or reuse cached) instance ``(depth, instance_idx)``.

    Returns a dict that includes every artifact a runner might need:
    ``net``, ``onnx_path``, ``vnnlib_path``, ``lb``, ``ub``,
    ``spec_halfspace``, plus diagnostic fields (``x_0``, ``C``, etc.).

    The ONNX file honors ``force_regenerate`` (architecture/weights are
    fixed by seed, so caching is safe). The **vnnlib file is always
    rewritten** because the spec threshold ``C`` can shift when the
    construction changes (e.g. the empirical-vs-Lipschitz C upgrade);
    a stale vnnlib on disk would silently feed the OLD spec to
    sound-verifier subprocess runners (αβ-CROWN, NeuralSAT) while the
    in-memory ``spec_halfspace`` carries the NEW spec — a hard-to-spot
    correctness bug. Cost is sub-millisecond per file, so we always
    regenerate.
    """
    if net is None:
        net, _ = get_network(depth)
    inst = generate_instance(net, depth, instance_idx)

    vp = vnnlib_path(depth, instance_idx)
    write_vnnlib(vp, inst['lb'], inst['ub'], inst['C'])
    inst['vnnlib_path'] = vp
    inst['onnx_path'] = onnx_path(depth)
    inst['net'] = net
    return inst


def load_instances(depth: int) -> List[Dict[str, Any]]:
    """List all 10 instances for a depth, generating files lazily.

    Each returned dict has all fields :func:`get_instance` returns plus
    ``ground_truth='unsat'`` (UNSAT-by-construction), the depth,
    instance index, and parameter count.
    """
    if depth not in EXP4_DEPTHS:
        raise ValueError(
            f'depth must be one of EXP4_DEPTHS={EXP4_DEPTHS}, got {depth}')
    net, _ = get_network(depth)
    instances: List[Dict[str, Any]] = []
    for idx in range(EXP4_INSTANCES_PER_DEPTH):
        inst = get_instance(depth, idx, net=net)
        inst['ground_truth'] = 'unsat'
        inst['n_params'] = n_params(depth)
        instances.append(inst)
    return instances
