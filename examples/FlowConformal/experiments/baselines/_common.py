"""Shared utilities for probabilistic-baseline runners.

Each baseline runner produces CSV rows with the following schema::

    benchmark, instance, baseline, verdict, wall_s, error,
    <baseline-specific fields...>

Verdicts:
    UNSAT          — proved (probabilistically) safe within the tool's
                     guarantee (no counterexample, certified bound rules
                     out the unsafe halfspace).
    SAT            — counterexample / unsafe with probability >= 1-α.
    UNKNOWN        — tool ran but cannot certify either direction at the
                     requested probability.
    ERROR          — tool crashed or pre-conditions failed.
    NOT_APPLICABLE — tool fundamentally does not support this benchmark
                     (e.g. ProbStar on a non-piecewise-linear network,
                     RS on a non-classification benchmark).

The runners DO NOT integrate into ``n2v.probabilistic`` — they call
``~/v/other/smoothing``, ``~/v/other/SaVer-Toolbox``, ``~/v/other/StarV``
directly via ``sys.path`` injection. Any tool import failure is caught
and surfaced as ``verdict='ERROR'`` for that row.

Loader helpers: we re-use the Exp 1 / Exp 2 loaders (in
``experiments.exp1_vnncomp_subset._common`` and
``experiments.exp2_prob_scale._common``) so each baseline sees the same
``(network, boxes, spec, name)`` tuples our pipeline does.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

# Columns common to every baseline runner; each runner extends this with
# its own tool-specific fields (e.g. ``radius`` for RS, ``n_samples`` for
# SaVer/Hashemi, etc.).
COMMON_FIELDS = [
    'benchmark', 'instance', 'baseline', 'verdict',
    'wall_s', 'error',
]


def open_csv_writer(out_csv: Path, extra_fields: Iterable[str]):
    """Open ``out_csv`` for writing, write the header, return the
    ``csv.DictWriter`` and the file handle. Caller closes the handle.
    """
    fields = list(COMMON_FIELDS) + list(extra_fields)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    f.flush()
    return writer, f, fields


def _fmt(v, spec):
    if v is None:
        return ''
    try:
        return f'{v:{spec}}'
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def halfspace_disjoint_from_box(spec, lb_y: np.ndarray,
                                ub_y: np.ndarray) -> Optional[bool]:
    """Check whether an output bounding box ``[lb_y, ub_y]`` is disjoint
    from the unsafe region defined by ``spec``.

    The box is "definitely safe" (returns True) if every disjunct in the
    spec has at least one row ``g_i^T y <= h_i`` that the box upper
    interval-evaluates strictly above ``h_i``. Returns False if any
    disjunct's row interval-bound never exceeds ``h_i``, meaning the
    box may intersect the unsafe region. Returns None if the spec
    structure is not recognised.

    Spec encodings handled (matches ``n2v.utils.load_vnnlib`` /
    ``run_verification_pipeline`` conventions):

    * ``HalfSpace``: a single AND-of-rows constraint ``G y <= g``
      defining ONE unsafe disjunct.
    * ``list[HalfSpace]``: OR-of-ANDs — multiple unsafe disjuncts; box
      is disjoint iff disjoint from every disjunct.
    * ``list[dict]`` with key ``'Hg'``: AND-of-OR groups (cora-style).
      Each group is a list of HalfSpace constraints; treated as multiple
      ORs that all need to be ruled out.

    For interval evaluation: for row vector ``g_i`` and box ``[lb, ub]``,
    the maximum of ``g_i^T y`` over the box is
    ``sum(max(g_i_j * lb_j, g_i_j * ub_j))``. If this max is strictly
    less than ``h_i``, then ``g_i^T y > h_i`` is impossible inside the
    box, so the unsafe disjunct (which requires ``g_i^T y <= h_i``) does
    NOT exclude the box; we need the OPPOSITE: the box should LIE
    OUTSIDE the unsafe halfspace, meaning ``g_i^T y > h_i`` everywhere
    in the box, i.e. ``min(g_i^T y) > h_i`` (where min is computed by
    summing ``min(g_i_j * lb_j, g_i_j * ub_j)``).

    Conventions reminder (per project memory ``project_sat_unsat_convention``):
    UNSAFE region encoded as ``G y <= g``; UNSAT = box disjoint from
    UNSAFE = there exists a row ``g_i`` of every unsafe disjunct s.t.
    ``g_i^T y > h_i`` for ALL y in the box.
    """
    from n2v.sets.halfspace import HalfSpace

    lb = np.asarray(lb_y, dtype=np.float64).flatten()
    ub = np.asarray(ub_y, dtype=np.float64).flatten()

    def _disjunct_excluded(hs: HalfSpace) -> bool:
        """Box ``[lb, ub]`` ∩ {y: G y <= g} = ∅ iff some row's interval
        min strictly exceeds the row rhs.
        """
        G = np.asarray(hs.G, dtype=np.float64)
        g = np.asarray(hs.g, dtype=np.float64).flatten()
        if G.shape[1] != lb.size:
            return False  # dim mismatch: be conservative
        # min over box of G y (per row): sum elementwise min of G_ij*lb_j
        # vs G_ij*ub_j.
        prod_lb = G * lb  # broadcasts over rows
        prod_ub = G * ub
        row_min = np.minimum(prod_lb, prod_ub).sum(axis=1)
        return bool(np.any(row_min > g + 1e-9))

    if isinstance(spec, HalfSpace):
        return _disjunct_excluded(spec)

    if isinstance(spec, list) and len(spec) > 0:
        first = spec[0]
        if isinstance(first, HalfSpace):
            # OR-of-ANDs — UNSAT iff every disjunct ruled out.
            return all(_disjunct_excluded(hs) for hs in spec)
        if isinstance(first, dict):
            # AND-of-OR groups: each group has 'Hg' field; the group
            # itself is one ANDed conjunction (single HalfSpace whose
            # rows are the AND-of-rows), and groups are OR-combined
            # (the unsafe set is the UNION over groups). UNSAT iff
            # every group's HalfSpace excludes the box.
            for group in spec:
                disjunct = group.get('Hg', None)
                if disjunct is None:
                    return None
                if isinstance(disjunct, HalfSpace):
                    if not _disjunct_excluded(disjunct):
                        return False
                elif isinstance(disjunct, list):
                    # rare: list of HalfSpace inside a group dict
                    if all(isinstance(d, HalfSpace) for d in disjunct):
                        if not all(_disjunct_excluded(d) for d in disjunct):
                            return False
                    else:
                        return None
                else:
                    return None
            return True

    return None


def extract_disjuncts_from_spec(spec):
    """Flatten a spec into a list of polytope disjuncts ``[(G_i, g_i), ...]``
    where the unsafe set is the OR ``U = \\/_i { y : G_i y <= g_i }`` and
    each disjunct is a single conjunctive polytope (potentially with
    multiple rows AND-ed together).

    Returns:
        list of ``(G, g)`` numpy float64 tuples, or ``None`` if the spec
        cannot be cast as a flat OR of polytopes (e.g. a true
        AND-of-OR-of-ANDs with multiple groups).

    Spec encodings handled (matches ``n2v.utils.load_vnnlib`` /
    ``run_verification_pipeline``):

    * ``HalfSpace`` (single conjunct): one polytope.
    * ``list[HalfSpace]`` (OR-of-ANDs): k polytopes, one per HalfSpace.
    * ``list[dict]`` of length 1 with key ``'Hg'``: unwrap; the inner
      value is either a single ``HalfSpace`` (one disjunct) or a
      ``list[HalfSpace]`` (k disjuncts).
    * ``list[dict]`` of length > 1 (multi-group AND-of-OR-of-ANDs):
      cannot be flattened to a single OR — returns ``None``.
    """
    from n2v.sets.halfspace import HalfSpace

    def _hs_pair(hs: HalfSpace):
        G = np.asarray(hs.G, dtype=np.float64)
        g = np.asarray(hs.g, dtype=np.float64).flatten()
        return G, g

    if isinstance(spec, HalfSpace):
        return [_hs_pair(spec)]

    if isinstance(spec, list):
        if len(spec) == 0:
            return None
        if all(isinstance(e, HalfSpace) for e in spec):
            return [_hs_pair(hs) for hs in spec]
        if len(spec) == 1 and isinstance(spec[0], dict) and 'Hg' in spec[0]:
            inner = spec[0]['Hg']
            if isinstance(inner, HalfSpace):
                return [_hs_pair(inner)]
            if (isinstance(inner, list) and len(inner) > 0
                    and all(isinstance(e, HalfSpace) for e in inner)):
                return [_hs_pair(hs) for hs in inner]
            return None
        # Multi-group AND-of-OR-of-ANDs: not a flat OR; can't combine
        # via Bonferroni without a product expansion (which can blow up
        # in size and isn't needed for current Exp 1/Exp 2 benchmarks).
        if len(spec) > 1 and all(isinstance(e, dict) for e in spec):
            return None

    return None


def halfspace_witness_from_samples(spec, ys: np.ndarray
                                   ) -> Optional[np.ndarray]:
    """Return the index of any sample in ``ys`` that lies in the unsafe
    region, or None if no sample does. Used to detect SAT outcomes from
    Monte Carlo–style baselines (RS, SaVer) that draw samples already.
    """
    from n2v.sets.halfspace import HalfSpace

    def _row_violates(hs: HalfSpace, y_batch: np.ndarray):
        # y is unsafe under this disjunct iff G y <= g for ALL rows.
        G = np.asarray(hs.G, dtype=np.float64)
        g = np.asarray(hs.g, dtype=np.float64).flatten()
        prod = y_batch @ G.T  # (n, n_rows)
        return np.all(prod <= g + 1e-9, axis=1)

    if isinstance(spec, HalfSpace):
        viol = _row_violates(spec, ys)
        idx = np.flatnonzero(viol)
        return idx[0] if idx.size > 0 else None

    if isinstance(spec, list) and len(spec) > 0 and isinstance(spec[0], HalfSpace):
        # OR of ANDs: any disjunct hit means SAT.
        for hs in spec:
            viol = _row_violates(hs, ys)
            idx = np.flatnonzero(viol)
            if idx.size > 0:
                return idx[0]
        return None

    if isinstance(spec, list) and len(spec) > 0 and isinstance(spec[0], dict):
        # AND-of-OR groups encoded as list[dict] with 'Hg' HalfSpace.
        # Unsafe set is the UNION over groups; SAT iff any group is hit.
        for group in spec:
            disjunct = group.get('Hg', None)
            if isinstance(disjunct, HalfSpace):
                viol = _row_violates(disjunct, ys)
                idx = np.flatnonzero(viol)
                if idx.size > 0:
                    return idx[0]
            elif isinstance(disjunct, list):
                for hs in disjunct:
                    if isinstance(hs, HalfSpace):
                        viol = _row_violates(hs, ys)
                        idx = np.flatnonzero(viol)
                        if idx.size > 0:
                            return idx[0]
        return None

    return None


# ---------------------------------------------------------------------------
# Instance loader factories
# ---------------------------------------------------------------------------

# Each entry maps a benchmark name to a callable that yields a list of
# (instance_name, loader) tuples, where ``loader()`` returns
# ``(network, boxes, spec, name)``. Reuses Exp 1 / Exp 2 helpers.

VNNCOMP_BENCHMARK_ROOTS = {
    'acasxu_2023': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023',
    'cifar100_2024': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/cifar100_2024',
    'collins_rul_cnn_2022': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/collins_rul_cnn_2022',
    'cora_2024': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/cora_2024',
    'dist_shift_2023': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/dist_shift_2023',
    'linearizenn_2024': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/linearizenn_2024',
    'malbeware': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/malbeware',
    'metaroom_2023': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/metaroom_2023',
    'ml4acopf_2024': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/ml4acopf_2024',
    'safenlp_2024': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/safenlp_2024',
    'tllverify_2023': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/tllverifybench_2023',
    'vit_2023': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/vit_2023',
    'tinyimagenet_2024': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024',
    'yolo_2023': '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/cctsdb_yolo_2023',
}


def load_vnncomp_instances(benchmark_name: str, n: int):
    """Return up to ``n`` instances from a VNN-COMP benchmark, as a list
    of ``(instance_name, loader)`` tuples. Each ``loader()`` returns
    ``(network, boxes, spec, name)`` matching the Exp 2 contract.
    """
    import os

    from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
        load_instance, parse_instances_csv,
    )

    root_str = VNNCOMP_BENCHMARK_ROOTS.get(benchmark_name)
    if root_str is None:
        raise KeyError(f'unknown VNNCOMP benchmark: {benchmark_name}')
    root = Path(os.path.expanduser(root_str))
    instances_csv = root / 'instances.csv'
    if not instances_csv.exists():
        raise FileNotFoundError(f'instances.csv missing at {instances_csv}')

    rows = parse_instances_csv(instances_csv)[:n]
    out = []
    for (onnx_rel, vnn_rel, vnncomp_t) in rows:
        name = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'

        def _make(onnx_rel=onnx_rel, vnn_rel=vnn_rel, name=name):
            net, boxes, spec = load_instance(root, onnx_rel, vnn_rel)
            return net, boxes, spec, name

        # Carry the VNN-COMP per-row timeout as the third tuple element so
        # ``run_baseline_sweep`` can enforce it via SIGALRM. Zero/missing
        # values fall back to the runner's default timeout.
        timeout_s = vnncomp_t if isinstance(vnncomp_t, int) and vnncomp_t > 0 else 0
        out.append((name, _make, timeout_s))
    return out


def load_image_instances(benchmark_name: str, n: int):
    """Return up to ``n`` instances from an image-classification benchmark
    in Exp 2 (cifar10_resnet110, vit_small_cifar10).

    Each ``loader()`` returns ``(network, boxes, spec, name)``. Note
    that the Exp 2 ResNet-110 loader requires Cohen-et-al pretrained
    weights; if missing it raises ``FileNotFoundError``. The runner is
    expected to catch the error and emit a TODO row.
    """
    if benchmark_name == 'cifar10_resnet110':
        return _load_cifar10_resnet110(n)
    if benchmark_name == 'vit_small_cifar10':
        return _load_vit_small_cifar10(n)
    raise KeyError(f'unknown image benchmark: {benchmark_name}')


def _load_cifar10_resnet110(n: int):
    from examples.FlowConformal.experiments.exp2_prob_scale import (
        exp2_run_cifar10_resnet110 as mod,
    )

    network = mod._load_pretrained(mod._DEFAULT_SIGMA)
    if network is None:
        raise FileNotFoundError(
            f'RS ResNet-110 pretrained weights missing at '
            f'{mod._SMOOTHING_REPO}/models/cifar10/resnet110/'
            f'noise_{mod._DEFAULT_SIGMA}/checkpoint.pth.tar; '
            f'download from {mod._WEIGHTS_URL}')
    imgs, labels = mod._load_cifar10_test(n)

    # ResNet-110 (and ViT-Small) have no VNN-COMP per-row timeout. Per
    # ``docs/plans/2026-04-27-paper-experiments-design.md`` Exp 2 §
    # "Timeouts" the design fixes 300s/instance for these adversarial-
    # robustness setups (no published VNN-COMP timeout exists).
    image_default_timeout_s = 300
    out = []
    for i in range(len(imgs)):
        name, loader = mod._make_loader(network, imgs[i], int(labels[i]), i)
        out.append((name, loader, image_default_timeout_s))
    return out


def _load_vit_small_cifar10(n: int):
    from examples.FlowConformal.experiments.exp2_prob_scale import (
        exp2_run_vit_small_cifar10 as mod,
    )

    # The vit_small loader has its own pretrained chain (timm -> torchvision
    # -> tiny in-file). We surface its own errors verbatim.
    network, _ = mod._load_model(prefer='timm')
    if network is None:
        raise FileNotFoundError(
            'vit_small_cifar10: no model variant available '
            '(timm/torchvision/tiny all failed)')
    imgs, labels = mod._load_cifar10_test(n)

    # See ``_load_cifar10_resnet110`` for the 300s timeout rationale.
    image_default_timeout_s = 300
    out = []
    for i in range(len(imgs)):
        name, loader = mod._make_loader(network, imgs[i], int(labels[i]), i)
        out.append((name, loader, image_default_timeout_s))
    return out


# ---------------------------------------------------------------------------
# Network adaptors
# ---------------------------------------------------------------------------

def empirical_coverage_for_box(
    model_fn: Callable,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    box_lb: np.ndarray,
    box_ub: np.ndarray,
    *,
    n_test: int = 1000,
    seed: int = 0,
    pad: float = 1e-6,
) -> tuple[float, float, int]:
    """Empirical coverage of a probabilistic reach BOX over the input
    distribution P_X = uniform on the L_infinity ball ``[input_lb, input_ub]``.

    Samples ``n_test`` held-out test points uniformly from the input box,
    pushes them through ``model_fn``, and returns the fraction whose
    output lies inside ``[box_lb, box_ub]`` (per-coordinate inclusion,
    with a small ``pad`` to avoid float boundary misses).

    Args:
        model_fn: Numpy-in / numpy-out network forward.
        input_lb, input_ub: Flat 1-D arrays of input bounds.
        box_lb, box_ub: Flat 1-D arrays of the reach-set box bounds.
        n_test: Number of held-out test samples (default 1000).
        seed: RNG seed for the test set.
        pad: Per-coordinate slack for boundary inclusion.

    Returns:
        ``(coverage, sigma, n_test)`` where ``coverage`` is the fraction
        of samples in the box, ``sigma = sqrt(p*(1-p)/n)`` is the
        binomial std-err.
    """
    lb_in = np.asarray(input_lb, dtype=np.float64).flatten()
    ub_in = np.asarray(input_ub, dtype=np.float64).flatten()
    box_lb = np.asarray(box_lb, dtype=np.float64).flatten()
    box_ub = np.asarray(box_ub, dtype=np.float64).flatten()
    rng = np.random.default_rng(seed + 17_293_847)
    xs = rng.uniform(lb_in, ub_in,
                     size=(n_test, lb_in.size)).astype(np.float32)
    ys = model_fn(xs)
    ys = np.asarray(ys, dtype=np.float64)
    if ys.ndim > 2:
        ys = ys.reshape(ys.shape[0], -1)
    if ys.shape[1] != box_lb.size:
        # Dim mismatch: cannot compute coverage.
        return float('nan'), float('nan'), n_test
    inside = np.all((ys >= box_lb - pad) & (ys <= box_ub + pad), axis=1)
    cov = float(inside.mean())
    sigma = float(np.sqrt(max(cov * (1.0 - cov), 0.0) / max(n_test, 1)))
    return cov, sigma, n_test


def torch_callable(network, batch_size: int = 100):
    """Wrap a torch ``nn.Module`` (or any object with ``.forward``-like
    behaviour) into a numpy-in/numpy-out callable suitable for
    ``n2v.probabilistic.conformal_reach``. Inference runs in eval mode on
    whatever device the network's parameters/buffers live on.

    The device is auto-detected once at wrap time. If the caller has
    moved ``network`` to GPU before calling this function, inference
    will run on GPU; otherwise it stays on CPU. CPU-only callers see
    no behavioural change (inputs are CPU-resident by default;
    ``.to(device)`` is a no-op when ``device`` is the same CPU).
    """
    import torch

    if hasattr(network, 'eval'):
        network.eval()

    # Auto-detect the network's device by inspecting parameters or
    # buffers. Falls back to CPU for parameter-less nets (e.g.
    # buffer-only ONNX-converted ACAS Xu wrappers).
    try:
        device = next(network.parameters()).device
    except (StopIteration, AttributeError):
        try:
            device = next(network.buffers()).device
        except (StopIteration, AttributeError):
            device = torch.device('cpu')

    @torch.no_grad()
    def _f(x_np: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(x_np), dtype=torch.float32).to(device)
        # Inference in chunks to avoid blowing memory on large nets.
        outs = []
        for i in range(0, x.shape[0], batch_size):
            y = network(x[i:i + batch_size])
            if hasattr(y, 'detach'):
                y = y.detach().cpu().numpy()
            outs.append(np.asarray(y))
        if len(outs) == 0:
            return np.zeros((0,))
        out = np.concatenate(outs, axis=0)
        # Ensure 2D (batch, output_dim) — flatten trailing dims if any.
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)
        return out

    return _f


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

KNOWN_BENCHMARKS = (
    list(VNNCOMP_BENCHMARK_ROOTS.keys())
    + ['cifar10_resnet110', 'vit_small_cifar10']
)


def add_common_args(parser):
    """Add the standard runner CLI flags."""
    parser.add_argument(
        '--benchmark', type=str, required=True,
        choices=KNOWN_BENCHMARKS,
        help='Benchmark name. Use --list to print available benchmarks.')
    parser.add_argument('--smoke', action='store_true',
                        help='Run only 2 instances for smoke test.')
    parser.add_argument('--instances', type=int, default=10,
                        help='Run first N instances if not --smoke.')
    parser.add_argument('--output-csv', type=Path, default=None,
                        help='Override default output path.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Master seed.')


def resolve_n_instances(args) -> int:
    return 2 if args.smoke else args.instances


def resolve_output_csv(args, baseline: str) -> Path:
    if args.output_csv:
        return Path(args.output_csv)
    suffix = '_smoke' if args.smoke else ''
    out_dir = Path(__file__).parent / 'outputs'
    return out_dir / f'baseline_{baseline}_{args.benchmark}{suffix}.csv'


def load_benchmark_instances(benchmark: str, n: int):
    """Dispatch on benchmark name to the right loader."""
    if benchmark in VNNCOMP_BENCHMARK_ROOTS:
        return load_vnncomp_instances(benchmark, n)
    return load_image_instances(benchmark, n)


# ---------------------------------------------------------------------------
# Sweep helper
# ---------------------------------------------------------------------------

def _baseline_raise_timeout(signum, frame):
    raise TimeoutError()


def run_baseline_sweep(
    *,
    baseline: str,
    benchmark: str,
    instances: list,
    out_csv: Path,
    extra_fields: list,
    process_one: Callable,
    default_timeout_s: int = 600,
):
    """Generic sweep loop for a baseline runner.

    ``process_one(loader, seed)`` must return a dict with at least
    ``verdict`` (one of UNSAT/SAT/UNKNOWN/ERROR/NOT_APPLICABLE/TIMEOUT),
    and optionally ``error`` and any ``extra_fields``. Wall-clock is
    timed by this loop. ``process_one`` should never raise — exceptions
    propagate as ERROR rows, except ``TimeoutError`` which yields a
    TIMEOUT row.

    Per-instance soft timeout: each ``instances`` entry may be a 2-tuple
    ``(name, loader)`` or a 3-tuple ``(name, loader, vnncomp_timeout_s)``.
    When a positive ``vnncomp_timeout_s`` is provided, this loop arms a
    SIGALRM with that budget for the duration of ``process_one`` so the
    baseline adheres to the same per-instance budget the sound verifiers
    got. Otherwise the SIGALRM uses ``default_timeout_s``.
    """
    import signal as _signal

    writer, f, _fields = open_csv_writer(out_csv, extra_fields)
    counts: dict = {}
    t_start = time.time()
    print(f'[baseline={baseline} bench={benchmark}] running '
          f'{len(instances)} instances; out={out_csv}; '
          f'default_timeout={default_timeout_s}s', flush=True)
    _signal.signal(_signal.SIGALRM, _baseline_raise_timeout)
    try:
        for k, item in enumerate(instances, start=1):
            # Backward-compatible unpacking: 2-tuple or 3-tuple.
            if len(item) >= 3:
                name, loader, vnncomp_t = item[0], item[1], item[2]
            else:
                name, loader = item[0], item[1]
                vnncomp_t = 0
            timeout_s = (vnncomp_t if isinstance(vnncomp_t, int)
                         and vnncomp_t > 0 else default_timeout_s)

            print(f'  [{k}/{len(instances)}] {name} budget={timeout_s}s',
                  flush=True)
            t0 = time.time()
            try:
                _signal.alarm(int(timeout_s))
                row = process_one(loader, name)
            except TimeoutError:
                row = {'verdict': 'TIMEOUT',
                       'error': f'per-instance timeout {timeout_s}s'}
            except Exception as e:  # last-resort safety net
                row = {'verdict': 'ERROR',
                       'error': f'unhandled {type(e).__name__}: {e}'}
            finally:
                _signal.alarm(0)
            wall = time.time() - t0
            row.setdefault('error', '')
            out_row = {fld: '' for fld in _fields}
            out_row['benchmark'] = benchmark
            out_row['instance'] = name
            out_row['baseline'] = baseline
            out_row['wall_s'] = _fmt(wall, '.2f')
            for k2, v in row.items():
                out_row[k2] = v if not isinstance(v, float) else _fmt(v, '.4g')
            writer.writerow(out_row)
            f.flush()
            v = row['verdict']
            counts[v] = counts.get(v, 0) + 1
            print(f'    verdict={v}  wall={wall:.1f}s', flush=True)
    finally:
        f.close()
    print(f'[baseline={baseline}] === done in '
          f'{(time.time() - t_start):.1f}s   counts={counts} ===', flush=True)
