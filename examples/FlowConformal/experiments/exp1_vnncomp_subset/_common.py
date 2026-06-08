"""Shared utilities for Exp 1 VNN-COMP benchmark sweeps (our method only).

Generic instance loader + sweep loop, parameterised by benchmark root
directory. Abstracts over benchmarks where:

  - ONNX input shape may not be (batch, 5) but (batch, N) or (batch, C, H, W).
  - VNN-LIB ``prop`` may be a single HalfSpace, a list[HalfSpace], or
    a list[dict] with 'Hg' field (AND-of-OR groups).
  - OR-of-input-regions handled identically to ACAS Xu prop_6.

Sound-verifier comparisons are done at analysis time against VNN-COMP's
public ``vnncomp2025_results/`` CSVs, not by re-running third-party tools.

The constants below are the locked paper config; do not change without
coordinating with the cross-benchmark comparands.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from n2v.sets.halfspace import HalfSpace
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


# >>> Locked paper config <<<
_FLOW_CONFIG = 'base'
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_ALPHA = 0.001
_VERIFICATION_METHOD = 'amls'

# VNN-COMP standard per-instance budget. AMLS may exceed; we use a soft
# Python SIGALRM at this value and emit verdict='TIMEOUT'.
VNNCOMP_TIMEOUT_S = 116


# ---------------------------------------------------------------------------
# Network wrapper
# ---------------------------------------------------------------------------

class _GenericONNXWrapper(nn.Module):
    """Wrap an ONNX-loaded ``GraphModule`` so it accepts a flat
    ``(batch, n)`` input batch and returns ``(batch, output_dim)``.

    The wrapper auto-detects the model's expected input shape from a
    single zero-tensor probe (because VNN-COMP ONNX exports often have
    the leading dim as 0/-1 in the graph metadata, which onnx2pytorch
    sometimes preserves). On forward, it reshapes ``(batch, n)`` to
    ``(batch, *expected_input_shape)``.
    """

    def __init__(self, inner: nn.Module, input_shape: tuple,
                 output_squeeze: bool = True,
                 batch_loop: bool = False,
                 batch_loop_unbatched: bool = False):
        """
        Args:
            inner: ONNX-loaded ``nn.Module``.
            input_shape: Per-sample input shape excluding the batch dim.
                E.g. ``(784,)`` for a flat MNIST classifier; ``(3, 32, 56)``
                for a metaroom CNN; ``(1, 1, 5)`` for ACAS Xu.
            output_squeeze: If True and inner output has shape
                ``(batch, 1, k)``, squeeze the singleton.
            batch_loop: If True, the inner ONNX module does NOT support
                B>1 along the leading dim — call it once per sample as
                ``inner(x[i:i+1])`` (i.e. with a leading batch=1 dim) and
                stack the outputs.
            batch_loop_unbatched: If True, the inner ONNX module is
                fundamentally unbatched: it expects ``inner(x[i])`` with
                NO leading batch dim. Used by VNN-COMP exports such as
                cctsdb_yolo_2023 where the graph contains hard-coded
                ``Slice(axes=[0])`` ops over the input that conflict with
                a leading batch dim. Detected by ``_detect_input_shape``
                when only the no-batch-prefix probe succeeds.
        """
        super().__init__()
        self.inner = inner
        self.input_shape = tuple(int(d) for d in input_shape)
        self.output_squeeze = output_squeeze
        self.batch_loop = batch_loop
        self.batch_loop_unbatched = batch_loop_unbatched

    def _call_inner(self, x: torch.Tensor) -> torch.Tensor:
        """Run inner with optional batch-loop fallback."""
        if self.batch_loop_unbatched:
            # Inner expects no leading batch dim. Drop it for each sample
            # and re-add it on the output via stack.
            outs = []
            for i in range(x.shape[0]):
                y = self.inner(x[i])
                outs.append(y)
            return torch.stack(outs, dim=0)
        if not self.batch_loop:
            return self.inner(x)
        outs = []
        for i in range(x.shape[0]):
            outs.append(self.inner(x[i:i + 1]))
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either flat (batch, n) or already-shaped (batch, *input_shape).
        if x.dim() == 2 and x.shape[1] == int(np.prod(self.input_shape)):
            x = x.reshape(x.shape[0], *self.input_shape)
        y = self._call_inner(x)
        # Some networks output (batch, 1, ..., k); squeeze singletons trailing
        # the batch dim to leave (batch, k) for downstream HalfSpace eval.
        if self.output_squeeze:
            while y.dim() > 2 and y.shape[1] == 1:
                y = y.squeeze(1)
        # If still > 2D, flatten trailing dims so downstream HalfSpace
        # (which expects (batch, output_dim)) gets a usable shape.
        if y.dim() > 2:
            y = y.reshape(y.shape[0], -1)
        return y


def _patch_batch_reshape_constants(model: nn.Module, input_shape: tuple) -> bool:
    """Try to patch batch-1-hardcoded ``OnnxReshape`` constants to support
    arbitrary batch sizes.

    Some VNN-COMP ONNX exports embed a literal ``Reshape`` whose target
    shape is ``(1, -1)`` (or ``(1, K)``), which collapses the leading
    batch dim instead of preserving it. We rewrite each such constant to
    ``(-1, K)`` where ``K`` is the per-sample flat size measured during a
    batch=1 probe. Output is then ``(B, K)`` for any ``B``.

    The patch is applied in-place on ``model``. Returns True if at least
    one constant was patched AND the model now accepts batch=2; False
    otherwise (caller should fall back to batch_loop).

    Soundness: at batch=1 the rewritten reshape produces an identical
    output to the original (``(-1, K)`` resolves to ``(1, K)`` when input
    has batch=1). At batch>1 it produces ``(B, K)``, which is the
    semantically correct generalisation; the rest of the network (Conv2d,
    Linear, ReLU) is naturally batch-agnostic, so outputs are equivalent
    to the per-sample loop up to float32 accumulation order.
    """
    # Collect (constant_module, reshape_module, predecessor_input_shape) by
    # hooking every submodule and running a batch=1 probe.
    captured: dict = {}
    hooks = []
    try:
        from onnx2torch.node_converters.reshape import OnnxReshape  # type: ignore
        from onnx2torch.node_converters.constant import OnnxConstant  # type: ignore
    except Exception:
        return False

    # Hook every Reshape op to capture its input tensor shape and the
    # second positional arg (the shape constant tensor).
    def make_hook(name):
        def _h(mod, inputs, output):
            if len(inputs) >= 2:
                captured[name] = {
                    'input_shape': tuple(inputs[0].shape),
                    'shape_arg': inputs[1].detach().clone(),
                    'output_shape': tuple(output.shape),
                }
        return _h

    for name, m in model.named_modules():
        if isinstance(m, OnnxReshape):
            hooks.append(m.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            _ = model(torch.zeros(1, *input_shape))
    except Exception:
        for h in hooks:
            h.remove()
        return False
    for h in hooks:
        h.remove()

    if not captured:
        return False

    # Match each Reshape with its constant predecessor: in the converted
    # GraphModule, the constant shares the same numeric prefix in its
    # name (e.g. "8/Constant" pairs with "8/Reshape").
    name_to_module = dict(model.named_modules())
    patched_any = False
    for rs_name, info in captured.items():
        # Pair: same prefix, different suffix.
        prefix = rs_name.rsplit('/', 1)[0] if '/' in rs_name else None
        if prefix is None:
            continue
        const_name = f'{prefix}/Constant'
        const_mod = name_to_module.get(const_name)
        if not isinstance(const_mod, OnnxConstant):
            continue
        # Check the original constant value: shape (rank,), 1-D int tensor.
        val = const_mod.value
        if not isinstance(val, torch.Tensor):
            continue
        if val.dim() != 1 or val.dtype not in (torch.int32, torch.int64):
            continue
        target = val.tolist()
        # We only patch when the leading entry is exactly 1 (batch dim
        # hard-coded) and the target rank is 2 — i.e. ``(1, -1)`` or
        # ``(1, K)`` flatten patterns. Higher-rank reshapes are left
        # alone to avoid invalid -1 substitutions.
        if len(target) != 2 or target[0] != 1:
            continue
        # Compute the concrete K from the captured input shape.
        in_shape = info['input_shape']
        # Total size at batch=1 = prod(in_shape); per-sample size = prod(in_shape[1:])
        per_sample = 1
        for d in in_shape[1:]:
            per_sample *= int(d)
        # Sanity: original output (1, K) should match (1, per_sample).
        out_shape = info['output_shape']
        if len(out_shape) != 2 or int(out_shape[1]) != per_sample:
            continue
        new_const = torch.tensor([-1, per_sample], dtype=val.dtype)
        const_mod.value = new_const
        patched_any = True

    if not patched_any:
        return False

    # Verify batch>1 now works.
    try:
        with torch.no_grad():
            _ = model(torch.zeros(2, *input_shape))
        return True
    except Exception:
        return False


def _detect_input_shape(model: nn.Module, vnn_input_dim: int
                         ) -> tuple[tuple, bool, bool]:
    """Probe ``model`` to find the input shape it expects, given that
    flatten-of-the-shape == ``vnn_input_dim``. Tries common candidate
    shapes; raises if none accepts a zero tensor.

    Returns ``(shape, batch_loop, batch_loop_unbatched)`` where:

    * ``batch_loop=True`` — the inner ONNX module supports batch=1 but
      not batch>1 (hard-coded reshape constants), and a graph-level
      patch could not rewrite the offending reshape. Wrapper falls back
      to a per-sample ``inner(x[i:i+1])`` loop.
    * ``batch_loop_unbatched=True`` — the inner ONNX module is
      fundamentally unbatched: only ``inner(zeros(*shape))`` (no leading
      batch dim at all) succeeds. Wrapper falls back to a per-sample
      ``inner(x[i])`` loop. Pattern observed on cctsdb_yolo_2023 where
      the graph has hard-coded ``Slice(axes=[0])`` ops over the input.
    """
    candidates: list[tuple] = [(vnn_input_dim,)]
    # Common image shapes whose flatten matches vnn_input_dim
    if vnn_input_dim == 400:  # collins_rul_cnn_2022 (1, 20, 20)
        candidates += [(1, 20, 20)]
    if vnn_input_dim == 784:
        # Includes (1, 784, 1) for relusplitter's mnist_fc_vnncomp2022
        # ONNX exports, which embed a trailing singleton dim.
        candidates += [(1, 28, 28), (1, 1, 784), (1, 784), (1, 784, 1)]
    if vnn_input_dim == 792:  # dist_shift_2023 mnist_concat (extra 8 features)
        candidates += [(1, 792)]
    if vnn_input_dim == 800:  # collins_rul_cnn_2022 NN_rul_full_window_40 (1, 40, 20)
        candidates += [(1, 40, 20)]
    if vnn_input_dim == 3072:
        candidates += [(3, 32, 32), (1, 3, 32, 32)]
    if vnn_input_dim == 5376:
        candidates += [(3, 32, 56), (1, 3, 32, 56)]
    if vnn_input_dim == 8112:  # yolo_2023 TinyYOLO (3, 52, 52)
        candidates += [(3, 52, 52), (1, 3, 52, 52)]
    # ACAS Xu legacy
    if vnn_input_dim == 5:
        candidates += [(1, 1, 5)]
    # Try square image shape if dim is a perfect square
    import math as _math
    sqrt = int(_math.isqrt(vnn_input_dim))
    if sqrt * sqrt == vnn_input_dim:
        candidates += [(1, sqrt, sqrt), (sqrt, sqrt)]
    # 3-channel square image (vit_2023, yolo_2023, generic CIFAR-/ImageNet-
    # size inputs). Only added when vnn_input_dim is divisible by 3 and the
    # per-channel size is a perfect square.
    if vnn_input_dim % 3 == 0:
        per_chan = vnn_input_dim // 3
        s = int(_math.isqrt(per_chan))
        if s * s == per_chan:
            candidates += [(3, s, s), (1, 3, s, s)]
    # Generic catch-all: try (1, n) and (n,)
    candidates += [(1, vnn_input_dim)]

    for shape in candidates:
        try:
            with torch.no_grad():
                _ = model(torch.zeros(1, *shape))
        except Exception:
            continue
        # Found a working shape with batch=1. Now check whether batch>1
        # also works.
        try:
            with torch.no_grad():
                _ = model(torch.zeros(2, *shape))
            return shape, False, False
        except Exception:
            pass
        # Batch>1 failed. Try to patch hard-coded reshape constants
        # (common pattern in VNN-COMP ONNX exports). If that succeeds,
        # the model now supports any batch size.
        if _patch_batch_reshape_constants(model, shape):
            return shape, False, False
        return shape, True, False
    # No candidate accepted ``torch.zeros(1, *shape)``. Some ONNX models
    # (cctsdb_yolo_2023) are fundamentally unbatched — they require
    # ``torch.zeros(*shape)`` with no leading batch dim. Probe each
    # candidate again without the batch prefix; if one works, fall back
    # to a per-sample unbatched loop.
    for shape in candidates:
        try:
            with torch.no_grad():
                _ = model(torch.zeros(*shape))
        except Exception:
            continue
        return shape, False, True
    raise RuntimeError(
        f'Could not detect input shape for model with vnn_input_dim={vnn_input_dim}; '
        f'tried {candidates}')


# ---------------------------------------------------------------------------
# Spec normalisation
# ---------------------------------------------------------------------------

def _normalize_spec(prop_field):
    """Return a spec object accepted by ``run_verification_pipeline``:
    a HalfSpace, a list[HalfSpace], or a list[dict] with 'Hg' field
    (AND-of-OR groups). Pass-through for the load_vnnlib output shapes
    we know how to handle; raise NotImplementedError for unsupported
    shapes (caller marks the row SKIPPED).
    """
    if isinstance(prop_field, HalfSpace):
        return prop_field
    if isinstance(prop_field, list):
        if len(prop_field) == 0:
            raise NotImplementedError('empty spec')
        # list[dict]: AND-of-OR groups (cora etc.). Pipeline accepts.
        if isinstance(prop_field[0], dict):
            return prop_field
        # list[HalfSpace]: OR-of-ANDs.
        if isinstance(prop_field[0], HalfSpace):
            return prop_field
    raise NotImplementedError(
        f'unsupported prop field type: {type(prop_field).__name__}')


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def _ensure_decompressed(p: Path) -> Path:
    """If ``p`` is missing but ``p.with_suffix(p.suffix + '.gz')`` exists
    (or a sibling ``.gz``), decompress it once. Returns the path to the
    decompressed file. No-op if ``p`` is already present.
    """
    if p.exists():
        return p
    gz = Path(str(p) + '.gz')
    if gz.exists():
        import gzip
        with gzip.open(gz, 'rb') as src, open(p, 'wb') as dst:
            dst.write(src.read())
        return p
    return p


def parse_instances_csv(instances_csv: Path) -> list[tuple[str, str, int]]:
    """Parse a VNN-COMP-style ``instances.csv``. Rows are
    ``(onnx_path, vnnlib_path, timeout_seconds)``. Returns
    ``[(onnx_rel, vnnlib_rel, timeout_int), ...]``.
    """
    out = []
    with open(instances_csv, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                t = int(float(row[2]))
            except ValueError:
                continue
            out.append((row[0].strip(), row[1].strip(), t))
    return out


def load_instance(benchmark_root: Path, onnx_rel: str, vnn_rel: str):
    """Load (network, boxes, spec) for one VNN-COMP instance.

    Returns:
        network: ``nn.Module`` accepting flat ``(batch, n)`` inputs.
        boxes: list of ``(lb, ub)`` flat 1-D tuples (length 1 typically;
            length > 1 for OR-of-input-regions like ACAS Xu prop_6).
        spec: spec object accepted by ``run_verification_pipeline``.

    Raises:
        NotImplementedError: spec shape not supported.
        FileNotFoundError: paths missing.
    """
    onnx_path = benchmark_root / onnx_rel.lstrip('./').lstrip('/')
    vnn_path = benchmark_root / vnn_rel.lstrip('./').lstrip('/')
    # Some VNN-COMP benchmarks ship .onnx.gz / .vnnlib.gz alongside the
    # listed paths; transparently fall back if the uncompressed file is
    # missing. Decompresses to a sibling .onnx / .vnnlib so the cost is
    # paid once across the sweep.
    onnx_path = _ensure_decompressed(onnx_path)
    vnn_path = _ensure_decompressed(vnn_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f'onnx not found: {onnx_path}')
    if not vnn_path.exists():
        raise FileNotFoundError(f'vnnlib not found: {vnn_path}')

    inner = load_onnx(str(onnx_path))
    if hasattr(inner, 'eval'):
        inner.eval()

    prop = load_vnnlib(str(vnn_path))

    # Box(es)
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        lbs, ubs = prop['lb'], prop['ub']
        boxes = [(np.asarray(lb).flatten(), np.asarray(ub).flatten())
                 for lb, ub in zip(lbs, ubs)]
    else:
        boxes = [(np.asarray(prop['lb']).flatten(),
                  np.asarray(prop['ub']).flatten())]

    # Detect model input shape from the flat dim.
    flat_dim = int(boxes[0][0].size)
    input_shape, batch_loop, batch_loop_unbatched = _detect_input_shape(
        inner, flat_dim,
    )
    network = _GenericONNXWrapper(
        inner, input_shape=input_shape,
        batch_loop=batch_loop,
        batch_loop_unbatched=batch_loop_unbatched,
    )

    spec = _normalize_spec(prop['prop'])
    return network, boxes, spec
