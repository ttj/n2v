"""Synthetic 1-Lipschitz ReLU MLP family for Exp 4 controlled scaling.

Each network is a fully-connected ReLU MLP with random Gaussian weights,
spectrally normalised so every linear layer has spectral norm 1. With
ReLU (also 1-Lipschitz) the composition is 1-Lipschitz overall, which
gives an analytic upper bound on the output range over any L∞ input box
and lets us construct UNSAT-by-construction specs for Exp 4.

Family parameters:
    * EXP4_DEPTHS — 7 depths spanning 4 orders of magnitude in #params:
      ~3.6K (D=2) → ~10M (D=40).
    * EXP4_WIDTH = 512 (held constant across depths).
    * EXP4_IN_DIM = 5 (small input dim so VNN-COMP-style l∞ boxes have
      modest L2 radius).
    * EXP4_OUT_DIM = 1.

Depth convention: ``D`` counts the number of *linear* layers, separated
by ReLUs. ``D=2`` is one input projection plus one output projection,
no hidden layer; ``D=40`` is one input + 38 hidden + one output.

Per-depth seed: ``hash(("exp4_synth", depth)) & 0x7FFFFFFF`` — pure
function of depth so the family is deterministic and any tool can
re-derive the same network.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

EXP4_DEPTHS: tuple[int, ...] = (2, 4, 8, 16, 24, 32, 40)
EXP4_WIDTH: int = 512
EXP4_IN_DIM: int = 5
EXP4_OUT_DIM: int = 1


def _stable_hash(*parts) -> int:
    """Cross-process-deterministic hash. Python's built-in ``hash()`` of
    strings is randomised by PYTHONHASHSEED, which makes
    ``hash(("exp4_synth", depth))`` return different values across
    process invocations — fatal for our reproducibility convention.
    """
    h = hashlib.sha256(repr(parts).encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'big') & 0x7FFFFFFF


def network_seed(depth: int) -> int:
    return _stable_hash("exp4_synth", depth)


def n_params(
    depth: int,
    width: int = EXP4_WIDTH,
    in_dim: int = EXP4_IN_DIM,
    out_dim: int = EXP4_OUT_DIM,
) -> int:
    """Closed-form parameter count for the family.

    For ``D=1`` the network degenerates to a single linear layer.
    """
    if depth <= 1:
        return in_dim * out_dim + out_dim
    p = in_dim * width + width                     # input projection
    p += (depth - 2) * (width * width + width)     # hidden layers
    p += width * out_dim + out_dim                 # output projection
    return p


class SyntheticMLP(nn.Module):
    """Fully-connected ReLU MLP with the EXP4 depth convention.

    Spectral normalisation is applied *after* construction by
    :func:`make_synthetic_network`, so this class is just the structural
    skeleton.
    """

    def __init__(
        self,
        depth: int,
        width: int = EXP4_WIDTH,
        in_dim: int = EXP4_IN_DIM,
        out_dim: int = EXP4_OUT_DIM,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f'depth must be >= 1, got {depth}')
        layers: list[nn.Module] = []
        if depth == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            for _ in range(depth - 2):
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _spectral_normalize_(linear: nn.Linear) -> None:
    """Rescale ``linear.weight`` so its largest singular value is exactly 1.

    One-shot exact normalisation via SVD — the network is built once at
    construction time, so power iteration's amortised cost isn't needed.
    """
    W = linear.weight.data
    sigma_max = torch.linalg.svdvals(W).max()
    if sigma_max > 0:
        linear.weight.data = W / sigma_max


def make_synthetic_network(
    depth: int,
    *,
    width: int = EXP4_WIDTH,
    in_dim: int = EXP4_IN_DIM,
    out_dim: int = EXP4_OUT_DIM,
    seed: int | None = None,
) -> SyntheticMLP:
    """Build the synthetic MLP for a given depth.

    Weights are sampled from ``N(0, 1)`` then spectrally normalised so
    each linear layer is 1-Lipschitz. The seed defaults to
    :func:`network_seed`, which is a pure function of ``depth`` so the
    same network is reproduced regardless of when or where it's built.
    """
    if seed is None:
        seed = network_seed(depth)
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    net = SyntheticMLP(depth, width=width, in_dim=in_dim, out_dim=out_dim)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=g))
                m.bias.zero_()
            _spectral_normalize_(m)
    net.eval()
    return net


def export_onnx(
    net: nn.Module,
    path: str | Path,
    in_dim: int = EXP4_IN_DIM,
    *,
    opset_version: int = 13,
) -> None:
    """Export the network as ONNX for αβ-CROWN / NeuralSAT ingestion.

    Uses a fixed batch dimension of 1 with no dynamic axes — VNN-COMP
    benchmarks typically ship single-example ONNX. Callers should
    re-batch internally if they need vectorised inference.

    Two export gotchas we fix here:

    1. ``do_constant_folding=True`` collapses constants like spectral
       norm scalars into the Gemm weights, so the exported graph has
       just ``Gemm + Relu`` nodes (no spurious ``Identity`` ops on
       parameters).

    2. We post-process with :func:`onnxsim.simplify` to strip any
       remaining no-op nodes. αβ-CROWN's ``onnx2pytorch`` parser
       mishandles ``Identity`` ops that duplicate parameter tensors
       — at depth 8 it emits 6 ``Identity`` nodes on ``net.0.bias``,
       which onnx2pytorch interprets as extra positional arguments
       to ``Linear.forward``, producing the obscure
       ``Linear.forward() takes 2 positional arguments but 3 were
       given`` error. With simplification the graph is just
       ``Gemm × depth + Relu × (depth-1)``.
    """
    sample = torch.zeros(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        net, sample, str(path),
        input_names=['X'], output_names=['Y'],
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Post-export simplification: strip Identity / unused nodes that
    # PyTorch's ONNX exporter emits when a parameter is referenced
    # multiple times by the autograd graph.
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(str(path))
        simplified, ok = simplify(model)
        if ok:
            onnx.save(simplified, str(path))
    except ImportError:
        # onnxsim is optional; without it we fall back to the
        # constant-folded export, which is good enough for shallow
        # networks but may trip onnx2pytorch at depth.
        pass
