"""Unit tests for the onnxruntime counterexample re-validation gate
(``n2v.utils.onnx_validate``), used by the VNN-COMP runner to confirm a
falsification witness still violates on the RAW ONNX before emitting ``sat``.

The gate is on the −150 verdict path, so these tests pin its mechanics
deterministically: it must ACCEPT a genuine counterexample and REJECT a safe
point. A silent over-rejection bug would turn every falsifier ``sat`` into
``unknown`` (a breadth catastrophe), which a torch-only test would miss — so we
export a real model to ONNX and round-trip through onnxruntime.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import HalfSpace
from n2v.utils.onnx_validate import onnx_forward, in_unsafe_region


@pytest.fixture(scope="module")
def identity_onnx():
    """Export y = x (2->2 identity Linear) to a temp ONNX file."""
    model = nn.Linear(2, 2, bias=False)
    model.weight.data = torch.eye(2)
    model.eval()
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    # dynamo=False uses the legacy TorchScript exporter (no onnxscript dep).
    torch.onnx.export(model, torch.zeros(1, 2), path,
                      input_names=["x"], output_names=["y"], opset_version=13,
                      dynamo=False)
    yield path
    os.remove(path)


class TestOnnxForward:
    def test_roundtrip_matches(self, identity_onnx):
        y = onnx_forward(identity_onnx, np.array([0.3, 0.7], dtype=np.float32))
        np.testing.assert_allclose(y, [0.3, 0.7], atol=1e-6)

    def test_accepts_flat_or_shaped_input(self, identity_onnx):
        # Flat input in ONNX-variable order is reshaped to the declared shape.
        y = onnx_forward(identity_onnx, [0.1, 0.9])
        assert y.shape == (2,)


class TestInUnsafeRegion:
    def test_single_halfspace(self):
        # Unsafe: y0 >= 0.5  (-y0 <= -0.5)
        groups = [[HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))]]
        assert in_unsafe_region([1.0, 0.0], groups)
        assert not in_unsafe_region([0.0, 0.0], groups)

    def test_and_across_groups(self):
        # (y0 >= 0.5) AND (y1 >= 0.5)
        groups = [
            [HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))],
            [HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.5]]))],
        ]
        assert in_unsafe_region([1.0, 1.0], groups)
        assert not in_unsafe_region([1.0, 0.0], groups)   # only one group hit
        assert not in_unsafe_region([0.0, 1.0], groups)

    def test_or_within_group(self):
        # One group, OR of two halfspaces: y0 >= 0.8  OR  y0 <= 0.2
        groups = [[
            HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.8]])),
            HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.2]])),
        ]]
        assert in_unsafe_region([0.9, 0.0], groups)
        assert in_unsafe_region([0.1, 0.0], groups)
        assert not in_unsafe_region([0.5, 0.0], groups)

    def test_tolerance_boundary(self):
        # y0 >= 0.5; a point 1e-5 short still counts unsafe at 1e-4 tol
        # (matches the grader's 1e-4 abs), but a point 1e-3 short does not.
        groups = [[HalfSpace(np.array([[-1.0]]), np.array([[-0.5]]))]]
        assert in_unsafe_region([0.5 - 1e-5], groups)        # within 1e-4
        assert not in_unsafe_region([0.5 - 1e-3], groups, tol=1e-4)


class TestGateEndToEnd:
    def test_accepts_real_ce_rejects_safe(self, identity_onnx):
        # Unsafe region: y0 >= 0.5 (since y = x, x0 >= 0.5 is a CE).
        groups = [[HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))]]
        # genuine counterexample
        y_ce = onnx_forward(identity_onnx, np.array([0.9, 0.1], dtype=np.float32))
        assert in_unsafe_region(y_ce, groups)
        # safe point -> gate rejects (would downgrade sat->unknown)
        y_safe = onnx_forward(identity_onnx, np.array([0.1, 0.1], dtype=np.float32))
        assert not in_unsafe_region(y_safe, groups)
