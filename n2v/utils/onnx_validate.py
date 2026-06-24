"""Re-validate a falsification counterexample on the RAW ONNX via onnxruntime.

VNN-COMP grades a ``sat`` witness by running the input through the ORIGINAL ONNX
model in onnxruntime and checking that (a) the witnessed outputs reproduce within
1e-3 relative error and (b) the .vnnlib constraints are met within 1e-4 absolute
error (rules.md). n2v finds counterexamples on the onnx2torch-CONVERTED model, so
a converted-vs-raw divergence (lossy/shimmed op conversion) could produce a
witness the grader rejects — scored *incorrect* (a catastrophic −150).

This module re-runs the witness input on the raw ONNX and reports whether it
genuinely lands in the unsafe region. The caller should emit the returned ORT
output as the witness ``Y`` so the grader's reproduction check passes by
construction. Conservative by design: any error or non-violation means "not a
sound counterexample", so the caller downgrades ``sat`` → ``unknown`` (0 points)
rather than emit a witness that would score −150.
"""

import numpy as np


def onnx_forward(onnx_path: str, x_flat) -> np.ndarray:
    """Run ``x_flat`` (flat, in ONNX input-variable order) through the raw ONNX
    in onnxruntime; return the output flattened. Reshapes ``x_flat`` to the
    model's declared input shape (dynamic/batch dims coerced to 1)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    x = np.asarray(x_flat, dtype=np.float32).reshape(shape)
    out = sess.run(None, {inp.name: x})[0]
    return np.asarray(out).reshape(-1)


def in_unsafe_region(y, groups, tol: float = 1e-4) -> bool:
    """AND-of-OR unsafe-region membership for output ``y`` with the VNN-COMP
    1e-4 absolute constraint tolerance (rules.md).

    ``groups`` is the ``List[List[HalfSpace]]`` produced by
    ``_extract_halfspace_groups``: AND across groups, OR within a group, and all
    rows of a halfspace must hold (``G @ y <= g + tol``). Mirrors the sound
    ``_output_satisfies_property`` check but at the grader's 1e-4 tolerance.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    for group in groups:                       # AND across groups
        hit = False
        for hs in group:                       # OR within a group
            G = np.asarray(hs.G, dtype=np.float64)
            g = np.asarray(hs.g, dtype=np.float64).reshape(-1, 1)
            if bool(np.all(G @ y <= g + tol)):
                hit = True
                break
        if not hit:
            return False
    return True
