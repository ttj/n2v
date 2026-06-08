"""ProbStar / StarV standalone CLI runner.

Self-contained: imports ONLY StarV (and numpy/onnx). Designed to be
invoked via subprocess from the n2v env's runners (mirrors how
:mod:`examples.FlowConformal.experiments._external_verifiers` dispatches
αβ-CROWN and NeuralSAT to their own conda envs).

Should be invoked with the ``starv`` conda env's python:

    python \\
        examples/FlowConformal/experiments/baselines/run_probstar_standalone.py \\
        --onnx_path <path>.onnx \\
        --vnnlib_path <path>.vnnlib \\
        --results_file /tmp/probstar_<tag>.json \\
        --timeout 100

Writes a JSON file at ``--results_file``:

    {
      "verdict": "UNSAT" | "SAT" | "UNKNOWN" | "NOT_APPLICABLE" | "ERROR",
      "p_filter": <float>,
      "lp_solver": <str>,
      "p_min": <float>,
      "p_max": <float>,
      "threshold": <float>,
      "n_disjuncts": <int>,
      "error": <str>
    }

The first line of the file is also the verdict word (for compatibility
with the αβ-CROWN/NeuralSAT-style ``_read_verdict`` parser), followed
by the JSON body.

Verdict policy mirrors :mod:`examples.FlowConformal.experiments.baselines.run_probstar`:

* UNSAT iff ``sum_disjunct(p_max) < unsafe_threshold``
* SAT iff ``sum_disjunct(p_min) > unsafe_threshold``
* UNKNOWN otherwise
* NOT_APPLICABLE if the network has ops StarV's loader can't handle,
  or the spec shape isn't supported.
* ERROR for any other failure (LP solver missing, init fails, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Find StarV — checked installed in the starv env's site-packages, but
# also support a local checkout via ``--starv-path``.
_STARV_PATH = Path(os.path.expanduser('~/v/other/StarV'))


def _import_starv(starv_path):
    """Lazy-import StarV symbols. Returns (modules_dict, error_str)."""
    if starv_path and str(starv_path) not in sys.path:
        sys.path.insert(0, str(starv_path))
    try:
        import StarV  # noqa: F401
        from StarV.set.star import Star
        from StarV.set.probstar import ProbStar
        from StarV.net.network import NeuralNetwork  # noqa: F401
        from StarV.verifier.verifier import quantiVerifyBFS
        from StarV.util.load import load_onnx_network
        from StarV.util.vnnlib import (
            read_vnnlib_simple, get_num_inputs_outputs,
        )
        return {
            'Star': Star, 'ProbStar': ProbStar,
            'load_onnx_network': load_onnx_network,
            'read_vnnlib_simple': read_vnnlib_simple,
            'get_num_inputs_outputs': get_num_inputs_outputs,
            'quantiVerifyBFS': quantiVerifyBFS,
        }, None
    except Exception as e:
        return None, f'starv_import_failed: {type(e).__name__}: {e}'


def _box_dict_to_lb_ub(box_dict, num_inputs):
    """``read_vnnlib_simple`` returns the input box as a dict
    ``{var_idx: (lb, ub)}``. Convert to dense (lb, ub) arrays."""
    lb = np.full(num_inputs, -np.inf, dtype=np.float64)
    ub = np.full(num_inputs, np.inf, dtype=np.float64)
    for k, (l, u) in box_dict.items():
        lb[k] = l
        ub[k] = u
    return lb, ub


def _write_results(results_file, payload):
    """Write the results JSON. First line is the verdict word so the
    αβ-CROWN/NeuralSAT-style ``_read_verdict`` parser sees it first."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # Replace NaN with None so the body is strict-JSON-valid
    sanitised = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in payload.items()
    }
    text = (
        payload['verdict'].lower() + '\n'
        + json.dumps(sanitised, indent=2)
    )
    results_file.write_text(text, encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx_path', required=True, type=Path)
    p.add_argument('--vnnlib_path', required=True, type=Path)
    p.add_argument('--results_file', required=True, type=Path,
                   help='JSON results file. First line is the verdict.')
    p.add_argument('--timeout', type=int, default=100,
                   help='Internal-budget hint; the outer run_cell.sh '
                        'shell timeout is the single kill switch.')
    p.add_argument('--p-filter', type=float, default=0.0,
                   help='ProbStar reach probability filter '
                        '(0.0 = exact reach).')
    p.add_argument('--lp-solver', type=str, default='gurobi',
                   choices=['gurobi', 'glpk'])
    p.add_argument('--gauss-alpha', type=float, default=2.5,
                   help='Truncation coefficient for the input '
                        'ProbStar Gaussian (sigma = (mu - lb) / alpha).')
    p.add_argument('--unsafe-threshold', type=float, default=0.05,
                   help='Verdict gate: UNSAT if sum p_max < threshold.')
    p.add_argument('--starv-path', type=Path, default=_STARV_PATH,
                   help='Path to the StarV checkout (added to sys.path '
                        'if not already importable).')
    args = p.parse_args()

    payload = {
        'verdict': 'ERROR',
        'p_filter': args.p_filter,
        'lp_solver': args.lp_solver,
        'p_min': float('nan'),
        'p_max': float('nan'),
        'threshold': args.unsafe_threshold,
        'n_disjuncts': 0,
        'error': '',
    }

    sv, err = _import_starv(args.starv_path)
    if sv is None:
        payload['error'] = err
        _write_results(args.results_file, payload)
        sys.exit(1)

    try:
        net = sv['load_onnx_network'](str(args.onnx_path), show=False)
    except Exception as e:
        payload['verdict'] = 'NOT_APPLICABLE'
        # Capture exception class + message; if message is empty (bare
        # `assert`), fall back to traceback-based summary so the CSV
        # row carries something diagnostic.
        msg = str(e).strip()
        if not msg:
            tb = traceback.format_exception_only(type(e), e)
            msg = ''.join(tb).strip()
        payload['error'] = f'load_onnx_network {type(e).__name__}: {msg}'
        _write_results(args.results_file, payload)
        sys.exit(0)

    try:
        n_in, n_out = sv['get_num_inputs_outputs'](str(args.onnx_path))
    except Exception as e:
        payload['error'] = (f'get_num_inputs_outputs '
                            f'{type(e).__name__}: {e}')
        _write_results(args.results_file, payload)
        sys.exit(1)

    try:
        spec_tuples = sv['read_vnnlib_simple'](
            str(args.vnnlib_path), n_in, n_out,
        )
    except Exception as e:
        payload['error'] = (f'read_vnnlib_simple '
                            f'{type(e).__name__}: {e}')
        _write_results(args.results_file, payload)
        sys.exit(1)

    Star = sv['Star']
    ProbStar = sv['ProbStar']
    quanti = sv['quantiVerifyBFS']

    total_p_min = 0.0
    total_p_max = 0.0
    n_disjuncts = 0

    try:
        # ``read_vnnlib_simple`` returns one (box, mat_list, rhs_list)
        # tuple per top-level assert. Within a tuple, mat_list and
        # rhs_list together encode an OR-of-AND clause: each
        # (mat[i], rhs[i]) pair is one AND disjunct, and the spec is
        # the OR over disjuncts. quantiVerifyBFS handles one disjunct
        # (mat_i, rhs_i) at a time; we sum p_min/p_max across disjuncts
        # by Bonferroni-union bound (conservative — overlap is double-
        # counted, but a SAFE result requires every disjunct to be safe
        # individually so the bound is sound).
        for (box_dict, mat_list, rhs_list) in spec_tuples:
            lb, ub = _box_dict_to_lb_ub(box_dict, n_in)
            S = Star(lb, ub)
            mu = 0.5 * (S.pred_lb + S.pred_ub)
            sig = (mu - S.pred_lb) / args.gauss_alpha
            Sig = np.diag(np.square(sig))
            P = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)

            for (G, g) in zip(mat_list, rhs_list):
                G_np = np.asarray(G, dtype=np.float64)
                g_np = np.asarray(g, dtype=np.float64).flatten()
                _S_out, _, _, _p_lb, _p_ub, p_min, p_max = quanti(
                    net, [P], G_np, g_np,
                    p_filter=args.p_filter,
                    lp_solver=args.lp_solver,
                    numCores=1, show=False,
                )
                total_p_min += float(p_min)
                total_p_max += float(p_max)
                n_disjuncts += 1
    except Exception as e:
        payload['error'] = (f'quantiVerifyBFS '
                            f'{type(e).__name__}: {e}')
        payload['n_disjuncts'] = n_disjuncts
        # Stash full traceback to stderr for forensics
        traceback.print_exc(file=sys.stderr)
        _write_results(args.results_file, payload)
        sys.exit(1)

    payload['p_min'] = total_p_min
    payload['p_max'] = total_p_max
    payload['n_disjuncts'] = n_disjuncts

    if total_p_max < args.unsafe_threshold:
        payload['verdict'] = 'UNSAT'
    elif total_p_min > args.unsafe_threshold:
        payload['verdict'] = 'SAT'
    else:
        payload['verdict'] = 'UNKNOWN'

    _write_results(args.results_file, payload)
    sys.exit(0)


if __name__ == '__main__':
    main()
