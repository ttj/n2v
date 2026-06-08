"""Randomized Smoothing runner (Cohen et al. 2019, locuslab/smoothing).

Calls ``Smooth.certify`` from ``~/v/other/smoothing/code/core.py``. RS
is fundamentally a *classifier-only* certifier — it returns a class
prediction plus an L2 certified radius. Verdict mapping:

    UNSAT  — predicted class equals the spec's true class AND certified
             radius >= verification budget ε_input. ("model is robust
             within the requested L∞ ball" — note: RS gives an L2
             radius; we use ``L2_radius >= sqrt(d) * eps_inf`` as a
             *necessary* condition for L∞-ε robustness, which is
             conservative (under-claiming UNSAT). The user should
             interpret this as "RS could not certify within the L∞
             budget" when UNSAT does not hold.)
    UNKNOWN — RS abstained or radius below the threshold.
    NOT_APPLICABLE — non-classification benchmark (regression nets,
             OR-of-input-regions like ACAS Xu, etc.).

The runner reads the spec to determine the true class. For
``make_classification_robustness_spec``-style halfspace lists (Exp 2)
this is straightforward: the spec rows have ``1`` in the true-class
column. For VNN-COMP ``robustness`` benchmarks the spec is parsed
similarly; non-classification benchmarks are skipped as
NOT_APPLICABLE.

Pretrained weights expected at:
  ``~/v/other/smoothing/models/cifar10/resnet110/noise_<sigma>/checkpoint.pth.tar``

If missing, the cifar10_resnet110 instance loader raises FileNotFoundError
and the runner emits a TODO and exits gracefully.

Usage:
    cd /path/to/n2v
    python -u -m \\
        examples.FlowConformal.experiments.baselines.run_rs \\
        --benchmark cifar10_resnet110 --smoke
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

from examples.FlowConformal.experiments.baselines._common import (
    add_common_args, load_benchmark_instances, resolve_n_instances,
    resolve_output_csv, run_baseline_sweep,
)


_BASELINE = 'rs'
_RS_PATH = Path(os.path.expanduser('~/v/other/smoothing/code'))
_DEFAULT_SIGMA = 0.25
_DEFAULT_N0 = 100
_DEFAULT_N = 10_000
_DEFAULT_ALPHA = 0.001
_DEFAULT_BATCH = 400
_EPS_LINF = 8.0 / 255.0  # standard CIFAR-10 budget


def _try_import_smooth():
    """Try to import ``Smooth`` from the RS repo.

    The upstream ``core.py`` imports ``binom_test`` from ``scipy.stats``
    (removed in scipy >= 1.12) and ``proportion_confint`` from
    ``statsmodels`` (which may not be installed). We provide shims for
    both so the import succeeds; the ``certify`` path uses both.

    Returns the ``Smooth`` class or None on failure.
    """
    if str(_RS_PATH) not in sys.path:
        sys.path.insert(0, str(_RS_PATH))
    # scipy.stats.binom_test shim (removed in scipy >= 1.12)
    try:
        import scipy.stats as _sst
        if not hasattr(_sst, 'binom_test'):
            try:
                from scipy.stats import binomtest as _binomtest
                _sst.binom_test = lambda x, n, p: _binomtest(x, n, p).pvalue
            except Exception:
                _sst.binom_test = lambda x, n, p: 1.0
    except Exception:
        pass
    # statsmodels.stats.proportion shim if missing
    try:
        import importlib
        try:
            importlib.import_module('statsmodels.stats.proportion')
        except ImportError:
            import types
            from scipy.stats import beta as _beta
            shim_mod = types.ModuleType('statsmodels.stats.proportion')

            def _proportion_confint(count, nobs, alpha=0.05, method='beta'):
                # Clopper-Pearson lower / upper bounds.
                if count == 0:
                    lo = 0.0
                else:
                    lo = float(_beta.ppf(alpha / 2, count, nobs - count + 1))
                if count == nobs:
                    hi = 1.0
                else:
                    hi = float(_beta.ppf(1 - alpha / 2,
                                          count + 1, nobs - count))
                return (lo, hi)

            shim_mod.proportion_confint = _proportion_confint
            stats_pkg = types.ModuleType('statsmodels.stats')
            sm_pkg = types.ModuleType('statsmodels')
            sys.modules.setdefault('statsmodels', sm_pkg)
            sys.modules.setdefault('statsmodels.stats', stats_pkg)
            sys.modules['statsmodels.stats.proportion'] = shim_mod
    except Exception:
        pass
    try:
        from core import Smooth  # type: ignore
        return Smooth, None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def _extract_true_class_from_spec(spec) -> int | None:
    """Return the true class index for a classification-robustness spec.

    Accepts three shapes:

    1. **Exp 2 ``make_classification_robustness_spec``** —
       ``list[HalfSpace]`` where each disjunct's row is ``e_true - e_j``
       (single +1 / single -1, rhs 0).
    2. **VNN-COMP packed format** (e.g. ``cifar100_2024``) —
       ``list[dict]`` where each dict has key ``'Hg'`` mapping to a
       list of single-row HalfSpaces with the same +1/-1 row pattern.
    3. **Single multi-row HalfSpace** — every row has +1 in the same
       true-class column.

    Returns the column index that holds +1 across all halfspaces, or
    ``None`` if the spec doesn't fit a classification-robustness shape.
    """
    from n2v.sets.halfspace import HalfSpace

    # Normalise to a flat list of HalfSpaces.
    halfspaces: list = []
    if isinstance(spec, HalfSpace):
        halfspaces.append(spec)
    elif isinstance(spec, list):
        for elem in spec:
            if isinstance(elem, HalfSpace):
                halfspaces.append(elem)
            elif isinstance(elem, dict) and 'Hg' in elem:
                hg = elem['Hg']
                if isinstance(hg, list):
                    halfspaces.extend(h for h in hg if isinstance(h, HalfSpace))
                elif isinstance(hg, HalfSpace):
                    halfspaces.append(hg)
    if not halfspaces:
        return None

    candidate = None
    for hs in halfspaces:
        G = np.asarray(hs.G, dtype=np.float64)
        # Walk rows independently — a packed multi-row HalfSpace also
        # works as long as every row's +1 sits in the same column.
        for row in G:
            plus_one = np.where(np.isclose(row, 1.0))[0]
            if plus_one.size != 1:
                return None
            c = int(plus_one[0])
            if candidate is None:
                candidate = c
            elif candidate != c:
                return None
    return candidate


def _infer_num_classes(spec, *, default: int = 10) -> int:
    """Infer logit-output dim from a classification-robustness spec.

    For ``list[dict]`` specs (VNN-COMP packed), reads the dict's
    ``'dim'`` field. For ``list[HalfSpace]`` specs, reads the
    HalfSpace's ``G.shape[1]``. Falls back to ``default``.
    """
    from n2v.sets.halfspace import HalfSpace

    if isinstance(spec, list):
        for elem in spec:
            if isinstance(elem, dict) and 'dim' in elem:
                return int(elem['dim'])
            if isinstance(elem, HalfSpace):
                return int(np.asarray(elem.G).shape[1])
    elif isinstance(spec, HalfSpace):
        return int(np.asarray(spec.G).shape[1])
    return default


def _process_factory(args):
    Smooth, import_err = _try_import_smooth()
    if Smooth is None:
        # Returns a function that emits ERROR for every instance.
        def _err(loader, name):
            return {'verdict': 'ERROR',
                    'error': f'rs_import_failed: {import_err}'}
        return _err

    sigma = args.sigma
    n0 = args.n0
    n_certify = args.n_certify
    alpha = args.alpha
    batch_size = args.batch_size
    eps = args.eps

    import torch

    def process_one(loader, name):
        try:
            net, boxes, spec, _ = loader()
        except FileNotFoundError as e:
            return {'verdict': 'ERROR', 'error': f'load_missing: {e}'}
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'load {type(e).__name__}: {e}'}

        # RS only handles classification + single input box.
        true_c = _extract_true_class_from_spec(spec)
        if true_c is None:
            return {'verdict': 'NOT_APPLICABLE',
                    'error': 'spec is not classification-robustness'}
        if len(boxes) != 1:
            return {'verdict': 'NOT_APPLICABLE',
                    'error': 'OR-of-input-regions not supported by RS'}

        lb, ub = boxes[0]
        lb = np.asarray(lb, dtype=np.float32).flatten()
        ub = np.asarray(ub, dtype=np.float32).flatten()
        # Recover the center image: midpoint of the box (up to floating-
        # point round-off this is the original clean image).
        center = 0.5 * (lb + ub)
        # Image needs to be (C, H, W). The Exp 2 ResNet-110 wrapper
        # accepts (B, 3072) and reshapes internally. RS's certify wants
        # (C, H, W) and prepares its own batch noise. We reshape here.
        d = center.size
        if d == 3072:
            # CIFAR-style 3×32×32 image. Number of classes depends on the
            # benchmark — infer from the spec's classification structure
            # rather than hardcoding 10. CIFAR-10 → 10; cifar100 → 100.
            x = torch.as_tensor(center.reshape(3, 32, 32), dtype=torch.float32)
            num_classes = _infer_num_classes(spec, default=10)
        else:
            return {'verdict': 'NOT_APPLICABLE',
                    'error': f'RS expects CIFAR-10/100 input (3072), got {d}'}

        # The RS implementation calls ``base_classifier(batch + noise)``
        # where ``batch`` has shape (B, C, H, W) — it does NOT pass a
        # flat (B, 3072). Wrap the network to accept (B, C, H, W) by
        # flattening internally if needed (the cifar10_resnet110 wrapper
        # already handles both (B, 2-D) and (B, C, H, W) inputs).
        # Move to CUDA if available; the RS code uses ``device='cuda'``
        # for noise, which fails on CPU-only machines.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            net_dev = net.to(device).eval()
        except Exception:
            net_dev = net.eval()
        x = x.to(device)

        # Patch the RS hard-coded ``device='cuda'`` if running on CPU by
        # monkey-patching ``torch.randn_like`` calls is too invasive.
        # Instead, when no CUDA is available, fall back to a Python-only
        # certify reimplementation (same logic, no device hard-code).
        try:
            if device == 'cuda':
                smooth = Smooth(net_dev, num_classes=num_classes, sigma=sigma)
                pred, radius = smooth.certify(x, n0=n0, n=n_certify,
                                              alpha=alpha,
                                              batch_size=batch_size)
            else:
                pred, radius = _certify_cpu(
                    net_dev, x, sigma=sigma, num_classes=num_classes,
                    n0=n0, n=n_certify, alpha=alpha, batch_size=batch_size,
                )
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'rs_certify {type(e).__name__}: {e}'}

        ABSTAIN = -1
        # L∞ ε relates to L2 radius: an L2 ball of radius r contains the
        # L∞ ball of radius r/sqrt(d). To certify L∞-ε we need
        # ``radius >= sqrt(d) * eps`` (sufficient, not necessary). This
        # is the conservative direction.
        threshold_l2 = math.sqrt(d) * eps

        if pred == ABSTAIN:
            verdict = 'UNKNOWN'
        elif pred != true_c:
            # RS predicts a different class than the spec's true class.
            verdict = 'SAT'
        elif radius >= threshold_l2:
            verdict = 'UNSAT'
        else:
            verdict = 'UNKNOWN'

        return {
            'verdict': verdict,
            'sigma': sigma,
            'n0': n0,
            'n_certify': n_certify,
            'alpha': alpha,
            'pred_class': int(pred),
            'true_class': int(true_c),
            'l2_radius': float(radius),
            'eps_linf_threshold_l2': float(threshold_l2),
            'error': '',
        }

    return process_one


def _certify_cpu(network, x, *, sigma, num_classes, n0, n, alpha,
                  batch_size):
    """CPU fallback for ``Smooth.certify`` — same logic as ``core.py``
    but without the hard-coded ``device='cuda'`` for noise generation.
    """
    import torch
    from math import ceil
    from scipy.stats import norm
    from statsmodels.stats.proportion import proportion_confint

    network.eval()

    def _sample_noise(num):
        counts = np.zeros(num_classes, dtype=int)
        with torch.no_grad():
            remaining = num
            while remaining > 0:
                bsz = min(batch_size, remaining)
                remaining -= bsz
                batch = x.repeat((bsz, 1, 1, 1))
                noise = torch.randn_like(batch) * sigma
                preds = network(batch + noise).argmax(1)
                idx = preds.cpu().numpy()
                for c in idx:
                    counts[int(c)] += 1
        return counts

    counts0 = _sample_noise(n0)
    cAHat = int(counts0.argmax())
    counts = _sample_noise(n)
    nA = int(counts[cAHat])
    pABar = proportion_confint(nA, n, alpha=2 * alpha, method='beta')[0]
    if pABar < 0.5:
        return -1, 0.0
    radius = sigma * norm.ppf(pABar)
    return cAHat, float(radius)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--sigma', type=float, default=_DEFAULT_SIGMA,
                        help='RS noise level (default 0.25).')
    parser.add_argument('--n0', type=int, default=_DEFAULT_N0,
                        help='RS selection samples (default 100).')
    parser.add_argument('--n-certify', type=int, default=_DEFAULT_N,
                        help='RS certification samples (default 10000).')
    parser.add_argument('--alpha', type=float, default=_DEFAULT_ALPHA,
                        help='RS failure probability (default 1e-3).')
    parser.add_argument('--batch-size', type=int, default=_DEFAULT_BATCH,
                        help='RS evaluation batch size (default 400).')
    parser.add_argument('--eps', type=float, default=_EPS_LINF,
                        help='L∞ verification budget (default 8/255).')
    args = parser.parse_args()

    n = resolve_n_instances(args)
    try:
        instances = load_benchmark_instances(args.benchmark, n)
    except FileNotFoundError as e:
        print(f'[{_BASELINE}] TODO/load failed: {e}', file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f'[{_BASELINE}] load error: {type(e).__name__}: {e}',
              file=sys.stderr)
        sys.exit(2)
    if not instances:
        print(f'[{_BASELINE}] no instances', file=sys.stderr)
        sys.exit(0)

    out_csv = resolve_output_csv(args, _BASELINE)
    extra_fields = [
        'sigma', 'n0', 'n_certify', 'alpha',
        'pred_class', 'true_class', 'l2_radius', 'eps_linf_threshold_l2',
    ]
    run_baseline_sweep(
        baseline=_BASELINE, benchmark=args.benchmark,
        instances=instances, out_csv=out_csv,
        extra_fields=extra_fields,
        process_one=_process_factory(args),
    )


if __name__ == '__main__':
    main()
