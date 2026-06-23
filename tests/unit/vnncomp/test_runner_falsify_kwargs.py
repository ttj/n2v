"""Tests for the production runner's per-method falsify-budget whitelist.

The runner passes n_samples/method/seed to falsify() EXPLICITLY, so any
falsify_kwargs from a category config must be filtered to budget knobs only —
otherwise falsify() raises "multiple values for argument". This pins that
contract so a future config (or a new budget knob) can't silently break it.
"""

import os
import sys

# The production runner lives under examples/Submission and inserts its own
# sys.path for benchmark_configs at import time.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'examples', 'Submission', 'VNN_COMP2026'))
import vnncomp_runner as R  # noqa: E402


def test_whitelist_keeps_budget_knobs():
    src = {'n_iters': 20000, 'n_restarts': 1, 'n_steps': 30,
           'batch': 128, 'p_init': 0.3, 'step_size': 0.01}
    assert R._whitelist_falsify_kwargs(src) == src


def test_whitelist_drops_explicit_and_unknown_keys():
    # n_samples/method/seed are passed explicitly to falsify() -> must be dropped
    # (else a duplicate-argument TypeError); unknown keys dropped too.
    out = R._whitelist_falsify_kwargs(
        {'n_iters': 5000, 'n_samples': 999, 'method': 'square',
         'seed': 7, 'bogus': 1})
    assert out == {'n_iters': 5000}


def test_whitelist_none_and_empty():
    assert R._whitelist_falsify_kwargs(None) == {}
    assert R._whitelist_falsify_kwargs({}) == {}
