"""Verification-method ablation with shared (flow, q) per instance.

Improvement over :mod:`ablation_run_verify_method`: instead of training
a fresh flow per method (where flow randomness contaminates the
verifier-quality signal), this runner calibrates once per instance and
runs all 8 verifiers against the *same* ``(flow, q)`` tuple. This
isolates the verification-method axis as the only varying dimension.

Probe: 20 instances *randomly sampled* (seed=47) from the full 186
ACAS Xu instances. ``use_falsifier=False`` is hard-coded — the point
is to test whether each verifier (a) reports UNSAT for ground-truth
UNSAT and (b) reports UNKNOWN for ground-truth SAT (no SAT short-
circuit allowed).

Outputs one CSV per method under
``examples/FlowConformal/experiments/exp_ablation/outputs/`` with
filename ``ablation_shared_flow_<method>.csv``. The shared-flow
filenames are prefix-distinct from the older
``ablation_verify_method_<method>.csv`` so the previous results are
preserved.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from n2v.nn import NeuralNetwork
from n2v.probabilistic import FlowReachConfig
from n2v.sets import Box
from n2v.utils.verify_specification import (
    ProbVerifyConfig,
    verify_specification,
)

_OUT_DIR = Path(__file__).resolve().parent / 'outputs'
_PROBE_SEED = 47
_DEFAULT_N_INSTANCES = 20

_METHODS = (
    'scenario', 'amls', 'is_tilted',
    'amls_bounded', 'amls_bounded_union', 'raw_mc_uniform',
)

_SUPPORTED_BENCHMARKS = ('acasxu_2023', 'tllverify_2023')

# Per-benchmark calibration / verifier hparams, copied from the
# production exp1_run_ours config so the ablation matches what the
# headline numbers were produced with. Falsifier is intentionally
# OFF (the ablation tests the verifiers' UNSAT certification, not
# the SAT short-circuit).
_BENCHMARK_HPARAMS: dict[str, dict] = {
    'acasxu_2023': dict(
        calib=dict(alpha=0.001, n_train=5_000, flow_epochs=2_000,
                   flow_config='base'),
        verify=dict(scenario_n_samples=2_000, scenario_beta=0.001),
    ),
    'tllverify_2023': dict(
        calib=dict(alpha=0.001, n_train=10_000, flow_epochs=2_000,
                   flow_config='base'),
        verify=dict(scenario_n_samples=2_000, scenario_beta=0.001),
    ),
}

_FIELDS = [
    'instance_idx', 'box_idx', 'onnx_file', 'vnnlib_file', 'ground_truth',
    'method', 'verdict', 'q', 'coverage', 'epsilon_total',
    'amls_bounded_eps_2_upper', 'flow_train_s', 'verify_s',
    'wall_s', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _fmt(v, spec):
    if v is None:
        return ''
    try:
        if isinstance(v, float) and not np.isfinite(v):
            return ''
    except TypeError:
        pass
    return f'{v:{spec}}'


def _sample_instances(benchmark: str, n: int,
                      seed: int) -> List[Tuple[str, str, int]]:
    """Return ``n`` randomly-sampled instances for the given benchmark.

    When ``n >= len(full)`` returns the full set in deterministic
    seed-shuffled order so output ordering stays stable across reruns.
    """
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
        list_instances,
    )
    full = list(list_instances(benchmark))
    rng = random.Random(seed)
    if n >= len(full):
        rng.shuffle(full)
        return full
    return rng.sample(full, n)


def _load_instance(benchmark: str, onnx_rel: str, vnn_rel: str):
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
        load_one_instance,
    )
    return load_one_instance(benchmark, onnx_rel, vnn_rel)


def _gt_for(benchmark: str, onnx_rel: str, vnn_rel: str) -> str:
    from examples.FlowConformal.experiments._ground_truth_lookup import (
        lookup_ground_truth,
    )
    onnx_name = Path(onnx_rel).name
    vnn_name = Path(vnn_rel).name
    inst = f'{onnx_name} + {vnn_name}'
    try:
        gt = lookup_ground_truth('exp1', benchmark, inst)
    except Exception:
        return ''
    return (gt or '').lower()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=_SUPPORTED_BENCHMARKS,
                   default='acasxu_2023',
                   help='Which Exp 1 benchmark to draw instances from.')
    p.add_argument('--smoke', action='store_true',
                   help='run a single instance × all methods (sanity check).')
    p.add_argument('--n-instances', type=int, default=_DEFAULT_N_INSTANCES,
                   help='Number of instances to sample. Pass a number ≥ '
                        'the benchmark size to run the full benchmark.')
    p.add_argument('--instance-seed', type=int, default=_PROBE_SEED,
                   help='RNG seed for the random-instance sampler.')
    p.add_argument('--methods', nargs='+', choices=_METHODS,
                   default=list(_METHODS))
    p.add_argument('--output-prefix', type=str,
                   default=None,
                   help='CSV filename prefix; one CSV per method as '
                        '<prefix>_<method>.csv. Default = '
                        '"ablation_shared_flow_<benchmark>".')
    args = p.parse_args()

    benchmark = args.benchmark
    output_prefix = (
        args.output_prefix
        if args.output_prefix is not None
        else f'ablation_shared_flow_{benchmark}'
    )
    n_inst = 1 if args.smoke else args.n_instances
    instances = _sample_instances(benchmark, n_inst, args.instance_seed)

    print(f'[shared-flow ablation] benchmark={benchmark} '
          f'sampled {len(instances)} instance(s) '
          f'with seed={args.instance_seed}')
    for k, (onnx, vnn, _) in enumerate(instances):
        print(f'  {k:2d}: {Path(onnx).name} + {Path(vnn).name}')
    print(f'[shared-flow ablation] methods: {list(args.methods)}')

    hp = _BENCHMARK_HPARAMS[benchmark]
    calib_kwargs = hp['calib']
    verify_kwargs = hp['verify']
    print(f'[shared-flow ablation] calib hparams: {calib_kwargs}')
    print(f'[shared-flow ablation] verify hparams: {verify_kwargs}')

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Per-method CSV writers; opened once and reused across the whole
    # instance loop so each method's CSV grows row-by-row in lockstep.
    files = {}
    writers = {}
    for method in args.methods:
        out_csv = _OUT_DIR / f'{output_prefix}_{method}.csv'
        files[method] = open(out_csv, 'w', newline='')
        w = csv.DictWriter(files[method], fieldnames=_FIELDS)
        w.writeheader()
        files[method].flush()
        writers[method] = w

    t_start = time.time()
    try:
        for inst_idx, (onnx_rel, vnn_rel, _) in enumerate(instances):
            onnx_name = Path(onnx_rel).name
            vnn_name = Path(vnn_rel).name
            gt = _gt_for(benchmark, onnx_rel, vnn_rel)
            elapsed = time.time() - t_start
            print(f'\n[{inst_idx+1}/{len(instances)} t={elapsed:.0f}s] '
                  f'{onnx_name} + {vnn_name}  gt={gt or "?"}')

            try:
                network, boxes, spec = _load_instance(
                    benchmark, onnx_rel, vnn_rel)
            except Exception as e:
                err = f'load {type(e).__name__}: {e}'
                print(f'  ERROR loading: {err}')
                for method in args.methods:
                    row = {f: '' for f in _FIELDS}
                    row.update({
                        'instance_idx': inst_idx, 'box_idx': 0,
                        'onnx_file': onnx_name, 'vnnlib_file': vnn_name,
                        'ground_truth': gt, 'method': method,
                        'verdict': 'ERROR', 'error': err,
                        'timestamp': _now_iso(),
                    })
                    writers[method].writerow(row); files[method].flush()
                continue

            # OR-of-input-regions specs (e.g. ACAS Xu prop_3, prop_4)
            # come back as multiple boxes; treat each as an independent
            # row. Calibrate once per box, then loop verifiers.
            for box_idx, (lb, ub) in enumerate(boxes):
                print(f'  box {box_idx}/{len(boxes)-1}')
                t0 = time.time()
                try:
                    net_wrapped = NeuralNetwork(network)
                    input_box = Box(
                        np.asarray(lb).flatten(),
                        np.asarray(ub).flatten(),
                    )
                    prob_set = net_wrapped.reach(
                        input_box, method='flow_matching',
                        config=FlowReachConfig(
                            epsilon=calib_kwargs['alpha'],
                            n_train=calib_kwargs['n_train'],
                            flow_epochs=calib_kwargs['flow_epochs'],
                            flow_config=calib_kwargs['flow_config'],
                            seed=_PROBE_SEED,
                        ),
                    )
                    cov = prob_set.estimate_coverage(
                        network, input_box, n_test=2_000,
                        seed=_PROBE_SEED + 2_000_000,
                    )
                except Exception as e:
                    err = f'calibrate {type(e).__name__}: {e}'
                    print(f'    ERROR calibrating: {err}')
                    for method in args.methods:
                        row = {f: '' for f in _FIELDS}
                        row.update({
                            'instance_idx': inst_idx, 'box_idx': box_idx,
                            'onnx_file': onnx_name, 'vnnlib_file': vnn_name,
                            'ground_truth': gt, 'method': method,
                            'verdict': 'ERROR', 'error': err,
                            'timestamp': _now_iso(),
                        })
                        writers[method].writerow(row); files[method].flush()
                    continue
                calib_s = time.time() - t0
                q = prob_set.threshold
                print(f'    calibrate ok in {calib_s:.1f}s  '
                      f'q={q:.4f}  cov={cov:.4f}')

                # AMLS-bounded / raw-MC require an eps_2 target; legacy
                # ablation pipeline defaulted this to alpha. Mirror that
                # as a fallback, but honor an explicit override so the η
                # ablation knob is actually live.
                eps_2_target = calib_kwargs.get(
                    'amls_bounded_eps_2_target', calib_kwargs['alpha'])

                for method in args.methods:
                    tv = time.time()
                    try:
                        result = verify_specification(
                            prob_set, spec,
                            config=ProbVerifyConfig(
                                method=method,
                                n_samples=verify_kwargs['scenario_n_samples'],
                                beta=verify_kwargs['scenario_beta'],
                                seed=_PROBE_SEED,
                                amls_bounded_eps_2_target=eps_2_target,
                            ),
                        )
                        verdict = result.verdict
                        eps_total = result.epsilon_total
                        eps_2_b = result.amls_bounded_eps_2_upper
                        err = ''
                    except Exception as e:
                        verdict = 'ERROR'
                        eps_total = None
                        eps_2_b = None
                        err = f'verify {type(e).__name__}: {e}'
                    v_s = time.time() - tv
                    wall_s = calib_s + v_s

                    row = {f: '' for f in _FIELDS}
                    row.update({
                        'instance_idx': inst_idx,
                        'box_idx': box_idx,
                        'onnx_file': onnx_name,
                        'vnnlib_file': vnn_name,
                        'ground_truth': gt,
                        'method': method,
                        'verdict': verdict,
                        'q': _fmt(q, '.6f'),
                        'coverage': _fmt(cov, '.4f'),
                        'epsilon_total': _fmt(eps_total, '.4e'),
                        'amls_bounded_eps_2_upper': _fmt(eps_2_b, '.4e'),
                        'flow_train_s': _fmt(calib_s, '.1f'),
                        'verify_s': _fmt(v_s, '.1f'),
                        'wall_s': _fmt(wall_s, '.1f'),
                        'error': err,
                        'timestamp': _now_iso(),
                    })
                    writers[method].writerow(row); files[method].flush()
                    print(f'      {method:<22} {verdict:<8} '
                          f'verify={v_s:.1f}s')

    finally:
        for f in files.values():
            f.close()

    total_min = (time.time() - t_start) / 60
    print(f'\n=== shared-flow ablation complete: {total_min:.1f} min ===')
    print(f'Outputs: {_OUT_DIR}/{output_prefix}_<method>.csv')


if __name__ == '__main__':
    main()
