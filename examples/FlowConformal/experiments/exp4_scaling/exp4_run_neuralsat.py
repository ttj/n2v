"""Exp 4 — NeuralSAT runner on synthetic-network family.

One script invocation iterates the 10 instances at a given depth and
writes one CSV row per instance under
``examples/FlowConformal/experiments/exp4_scaling/outputs/exp4_d<D>_neuralsat.csv``.

Calls NeuralSAT via :mod:`examples.FlowConformal.experiments._external_verifiers`
(subprocess to its own conda env). Per-instance timeout is fixed at
600s per the README's Exp 4 spec.

Usage::

    cd /path/to/n2v

    # Smoke (1 instance at depth 2):
    python -m \\
        examples.FlowConformal.experiments.exp4_scaling.exp4_run_neuralsat \\
        --depth 2 --smoke

    # Full sweep at depth 16:
    nohup python -u -m \\
        examples.FlowConformal.experiments.exp4_scaling.exp4_run_neuralsat \\
        --depth 16 \\
        > examples/FlowConformal/experiments/exp4_scaling/outputs/exp4_d16_neuralsat.log 2>&1 &
    disown

Each instance has ``ground_truth='unsat'`` by construction (1-Lipschitz
networks with specs at empirical-max + 0.1). A ``verdict='SAT'``
indicates a soundness violation in NeuralSAT — should never happen on
this family.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

from examples.FlowConformal.experiments._external_verifiers import (
    run_neuralsat,
)
from examples.FlowConformal.experiments._runner_utils import (
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments.exp4_scaling._benchmarks import (
    load_instances,
)
from examples.FlowConformal.experiments.exp4_scaling.networks import (
    EXP4_DEPTHS,
    EXP4_WIDTH,
)

_TIMEOUT_S = 300  # reduced from 600s; BaB scaling failure visible within 60s, matches VNN-COMP standard for similar benchmarks
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'depth', 'width', 'n_params', 'instance_idx', 'method',
    'verdict', 'wall_s', 'timeout_s', 'x_0_seed', 'eps',
    'spec_threshold', 'empirical_max', 'ground_truth',
    'amls_bounded_eps_2_upper', 'amls_levels_used',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _write_timeout_row(out_csv, depth, instance_idx, timeout_s):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'depth': depth, 'instance_idx': instance_idx,
        'method': 'neuralsat', 'verdict': 'TIMEOUT',
        'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, required=True, choices=EXP4_DEPTHS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only this 0-based index; APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <timeout_s>" per instance, exit.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    instances = load_instances(args.depth)

    if args.list_instances:
        for idx in range(len(instances)):
            print(f'{idx} {_TIMEOUT_S}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp4_d{args.depth}_neuralsat.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        _write_timeout_row(out_csv, args.depth, args.instance_idx, _TIMEOUT_S)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        instances = [instances[args.instance_idx]]
        append_mode = True
        print(f'[exp4 d={args.depth}] running only idx={args.instance_idx}; '
              f'appending to {out_csv}', flush=True)
    elif args.smoke:
        instances = instances[:1]
        print('[smoke] Running only the first instance', flush=True)
    print(f'Loaded {len(instances)} instances at depth={args.depth} '
          f'(width={EXP4_WIDTH})', flush=True)
    print(f'Tool: NeuralSAT  device={args.device}  timeout={_TIMEOUT_S}s',
          flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader()
            f.flush()

        for k, inst in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            print(f'[{k}/{len(instances)}  t={elapsed:.0f}s  '
                  f'budget={_TIMEOUT_S}s] depth={inst["depth"]} '
                  f'idx={inst["instance_idx"]} '
                  f'C={inst["C"]:.4f}',
                  flush=True)

            tag = f'exp4_d{inst["depth"]}_i{inst["instance_idx"]}'
            # Disable NeuralSAT's PGD pre-attack to match αβ-CROWN's
            # `attack: pgd_order: skip` and our pipeline's
            # `use_falsifier=False`. Exp 4 is UNSAT-by-construction so
            # the attack would only burn time finding nothing.
            verdict, wall_s, err = run_neuralsat(
                onnx_path=inst['onnx_path'],
                vnnlib_path=inst['vnnlib_path'],
                timeout_s=_TIMEOUT_S,
                device=args.device,
                instance_tag=tag,
                extra_args=['--disable_attack'],
            )

            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'depth': inst['depth'],
                'width': EXP4_WIDTH,
                'n_params': inst['n_params'],
                'instance_idx': inst['instance_idx'],
                'method': 'neuralsat',
                'verdict': verdict,
                'wall_s': f'{wall_s:.2f}' if wall_s is not None else '',
                'timeout_s': _TIMEOUT_S,
                'x_0_seed': inst['instance_seed'],
                'eps': f'{inst["eps"]:.4f}',
                'spec_threshold': f'{inst["C"]:.6f}',
                'empirical_max': f'{inst["empirical_max"]:.6f}',
                'ground_truth': inst['ground_truth'],
                'error': err,
                'timestamp': _now_iso(),
            })
            writer.writerow(out_row)
            f.flush()
            counts[verdict] = counts.get(verdict, 0) + 1
            print(f'    verdict={verdict}  wall={wall_s:.1f}s  err={err!r}',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        if counts.get('ERROR', 0) > 0 or counts.get('TIMEOUT', 0) > 0:
            print(f'[smoke] FAIL: pipeline errored or timed out, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        if counts.get('SAT', 0) > 0:
            print(f'[smoke] FAIL: NeuralSAT returned SAT on a '
                  f'UNSAT-by-construction instance — likely spec/config '
                  f'bug. counts={counts}', file=sys.stderr)
            sys.exit(2)
        print('[smoke] PASS: NeuralSAT ran end-to-end without errors.')


if __name__ == '__main__':
    main()
