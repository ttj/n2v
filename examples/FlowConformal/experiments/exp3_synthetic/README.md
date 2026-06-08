# Experiment 3: Synthetic Validation

Validates the flow-conformal framework on synthetic benchmarks where the
exact (or near-exact) reach-set volume is known. We compare **OUR
method only** here (verification_method='amls_bounded', Phase 5d locked
config); third-party probabilistic baselines are wired in separately.

See `docs/plans/2026-04-27-paper-experiments-design.md` (Experiment 3).

## Benchmarks

1. **3D banana classifier** (`ThreeBlobClassifier3D`) — 3D inputs in
   [-1, 1]^3, 3D output logits. Exact reach-set volume estimated from
   the cached Star-union (~213.72; (1-α)·V is the tightness floor).
2. **Higher-dim 1-Lipschitz nets** (5D, 10D, 20D) — identity-activation
   nets so the composed map is purely linear and the reach set is a
   zonotope with closed-form volume `|det(W_total)| · prod(ub - lb)`.
   Input box: `[-0.5, 0.5]^dim`. Halfspace query: `y[0] >= 1e6`
   (always UNSAT — we are testing volume tightness, not verdict
   correctness).

## Files

- `networks.py` — `OneLipschitzNet`, plus
  `make_synthetic_2d/3d/5d/10d/20d`.
- `exact_volumes.py` — `exact_volume_linear_net` (closed form for the
  identity-activation case), cached MC reach-set values for the bananas
  (`exact_volume_three_blob_3d`, `exact_volume_two_banana`), and
  `mc_ground_truth_volume` (fallback for nonlinear activations).
- `_benchmarks.py` — benchmark registry + per-benchmark hparam
  overrides (`PER_BENCHMARK_CONFIG`).
- `_score_pipeline.py` — score-family dispatcher used by the ours
  runner (flow / hyperrect / ellipsoid / gmm).
- `exp3_run_ours.py` — flow-conformal pipeline. Takes
  `--benchmark {2d_banana, 3d_banana, synth_2d, synth_3d, synth_5d,
  synth_10d, synth_20d}` × `--score {flow, hyperrect, ellipsoid, gmm}`
  × `--spec {unsat, sat}`.
- `exp3_run_hashemi_clipping.py` — Hashemi-clipping pbox volume
  baseline. Same `--benchmark` / `--spec` axes.
- `exp3_run_starset_approx.py` — sound deterministic baseline using
  `n2v.nn.reach.reach_pytorch_model(method='approx')`.
- `outputs/` — CSVs are written here at runtime.

## Locked Phase 5d config

| param | value |
|---|---|
| verification_method | amls_bounded |
| amls_max_levels | 30 |
| alpha | 0.001 |
| n_train | 5000 (synth_5/10/20d); 2000 (3d_banana) |
| flow_epochs | 2000 |
| flow_config | base (h128/L4) |
| scenario_n_samples | 2000 |
| scenario_beta | 0.001 |

## How to run

Smoke (1 seed, reduced training):

```bash
cd /path/to/n2v
python -u -m \
    examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours \
    --benchmark 3d_banana --score flow --spec unsat --smoke
```

Full sweep across all benchmarks × the three sample-budget configs:

```bash
bash examples/FlowConformal/experiments/run_paper_sweeps.sh --phase exp3
```

## Output schema

See [`../../CSV_SCHEMAS.md` §3](../../CSV_SCHEMAS.md#3-experiment-3--synthetic-volume-comparison)
for the exact column layout per runner. Filenames follow:

- `exp3_<benchmark>_<score>_<spec>_ours[_<config>].csv`
- `exp3_<benchmark>_<spec>_hashemi_clipping[_<config>].csv`
- `exp3_<benchmark>_<spec>_starset_approx.csv`

The exact-volume reference for synth_<N>d is
`|det(W_total)|·prod(ub-lb)` (closed form, identity-activation linear
net). The bananas use cached MC values
(`exact_volume_three_blob_3d`, `exact_volume_two_banana`).
