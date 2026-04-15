# v2 Hashemi Comparison — Adaptive Sinkhorn reg Baseline

This directory contains the Hashemi comparison re-run with the adaptive Sinkhorn `reg` bug fix applied. It is the corrected baseline for the main paper's flow vs clipping-block comparison.

## What changed from v1

The ONLY difference from [`v1_original_buggy/`](../v1_original_buggy/) is that this script does not pin `sinkhorn_reg` or `sinkhorn_iters` at the `train_flow` call, so it uses the library's current defaults:

- `sinkhorn_reg='auto'` → adaptive reg via `compute_adaptive_sinkhorn_reg` (α=0.1, chosen in the [alpha sweep audit](../../../../../../docs/audits/2026-04-14-adaptive-sinkhorn-alpha-sweep.md))
- `sinkhorn_iters=50` → same as v1

Everything else is byte-identical: seeds, network initialization, test inputs, radii, flow config, sample counts, conformal parameters, etc. The A/B is strictly between the buggy `reg=0.05` and the adaptive reg fix.

## Bug context

See [`docs/audits/2026-04-13-hashemi-clipping-block-audit.md`](../../../../../../docs/audits/2026-04-13-hashemi-clipping-block-audit.md) for the underflow analysis and [`docs/plans/2026-04-14-adaptive-sinkhorn-reg-design.md`](../../../../../../docs/plans/2026-04-14-adaptive-sinkhorn-reg-design.md) for the design.

## How to run

```
/home/sasakis/miniconda3/envs/n2v/bin/python exp_hashemi_comparison.py
```

Expected runtime: ~50 minutes on CPU, same as v1. Produces `exp_hashemi_comparison.csv` and 5 figures in this directory.

## Expected outcome vs v1

Based on the earlier diagnostic benchmarks in the alpha sweep audit, the expected improvements in v2 vs v1 are:

- **Banana network at all radii**: approximately unchanged (the data scale was already small enough that reg=0.05 happened to work)
- **Classifier at small radii (0.05–0.25)**: approximately unchanged
- **Classifier at large radius (1.0)**: flow volume drops by ~2.75x due to Sinkhorn properly converging. The "Anomaly 2" discontinuity in v1's scaling curve should disappear.

If v2 does NOT show the expected improvement at classifier r=1.0, it suggests either (a) the alpha sweep's extrapolation was wrong, (b) a different bug is in play, or (c) the training protocol needs further tuning.
