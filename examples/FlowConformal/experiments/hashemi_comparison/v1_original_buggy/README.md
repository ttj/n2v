# v1 Hashemi Comparison — Original Buggy Baseline

This directory contains the original Hashemi comparison run that was completed on 2026-04-13 before the adaptive Sinkhorn `reg` fix. It is preserved as an archival snapshot for before/after comparison.

## Known bug

The flow training pipeline used a hardcoded `sinkhorn_reg=0.05` that numerically underflows when the output scale exceeds the small 2D banana. At classifier r=1.0, the Sinkhorn kernel `K_ij = exp(-cost/reg)` underflowed almost everywhere, producing degenerate nearest-neighbor coupling instead of true optimal transport. The resulting flow at that config is ~2.75× looser than it should be.

Full details: see [`docs/audits/2026-04-14-adaptive-sinkhorn-alpha-sweep.md`](../../../../../../docs/audits/2026-04-14-adaptive-sinkhorn-alpha-sweep.md) for the sweep that characterized the fix, and [`docs/plans/2026-04-14-adaptive-sinkhorn-reg-design.md`](../../../../../../docs/plans/2026-04-14-adaptive-sinkhorn-reg-design.md) for the design rationale.

## How to reproduce

After Task 8 lands, the frozen script in this directory passes `sinkhorn_reg=0.05, sinkhorn_iters=50` explicitly to `train_flow`, so running it reproduces the buggy behavior byte-for-byte (modulo RNG determinism) even though the library's default is now adaptive.

```
/home/sasakis/miniconda3/envs/n2v/bin/python exp_hashemi_comparison.py
```

Expected runtime: ~50 minutes on CPU.

## Why it is here

Paper reviewers may want to see side-by-side numbers from the buggy baseline vs the fixed version. We preserve this archive rather than overwrite it so the audit trail stays intact. Once the final paper is submitted and the bug fix is no longer load-bearing, this archive can be deleted.
