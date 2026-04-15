# v1 vs v2 Hashemi Comparison Summary

**Change**: adaptive Sinkhorn `reg` fix in [`n2v/probabilistic/flow/train.py`](../../../../../../n2v/probabilistic/flow/train.py). See the [design doc](../../../../../../docs/plans/2026-04-14-adaptive-sinkhorn-reg-design.md) for background and the [α sweep audit](../../../../../../docs/audits/2026-04-14-adaptive-sinkhorn-alpha-sweep.md) for how α=0.1 was chosen.

**A/B scope**: v2 differs from v1 by exactly one knob — `sinkhorn_reg` goes from the buggy hardcoded `0.05` to `'auto'` (adaptive via `compute_adaptive_sinkhorn_reg`). All other hyperparameters, seeds, network initialization, test inputs, radii, and sample counts are byte-identical.

## The bug in one sentence

Hardcoded `sinkhorn_reg=0.05` caused the Sinkhorn kernel `K = exp(-cost/reg)` to numerically underflow when output data scale was moderate or large (e.g., classifier logits at r=1.0 with median squared cost ≈ 8.8 → `cost/reg ≈ 176` → `K ≈ 10⁻⁷⁷`), silently degrading the coupling to near-nearest-neighbor assignment and inflating the calibrated reach-set volume ~2.75× at the worst affected config.

## Headline result

| Network | Radius | v1 mean flow volume | v2 mean flow volume | **v1/v2 ratio** |
|---|---|---|---|---|
| RotatedBananaNet | 0.05 | 0.0160 | 0.0151 | 1.06× |
| RotatedBananaNet | 0.10 | 0.0476 | 0.0431 | 1.10× |
| RotatedBananaNet | 0.20 | 0.1407 | 0.1269 | 1.11× |
| RotatedBananaNet | 0.40 | 0.4608 | 0.4222 | 1.09× |
| ThreeBlobClassifier | 0.10 | 0.1076 | 0.0869 | 1.24× |
| ThreeBlobClassifier | 0.25 | 0.8462 | 0.8123 | 1.05× |
| ThreeBlobClassifier | 0.50 | 5.3636 | 4.6791 | 1.13× |
| **ThreeBlobClassifier** | **1.00** | **150.9276** | **58.9247** | **2.60×** |

**The classifier r=1.0 drop of 2.60× matches the alpha sweep prediction (~2.75×) within noise.** Every other config tightens by 5–24% — not dramatic, but non-zero (the bug was silently affecting those configs too, just less severely because their cost/reg ratios were closer to the numerical-stability boundary).

## Impact on the clip/flow tightness ratio

The v1 preliminary summary noted an unexplained "Anomaly 2": the clip/flow ratio grew with radius on the classifier, peaked at ~8.4× at r=0.5, and then dropped to ~2.3× at r=1.0. The drop suggested the flow was failing at the largest radius. v2 shows the fix restores the expected monotone-ish shape:

| Network | Radius | v1 clip/flow ratio | v2 clip/flow ratio |
|---|---|---|---|
| RotatedBananaNet | 0.05 | 0.83× | 0.88× |
| RotatedBananaNet | 0.10 | 1.10× | 1.22× |
| RotatedBananaNet | 0.20 | 1.41× | 1.57× |
| RotatedBananaNet | 0.40 | 1.71× | 1.86× |
| ThreeBlobClassifier | 0.10 | 3.09× | 3.83× |
| ThreeBlobClassifier | 0.25 | 6.35× | 6.64× |
| ThreeBlobClassifier | 0.50 | 8.40× | 9.46× |
| **ThreeBlobClassifier** | **1.00** | **2.32×** | **5.86×** |

v2's classifier curve now reads `3.83 → 6.64 → 9.46 → 5.86` — still a slight drop at r=1.0 (from 9.46 at r=0.5), but much less dramatic than v1's `2.32`. The residual drop at r=1.0 is likely a secondary effect that's *not* caused by the Sinkhorn bug — possibly multi-modal distributions at extreme perturbations, or a preimage-search aggressiveness difference (see "verdict changes" below). That's future work, not this fix.

## Coverage and soundness

- **Empirical coverage** on all 72 v2 configs: minimum 0.9990, maximum 1.0000. Target is `1 − ε₁ = 0.999`. Conformal guarantee holds.
- **Soundness cross-check vs. n2v exact ground truth**: zero false positives and zero false negatives for both flow and clip. No soundness regression from v1.

## Verdict changes

The flow verdict distribution shifted slightly between v1 and v2, most visibly at classifier r=1.0:

| Network | Radius | v1 verdicts (F/U/V) | v2 verdicts (F/U/V) |
|---|---|---|---|
| RotatedBananaNet | 0.05 | 3 / 0 / 6 | 3 / 0 / 6 |
| RotatedBananaNet | 0.10 | 6 / 0 / 3 | 6 / 0 / 3 |
| RotatedBananaNet | 0.20 | 6 / 0 / 3 | 6 / 0 / 3 |
| RotatedBananaNet | 0.40 | 3 / 6 / 0 | 2 / 7 / 0 |
| ThreeBlobClassifier | 0.10 | 0 / 0 / 9 | 0 / 0 / 9 |
| ThreeBlobClassifier | 0.25 | 3 / 0 / 6 | 2 / 1 / 6 |
| ThreeBlobClassifier | 0.50 | 2 / 1 / 6 | 2 / 1 / 6 |
| **ThreeBlobClassifier** | **1.00** | **3 / 3 / 3** | **0 / 6 / 3** |

(F = falsified, U = unknown, V = verified.)

The notable shifts are all at the largest-radius / hardest configs:

- **Banana r=0.40**: 3→2 falsified, 6→7 unknown. One run flipped from falsified to unknown.
- **Classifier r=0.25**: 3→2 falsified, 0→1 unknown. Also a flip.
- **Classifier r=1.00**: 3→0 falsified, 3→6 unknown. **All three previously-falsified runs flipped to unknown.**

The explanation: v2's flow reach set is much tighter at classifier r=1.0 (58.9 vs 150.9), so the scenario verification's latent-ball sampling explores fewer candidate points, and the preimage search has fewer opportunities to find genuine counterexamples in the input space. The pipeline falls back to "unknown" when the scenario check produces a violation candidate but the preimage search can't confirm a real input-space counterexample.

This is **not a soundness regression** — verified never contradicts exact, and unknown is always a valid fallback. It's a preimage-search aggressiveness trade-off that emerges from the tighter reach set. If it becomes a paper concern, the fix is to tune `preimage_n_restarts` / `preimage_n_steps` (documented in [`n2v/probabilistic/flow/scenario_verify.py`](../../../../../../n2v/probabilistic/flow/scenario_verify.py)) rather than to undo the reg fix. Left as future work.

## Takeaways

1. **The Sinkhorn numerical-underflow bug is now fixed at the library level.** Every caller of `train_flow(..., coupling='sinkhorn')` without an explicit `sinkhorn_reg` automatically gets adaptive reg going forward.
2. **The Hashemi comparison headline holds and gets stronger at the hardest config.** Flow now beats clip by 5.86× at classifier r=1.0 (up from 2.32×). The monotone(-ish) scaling story is cleaner.
3. **The "Anomaly 2" drop at r=1.0 is largely explained but not fully eliminated.** A residual drop from 9.46× at r=0.5 to 5.86× at r=1.0 remains — smaller and unrelated to Sinkhorn. Candidate explanations (multi-modal outputs, preimage search saturation) deferred to future work.
4. **v1 is preserved for audit.** The v1 archive at [`../v1_original_buggy/`](../v1_original_buggy/) retains the original buggy CSV and 5 figures byte-identical to the pre-fix state, plus a pinned script that reproduces the buggy behavior exactly if needed.

## Deferred

- Residual flow matching research direction (separate brainstorming session once v2 lands)
- ReFlow + EMA + Sinkhorn iteration reduction (training-speed optimizations, not on the critical path)
- Clip speedup via convex-hull vertex extraction (Paper 2 footnote optimization)
- GPU acceleration for flow training
- Scaling up the Hashemi comparison to MNIST and ACAS Xu
- Investigation of the residual drop at classifier r=1.0

## Files

- CSV: [`exp_hashemi_comparison.csv`](exp_hashemi_comparison.csv)
- Figures: [`exp_hashemi_comparison_volume_ratios.png`](exp_hashemi_comparison_volume_ratios.png), [`exp_hashemi_comparison_scaling.png`](exp_hashemi_comparison_scaling.png), [`exp_hashemi_comparison_verdicts.png`](exp_hashemi_comparison_verdicts.png), [`exp_hashemi_comparison_runtimes.png`](exp_hashemi_comparison_runtimes.png), [`exp_hashemi_comparison_hero_banana.png`](exp_hashemi_comparison_hero_banana.png)
- v1 archive: [`../v1_original_buggy/`](../v1_original_buggy/)
