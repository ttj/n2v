# n2v — VNN-COMP 2026 Benchmark Support

_Status snapshot: 2026-06-13. Supersedes the 2026-06-10 snapshot
(which predated the reach-layer work and reported 14/34)._

This document records which VNN-COMP 2026 benchmarks n2v can handle, and
documents the layer and specification types that are not yet supported.
The companion internal doc `docs/layer_soundness_coverage.md` records the
per-operator soundness validation that backs the "sound reach" claims.

## What "support" means

Load the ONNX model, parse the VNNLIB spec, and run **sound** set-based
reachability (`approx` Star) through every layer to completion. This is
stricter than "solved": it excludes SAT found by falsification (forward
passes only) and UNSAT from the probabilistic / conformal method (a
coverage guarantee, **not** a sound proof). Every unsupported case fails
as an **error → `unknown`**, never a silent wrong verdict — there is no
soundness *violation*.

## How this was measured

Per-benchmark scorecard sweep (`scorecard_one.py`, subprocess-isolated,
150 s reach cap): for the **first instance** of each benchmark, probe
spec-parse + model-load + degenerate reach, then cross-reference the op
inventory (55 op types / 605 models) for intra-benchmark gaps. "✓" = the
first instance works end-to-end; the Caveat/Blocker column flags
instances or models that differ. A scorecard is a one-instance snapshot,
not a full-benchmark score; timeouts are at the smoke budget and usually
indicate per-benchmark tuning, not a hard limitation.

## Summary

- **Specs: 34/34 parse** (VNNLIB 1.0, 2.0, relational, nonlinear).
- **Models: 31/34 load.**
- **Sound reach: ~27 benchmarks clear**, the rest blocked on a handful
  of unsupported ops, the missing relational engine, or model-load gaps.

## Tier 1 — fully working (spec ✓, load ✓, sound reach ✓) — 27

| Benchmark | Note |
|---|---|
| acasxu_2023 | |
| adaptive_cruise_control_non_linear_2026 | nonlinear spec; model is a plain MLP |
| cersyve | |
| cgan_2023 | ConvTranspose |
| cgan2026 | ConvTranspose |
| challenging_certified_training_2026 | |
| cifar100_2024 | |
| collins_rul_cnn_2022 | |
| cora_2024 | |
| dist_shift_2023 | has `Shape`, but it constant-folds |
| linearizenn_2024 | |
| lsnc_relu | |
| malbeware | |
| metaroom_2023 | |
| ml4acopf_2024 | full non-linear 14/118_ieee reach (Pow + Sin/Cos); see 300_ieee caveat below |
| nn4sys | Pow (x³) |
| relusplitter | |
| relusplitter_2026 | |
| safenlp_2024 | |
| sat_relu | |
| soundnessbench | |
| soundnessbench_2026 | |
| test | |
| tinyimagenet_2024 | |
| tllverifybench_2023 | |
| traffic_signs_recognition_2023 | Sign+Softmax; full-eps *sound verification* is perf-bound (>900 s) |
| yolo_2023 | |

## Tier 2 — partial: first instance works, some models do not — 2

| Benchmark | Blocker on the bad models |
|---|---|
| ml4acopf_2024 | `14/118_ieee` (linear and full) work; **`300_ieee` mis-converts in onnx2torch** (I-37) → invalid, a third_party bug independent of op support |
| collins_aerospace_benchmark | first instance runs but **perf-bound** (>150 s); Pow is supported |

## Tier 3 — blocked — 5

| Benchmark | Load | Reach | Blocker |
|---|---|---|---|
| smart_turn_multimodal_2026 | ✗ | — | `DequantizeLinear` has no onnx2torch converter (I-38) |
| vggnet16_2022 | ✗ | — | model file not committed; fetched by `setup.sh` (data, not code) |
| cctsdb_yolo_2023 | ✓ | ✗ | rank-1 data-axis `Slice` + YOLO control-flow ops (`ArgMax`, `ScatterND`, `Clip`, `Max`, `Min`, `Where`, `Range`, …) |
| vit_2023 | ✓ | ✗ | `OnnxShape` and transformer ops — **out of scope by decision** |
| isomorphic_acasxu_2026 | ✓ | N/A | relational (two-network) — parses, **no relational engine** |
| monotonic_acasxu_2026 | ✓ | N/A | relational — no engine (probe also mis-resolved its tuple model path) |

## Remaining work to "everything end-to-end"

1. **Unsupported ops** — the cctsdb_yolo control-flow cluster
   (`ArgMax/ScatterND/Clip/Max/Min/Where/Range/Expand/…`); `Shape` (vit,
   out of scope); `Erf/Sqrt/Exp` (smart_turn, which is load-blocked by
   DequantizeLinear regardless). `Pow`, `Sin`, `Cos` were added
   2026-06-13 and cleared nn4sys and ml4acopf-full.
2. **Relational engine** — joint reachability over the two-network
   specs; unblocks both acasxu-relational benchmarks.
3. **third_party fixes** — I-38 (DequantizeLinear → smart_turn), I-37
   (onnx2torch mis-conversion → ml4acopf 300_ieee).
4. **Perf / data** — tuning (collins, traffic_signs full-eps) and the
   vggnet16 download.

## Soundness note

The tool also has a **probabilistic** (conformal / flow) lane and a
**falsification** lane (random + PGD). Results from those carry a
coverage guarantee or a counterexample, **not** a sound proof, and must
be reported/scored accordingly. The sound vs. probabilistic choice per
benchmark is a tuning decision deferred to a later pass. This document is
exclusively about the **sound reach** lane.
