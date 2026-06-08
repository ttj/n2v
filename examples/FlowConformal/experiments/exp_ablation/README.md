# Ablation experiment

Per-row ablation that varies one design choice at a time on the locked
Phase 5d pipeline (flow + AMLS + post-falsifier-fix calibration).
Results are written to `outputs/ablation_<row>_<value>.csv` and
aggregated into a single markdown table by `ablation_aggregate.py`.

## Benchmark: 10-instance ACAS Xu probe

The probe instance list is sampled via
`ablation_shared_flow._sample_instances`, which draws from
`examples/FlowConformal/experiments/exp1_vnncomp_subset/_benchmarks.py::list_instances`
(seeded shuffle for determinism). The Phase 7 runners use 10
instances by default (per the 2026-04-27 paper-experiments plan;
pass `--n-instances 20` to use a larger sample). Per-instance
wall-clock is ~30-90 s with the locked pipeline; per ablation value
the full 10-instance probe is ~6-15 min.

## Known gaps and TODOs

- **AMLS hyperparameter knobs are not exposed at the
  `run_verification_pipeline` level**. The ablation script uses a
  monkey-patch on `n2v.probabilistic.flow.amls.amls_certify_spec`; this
  is sound because the pipeline is the only caller. Fix by adding
  `amls_quantile` / `amls_n_mcmc_steps` / `amls_mcmc_step_size` kwargs
  through `_flow_unsat_pipeline` and `run_verification_pipeline`.
- **Flow training knobs (coupling / ema / standardize) are not
  exposed** at the pipeline level either. Same monkey-patch pattern;
  fix by adding kwargs through `_train_flow` and the public pipeline
  signature.
- **`standardize_outputs=True` in the ablation triggers double
  whitening** because `run_verification_pipeline` already pre-whitens
  outputs before training. A "true" no-pre-whitening row would also
  need to disable the pipeline's whitening glue (see "Whitening glue
  for run_verification_pipeline" comment in
  `examples/FlowConformal/benchmarks/_common.py`). Documented in the
  script docstring; left as future work.
- **GMM(10) score is not implemented**. There is no `GMMScore` class
  in `n2v/probabilistic/flow/scores.py`. A negative-log-density score
  under a fitted `sklearn.mixture.GaussianMixture` would be ~30 LOC to
  add; sketch is in the script docstring. Until then, the row is
  written as `nan` and skipped.

## Output layout

Each ACAS-Xu CSV has the schema:
`onnx_file, vnnlib_file, verdict, q, worst_max_margin, amls_levels_used, wall_s, error`.

The aggregator (`ablation_aggregate.py`) reads ONLY from this `outputs/`
directory — there are no fallback paths into legacy locations. Any
missing cell renders as "(missing)" in the report instead of silently
substituting older methodology data.
