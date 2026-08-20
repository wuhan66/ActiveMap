# Baseline Adapter Boundary

Each subdirectory in this package is ActiveMap-owned glue for one registered baseline.
Adapters consume frozen dataset manifests and produce
`activemap-external-baseline-result-v1` records plus raw predictions.

Third-party model code, checkpoints, and vendored repositories do not belong here.
They are addressed by immutable source URL and commit from
`configs/experiments/external_baselines.yaml`.

An adapter must keep preprocessing, model invocation, postprocessing, and evaluation as
separate stages. It must not read labels during inference, alter the common split, tune
thresholds on test, replace unsupported actions with KEEP, or report metrics computed by
an incompatible external evaluator as if they were ActiveMap metrics.

