# Reproducibility Guide

## Scope

The code supports data construction, typed edit generation, prior-conditioned
updating, counterfactual evidence evaluation, budgeted selection, Safe Commit,
and executable vector-map writeback. The current paper claims only
single-episode downstream verification and safe writeback under the registered
protocol. It does not claim end-to-end active-vision compute savings,
long-horizon persistent deployment, or universal backend improvement.

## Environment

Use the root `environment.yml` for CPU and geospatial dependencies. Then add a
CUDA-matched PyTorch build only when training a neural component:

```bash
conda env create -f environment.yml
conda activate activemap
python -m pip install torch torchvision --index-url <matching-pytorch-index>
python -m pip install -e . --no-deps
python -m activemap --help
```

The `scripts/bootstrap_server.sh` helper performs the same setup on Linux and
expects `TORCH_INDEX_URL` only when PyTorch is absent.

## Reproduction Ladder

1. Run `bash scripts/run_smoke.sh` to generate synthetic selector and updater
   samples, train small models, evaluate them, and write rollout traces.
2. Run the curated unit tests listed in `README.md` to verify geometry edits,
   vector writeback, Safe Commit, counterfactual values, schemas, and splits.
3. Acquire external raw data under the original providers' terms and follow
   `docs/dataset_protocol.md`. Build geographic/task-disjoint manifests before
   extracting crops or candidate evidence.
4. Train an updater, freeze its validation-selected checkpoint, build selector
   supervision, train the selector, and evaluate final executable maps. Do not
   use any sealed test asset for model, threshold, budget, or figure selection.

## Deterministic Records

Every paper-facing run should retain the split and episode manifest hashes,
config, seed, checkpoint hash, calibration record, rollout trace, executed map
transaction, metric receipt, and bootstrap unit. Validation-only artifacts
should record `test_assets_read=false`.

## Release Boundaries

This release candidate omits raw imagery, vector labels, external model
weights, private logs, sealed-test artifacts, and author identity. The smoke
workflow is the executable verification target. Exact paper results require
the external datasets and the registered frozen checkpoints, which will be
handled under their respective licenses before a public release.
