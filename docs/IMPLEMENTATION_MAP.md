# Implementation Map

This repository packages the executable ActiveMap reference implementation,
not raw benchmark data, trained checkpoints, sealed evaluation artifacts, or
cluster orchestration.

## Core Package

`src/activemap/` contains the reusable implementation:

- `agent/`: typed evidence decisions, counterfactual utilities, and Safe
  Commit writeback;
- `data/`: map, temporal-update, structured HD-map, and RGB-D navigation
  adapters;
- `nn/` and `training/`: reference updater, selector, and visual-value heads;
- `oracle/`: validation-only counterfactual construction;
- `evaluation/`: executable map and navigation rollout evaluation;
- `cli.py`: dataset-contract validation and conversion entry points.

The synthetic workflow in `scripts/run_smoke.sh` exercises the core proposal,
selection, commit, and audit path without downloading external assets.

## Reproducible Dataset Pipelines

The public scripts below operate only on data acquired by the user under the
respective provider terms. They write all derived files to user-supplied output
locations and never contain dataset credentials or cluster paths.

| Domain | Main entry points | Output contract |
| --- | --- | --- |
| Temporal satellite map updates | `build_spacenet8_disaster_smoke.py`, `prepare_spacenet8_flood_smoke.py`, `prepare_spacenet8_multi_post_candidates.py`, `infer_spacenet8_multi_post_candidates.py`, `evaluate_spacenet8_active_multi_post.py` | candidate records, executable writeback metrics, bootstrap summaries |
| Structured HD-map updates | `build_argotweak_native_episodes.py`, `export_argotweak_official_proposals.py`, `build_argotweak_observable_features.py`, `train_argotweak_visual_selector.py`, `evaluate_argotweak_executable_policies.py` | typed structured-map scenes and policy evaluations |
| RGB-D navigation maps | `materialize_habitat_rgbd_navigation.py`, `materialize_habitat_scene_disjoint_suite.py`, `train_habitat_visual_value.py`, `run_habitat_online_mapping_pilot.py`, `evaluate_habitat_visual_value.py` | scene-disjoint navigation maps and rollout metrics |

The rendering utilities export individual assets and casebooks only after a
user has produced compatible local predictions. They do not bundle paper
figures, source imagery, or model checkpoints.

## Release Boundary

`configs/*_server.yaml`, raw data, derived data, model weights, experiment
outputs, and private cluster scripts are intentionally ignored. Dataset access,
data splits, and metric definitions are documented in
[`dataset_protocol.md`](dataset_protocol.md), [`metrics.md`](metrics.md), and
[`MODEL_DATA_AVAILABILITY.md`](../MODEL_DATA_AVAILABILITY.md).
