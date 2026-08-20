# ActiveMap Dataset Protocol

This document is the open-source dataset entry point for the current ICLR
paper package. It summarizes how raw mapping datasets are converted into
editable-map writeback episodes.

## Scope

ActiveMap evaluates single-episode editable vector-map writeback after a shared
candidate-perception front end. Dataset construction must therefore preserve:

- split isolation by geographic unit or task scenario;
- typed map operations: `KEEP`, `ADD`, `DELETE`, `RESHAPE`;
- candidate evidence provenance, cost, and budget feasibility;
- executable geometry needed for final vector writeback;
- receipts proving that validation/test assets are not read by train-only
  calibration.

## Dataset Roles

| Dataset | Geometry | Role in paper | Claim boundary |
| --- | --- | --- | --- |
| SpaceNet 7 | building polygons | primary remote-sensing editable-map writeback benchmark | sealed test supports conservative maintenance and false-write avoidance; non-KEEP recovery remains a boundary |
| MUNO21 | road graph / polylines | cross-geometry transfer to road topology | supports active control versus STOP and historical selector, not generic-selector dominance |
| SpaceNet 8 | disaster change polygons | cross-domain active evidence and Safe Commit stress test | qualified-backend evidence only |
| ArgoTweak | HD-map atomic changes | interface/adaptation boundary | no HD-map perception SOTA claim |
| Inria Aerial | building masks | optional boundary/pretraining support | not temporal update evidence |

## Standard Episode Schema

Each episode should expose:

- `episode_id`, `split`, `aoi_id` or task group;
- prior editable map geometry and rasterized prior mask;
- current observation metadata;
- target typed operation, used only for supervision/evaluation;
- candidate evidence catalog with evidence ID, source image/time/view/scale,
  validity, cost, and budget affordability;
- updater hypothesis: operation logits, confidence, geometry proposal, and
  predicted mask or graph state;
- controller state: selected evidence IDs, remaining budget, belief features,
  and STOP margin;
- writeback record: executed operation, committed geometry or `KEEP`, topology
  validity, final-map metrics, and audit hashes.

## SN7 Construction

1. Split by AOI before generating objects, crops, or episodes.
2. Match adjacent monthly building snapshots with stable IDs when reliable,
   otherwise local spatial matching with IoU and centroid-distance guards.
3. Derive typed operations:
   - `KEEP`: matched object with high overlap;
   - `ADD`: object appears in the later month;
   - `DELETE`: object disappears from the later month;
   - `RESHAPE`: matched object has material geometry change.
4. Filter invalid or mostly missing crops using UDM/non-black validity masks.
5. Build object-centered updater crops and episode-level candidate catalogs
   with identical operation filters and seeds.
6. Build selector states only after the updater checkpoint is frozen.

The canonical detailed Chinese data card remains
`docs/DATASET_CONSTRUCTION_AND_METRICS_ZH.md`; this file is the short
open-source protocol.

## MUNO21 Construction

MUNO21 converts multi-year road observations into graph-update episodes. The
same ActiveMap controller interface is used, but final quality is reported with
road metrics such as APLS and Pixel-F1. The road extractor is treated as a
frozen backend for controller comparison.

## SpaceNet 8 Construction

SpaceNet 8 uses pre-event and multiple post-event observations. Candidate
evidence corresponds to alternative post-event views or model outputs. This
dataset is used to show domain transfer under qualified change-perception
backends; it is not averaged into the SN7 sealed-test claim.

## Split And Leakage Rules

- Never split by crop or candidate row when AOI/task identity is available.
- Train-only thresholds, STOP margins, and Safe Commit gates must be fitted
  without validation outcomes.
- Frozen test access requires an explicit receipt and must not be used for
  model selection, threshold tuning, figure selection, or qualitative mining.
- Validation-only extensions such as V5 can be reported only with their split
  and promotion status.

## Reproducibility Checklist

- Raw data source and license recorded.
- Split manifest hash recorded.
- Episode manifest hash recorded.
- Updater checkpoint hash recorded.
- Selector-state hash recorded.
- Calibration split hash recorded.
- Aggregation bootstrap unit recorded.
- `test_assets_read=false` recorded for validation-only runs.
