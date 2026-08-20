#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m activemap generate-selector-smoke data/processed/smoke/selector_samples.jsonl \
  --sample-count 96 --candidate-count 6 --seed 20260710
python -m activemap generate-updater-smoke data/processed/smoke_updater \
  --sample-count 40 --image-size 16 --seed 20260710
python -m activemap train-selector configs/smoke/selector.yaml
python -m activemap train-updater configs/smoke/updater.yaml
python -m activemap evaluate-selector data/processed/smoke/selector_samples.jsonl \
  outputs/smoke/selector/test.csv --checkpoint outputs/smoke/selector/best.pt --budgets 1,2
python -m activemap evaluate-updater outputs/smoke/updater/best.pt \
  data/processed/smoke_updater/updater_samples.jsonl outputs/smoke/updater/evaluation \
  --device auto --batch-size 8 --bootstrap 20
python -m activemap rollout-selector data/processed/smoke/selector_samples.jsonl \
  outputs/smoke/rollout --checkpoint outputs/smoke/selector/best.pt --budgets 1,2
