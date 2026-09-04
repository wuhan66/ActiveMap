#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PYTHON="$STORE/envs/activemap-agent/bin/python"
SFT="$STORE/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
PREF="$STORE/runs/sn7_active_catalog/onpolicy_rl_reentry_stage_a_eps035_20260731/preferences_safety_v1"
ROOT="$STORE/runs/sn7_active_catalog"
SEED="${SEED:-20260731}"
SMOKE_GPU="${SMOKE_GPU:-3}"
BETA05_GPU="${BETA05_GPU:-3}"
BETA10_GPU="${BETA10_GPU:-4}"

cd "$PROJECT"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

launch() {
  local output="$1" gpu="$2" beta="$3" train_samples="$4" val_samples="$5"
  shift 5
  "$PYTHON" scripts/launch_active_catalog_vlm_dpo.py \
    "$SFT" "$PREF/train.jsonl" "$PREF/val.jsonl" "$output" \
    --gpu "$gpu" --seed "$SEED" --python "$PYTHON" \
    --epochs 1 --learning-rate 2e-6 --batch-size 1 \
    --gradient-accumulation 16 --max-length 4096 --beta "$beta" \
    --logging-steps 1 --eval-steps 20 --save-steps 20 \
    "$@" \
    --max-train-samples "$train_samples" --max-eval-samples "$val_samples"
}

SMOKE="$ROOT/dpo_safety_smoke_20260731"
launch "$SMOKE" "$SMOKE_GPU" 0.05 2 2
test "$("$PYTHON" -c "import json; print(json.load(open('$SMOKE/process_result.json'))['status'])")" = completed

launch "$ROOT/dpo_safety_beta005_20260731" "$BETA05_GPU" 0.05 113 52 &
pid05=$!
launch "$ROOT/dpo_safety_beta010_20260731" "$BETA10_GPU" 0.10 113 52 &
pid10=$!
wait "$pid05"
wait "$pid10"
