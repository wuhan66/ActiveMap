#!/usr/bin/env bash
set -euo pipefail

REPO="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
BASE="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
SAM_REPO="${SAM_ROAD_REPO:-${BASE}/external/sam_road}"
SAM_ENV="${SAM_ROAD_ENV:-/home/wh/yes/envs/activemap-sam-road}"
GPU="${SAM_ROAD_GPU:-3}"
DATA="${BASE}/runs/joint_debug/post_acquisition_tool_pair_data_v4"
EPISODES="${BASE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
MANIFEST="${BASE}/processed/muno21_v2/updater/updater_samples.jsonl"
MODEL_ROOT="${BASE}/models/sam_road"
ADAPTED_RUN="${BASE}/runs/semantic/muno21_sam_road_head_seed20260716_v1"
CHECKPOINT="${ADAPTED_RUN}/checkpoints/best_full.ckpt"
OUT="${BASE}/runs/joint_debug/post_acquisition_sam_road_muno21_adapted_gate_v1"
LOG="${BASE}/logs/post_acquisition_sam_road_muno21_adapted_gate_v1.log"

for path in \
  "${SAM_ENV}/bin/python" \
  "${SAM_REPO}/config/toponet_vitb_256_spacenet.yaml" \
  "$CHECKPOINT" \
  "${MODEL_ROOT}/sam_vit_b_01ec64.pth" \
  "${DATA}/train.jsonl" "${DATA}/val.jsonl" "$EPISODES" "$MANIFEST"; do
  test -e "$path" || { echo "missing prerequisite: $path" >&2; exit 2; }
done
test ! -e "$OUT"
IFS=',' read -r utilization memory_used < <(
  nvidia-smi --id="$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( utilization > 10 || memory_used > 1024 )); then
  echo "GPU ${GPU} is not idle: utilization=${utilization}%, memory=${memory_used} MiB" >&2
  exit 3
fi

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="${REPO}/src"
export ACTIVEMAP_DISABLE_FROZEN_TEST=1
export CUDA_VISIBLE_DEVICES="$GPU"

run_builder() {
  local split=$1
  local limit=$2
  "${SAM_ENV}/bin/python" scripts/build_post_acquisition_semantic_tool_data.py \
    "${DATA}/${split}.jsonl" "$EPISODES" "$MANIFEST" \
    "$CHECKPOINT" "${OUT}/${split}" \
    --split "$split" --device cuda --backend sam-road \
    --sam-road-repo "$SAM_REPO" \
    --sam-road-config "$SAM_REPO/config/toponet_vitb_256_spacenet.yaml" \
    --sam-base-checkpoint "${MODEL_ROOT}/sam_vit_b_01ec64.pth" \
    --limit-per-class "$limit"
}

{
  echo "start=$(date --iso-8601=seconds) gpu=$GPU checkpoint=$CHECKPOINT"
  run_builder train 50
  run_builder val 30
  /home/wh/venvs/activemap/bin/python scripts/audit_post_acquisition_tool_features.py \
    "${OUT}/train/train.jsonl" "${OUT}/val/val.jsonl" \
    "${OUT}/feature_audit.json" --seed 20260821
  /home/wh/venvs/activemap/bin/python scripts/calibrate_semantic_tool_threshold.py \
    "${OUT}/train/train.jsonl" "${OUT}/val/val.jsonl" \
    "${OUT}/fixed_threshold_train_cv_gate.json" \
    --fixed-threshold 0.6 --seed 20260821
  echo "complete=$(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG"
