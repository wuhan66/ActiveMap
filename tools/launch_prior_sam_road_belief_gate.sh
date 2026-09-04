#!/usr/bin/env bash
set -euo pipefail

REPO="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
BASE="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
SAM_REPO="${SAM_ROAD_REPO:-${BASE}/external/sam_road}"
SAM_ENV="${SAM_ROAD_ENV:-/home/wh/yes/envs/activemap-sam-road}"
CORE_ENV="${ACTIVEMAP_CORE_ENV:-/home/wh/venvs/activemap}"
RUN_BELIEF_EVAL="${PRIOR_SAM_ROAD_RUN_EVAL:-1}"
RUN_NAME="${PRIOR_SAM_ROAD_RUN_NAME:-post_acquisition_prior_sam_road_belief_gate_v1}"
TRAIN_LIMIT="${PRIOR_SAM_ROAD_TRAIN_LIMIT:-50}"
VAL_LIMIT="${PRIOR_SAM_ROAD_VAL_LIMIT:-30}"
CHANGE_SEED="${PRIOR_SAM_ROAD_CHANGE_SEED:-20260716}"
EVAL_SEED="${PRIOR_SAM_ROAD_EVAL_SEED:-20260821}"
ASSET_ROOT_MAP="${ACTIVEMAP_ASSET_ROOT_MAP:-}"
GPU="${PRIOR_SAM_ROAD_GPU:-3}"
DATA="${BASE}/runs/joint_debug/post_acquisition_tool_pair_data_v4"
EPISODES="${BASE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
MANIFEST="${BASE}/processed/muno21_v2/updater/updater_samples.jsonl"
MODEL_ROOT="${BASE}/models/sam_road"
SOURCE_RUN="${BASE}/runs/semantic/muno21_sam_road_head_seed20260716_v1"
CHANGE_RUN="${PRIOR_SAM_ROAD_CHANGE_RUN:-${BASE}/runs/semantic/muno21_sparse_positive_change_seed${CHANGE_SEED}_v1}"
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/best_full.ckpt"
CHANGE_CHECKPOINT="${CHANGE_RUN}/checkpoints/best_head.pt"
CHANGE_SUMMARY="${CHANGE_RUN}/summary.json"
OUT="${BASE}/runs/joint_debug/${RUN_NAME}"
LOG="${BASE}/logs/${RUN_NAME}.log"

for path in \
  "${SAM_ENV}/bin/python" \
  "${SAM_REPO}/config/toponet_vitb_256_spacenet.yaml" \
  "$SOURCE_CHECKPOINT" "$CHANGE_CHECKPOINT" "$CHANGE_SUMMARY" \
  "${MODEL_ROOT}/sam_vit_b_01ec64.pth" \
  "${DATA}/train.jsonl" "${DATA}/val.jsonl" "$EPISODES" "$MANIFEST"; do
  test -e "$path" || { echo "missing prerequisite: $path" >&2; exit 2; }
done
if [[ "$RUN_BELIEF_EVAL" == "1" ]]; then
  test -e "${CORE_ENV}/bin/python" || {
    echo "missing prerequisite: ${CORE_ENV}/bin/python" >&2
    exit 2
  }
elif [[ "$RUN_BELIEF_EVAL" != "0" ]]; then
  echo "PRIOR_SAM_ROAD_RUN_EVAL must be 0 or 1" >&2
  exit 2
fi
for limit in "$TRAIN_LIMIT" "$VAL_LIMIT"; do
  [[ "$limit" == "all" || "$limit" =~ ^[1-9][0-9]*$ ]] || {
    echo "sample limits must be positive integers: train=$TRAIN_LIMIT val=$VAL_LIMIT" >&2
    exit 2
  }
done
for seed in "$CHANGE_SEED" "$EVAL_SEED"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "seeds must be non-negative integers: change=$CHANGE_SEED eval=$EVAL_SEED" >&2
    exit 2
  }
done
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
IFS=',' read -r utilization memory_used < <(
  nvidia-smi --id="$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( utilization > 10 || memory_used > 1024 )); then
  echo "GPU ${GPU} is not idle: utilization=${utilization}%, memory=${memory_used} MiB" >&2
  exit 3
fi

CURRENT_THRESHOLD=$("${SAM_ENV}/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_channel_thresholds"]["current"])' \
  "$CHANGE_SUMMARY")
mkdir -p "$OUT" "$(dirname "$LOG")"
cd "$REPO"
export PYTHONPATH="${REPO}/src"
export ACTIVEMAP_DISABLE_FROZEN_TEST=1
export CUDA_VISIBLE_DEVICES="$GPU"

run_builder() {
  local split=$1
  local limit=$2
  local root_map_args=()
  local limit_args=()
  if [[ -n "$ASSET_ROOT_MAP" ]]; then
    root_map_args=(--asset-root-map "$ASSET_ROOT_MAP")
  fi
  if [[ "$limit" != "all" ]]; then
    limit_args=(--limit-per-class "$limit")
  fi
  "${SAM_ENV}/bin/python" scripts/build_post_acquisition_semantic_tool_data.py \
    "${DATA}/${split}.jsonl" "$EPISODES" "$MANIFEST" \
    "$CHANGE_CHECKPOINT" "${OUT}/${split}" \
    --split "$split" --device cuda:0 --backend prior-sam-road \
    --sam-road-repo "$SAM_REPO" \
    --sam-road-config "$SAM_REPO/config/toponet_vitb_256_spacenet.yaml" \
    --sam-base-checkpoint "${MODEL_ROOT}/sam_vit_b_01ec64.pth" \
    --sam-road-source-checkpoint "$SOURCE_CHECKPOINT" \
    --change-summary "$CHANGE_SUMMARY" \
    "${limit_args[@]}" \
    "${root_map_args[@]}"
}

{
  echo "start=$(date --iso-8601=seconds) gpu=$GPU current_threshold=$CURRENT_THRESHOLD train_limit=$TRAIN_LIMIT val_limit=$VAL_LIMIT change_seed=$CHANGE_SEED eval_seed=$EVAL_SEED"
  echo "change_checkpoint=$CHANGE_CHECKPOINT"
  run_builder train "$TRAIN_LIMIT"
  run_builder val "$VAL_LIMIT"
  if [[ "$RUN_BELIEF_EVAL" == "1" ]]; then
    "${CORE_ENV}/bin/python" scripts/calibrate_semantic_tool_threshold.py \
      "${OUT}/train/train.jsonl" "${OUT}/val/val.jsonl" \
      "${OUT}/train_cv_belief_gate.json" \
      --fixed-threshold "$CURRENT_THRESHOLD" --seed "$EVAL_SEED" \
      --model-training-seed "$CHANGE_SEED"
  else
    echo "belief_eval=deferred reason=core_environment_unavailable"
  fi
  echo "complete=$(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG"
