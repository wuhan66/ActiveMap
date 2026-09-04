#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 SEED" >&2
  exit 2
fi

SEED="$1"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${TOOL_BELIEF_RUN_ROOT:-/mnt/mydisk/wh/ActiveMap/runs/agent}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent}"
DETAILS="${VAL_DETAILS:-${RUN_ROOT}/tool_belief_anchored_v4_seed20260821_sequence_eval_detailed/details.jsonl}"
RECORDS="${VAL_RECORDS:-${PROCESSED_ROOT}/tool_belief_v1/val.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROCESSED_ROOT}/tool_belief_v1/artifacts/temporal_change}"
GPU="${GPU:-3}"
EXPERIMENT_VERSION="${EXPERIMENT_VERSION:-v1}"
FUSION_MODE="${FUSION_MODE:-concat}"
GATE_BIAS="${GATE_BIAS:--2.0}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

assert_gpu_free() {
  local gpu_uuid
  gpu_uuid="$({ nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits || true; } \
    | awk -F, -v gpu_index="$GPU" \
      '$1 + 0 == gpu_index {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
  if [[ -z "$gpu_uuid" ]]; then
    echo "Physical GPU $GPU was not found" >&2
    exit 3
  fi
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -Fxq "$gpu_uuid"; then
    echo "Physical GPU $GPU is not free; refusing paired execution" >&2
    exit 4
  fi
}

completed_training() {
  local mode="$1"
  local run_dir="${RUN_ROOT}/tool_belief_spatial_decision_${EXPERIMENT_VERSION}_${mode}_seed${SEED}"
  local exit_code="/mnt/mydisk/wh/ActiveMap/logs/tool_belief_spatial_decision_${EXPERIMENT_VERSION}_${mode}_seed${SEED}.log.exit_code"
  [[ -s "${run_dir}/best_safety.pt" ]] \
    && [[ -s "${run_dir}/summary.json" ]] \
    && [[ -f "$exit_code" ]] \
    && [[ "$(tr -d '[:space:]' < "$exit_code")" == "0" ]]
}

train_mode() {
  local mode="$1"
  if completed_training "$mode"; then
    echo "Reusing completed ${mode} training for seed ${SEED}"
    return
  fi
  assert_gpu_free
  GPU="$GPU" ABLATION="$mode" SEED="$SEED" \
    EXPERIMENT_VERSION="$EXPERIMENT_VERSION" FUSION_MODE="$FUSION_MODE" \
    GATE_BIAS="$GATE_BIAS" \
    bash scripts/run_tool_belief_spatial_decision_head.sh
  if ! completed_training "$mode"; then
    echo "${mode} training did not produce a complete audited run" >&2
    exit 5
  fi
}

evaluate_mode() {
  local mode="$1"
  local run_dir="${RUN_ROOT}/tool_belief_spatial_decision_${EXPERIMENT_VERSION}_${mode}_seed${SEED}"
  local eval_dir="${run_dir}_eval"
  assert_gpu_free
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    scripts/evaluate_tool_belief_spatial_decision_head.py \
    "$run_dir/best_safety.pt" "$DETAILS" "$RECORDS" "$ARTIFACT_DIR" "$eval_dir" \
    --device cuda:0
}

for mode in spatial no_spatial; do
  train_mode "$mode"
  evaluate_mode "$mode"
done

echo "Completed paired spatial/no-spatial seed ${SEED}"
