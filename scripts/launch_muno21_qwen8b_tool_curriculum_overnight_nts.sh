#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-8B}"
SOURCE_DATA="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
CURRICULUM_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent/qwen8b_tool_curriculum_v1"
RUN_ROOT="${STORAGE_ROOT}/runs/agent/qwen3_8b_tool_curriculum_v1"
ROLLOUT_ROOT="${STORAGE_ROOT}/artifacts/paper_rollouts/qwen3_8b_tool_curriculum_v1"
LOG_ROOT="${STORAGE_ROOT}/logs/qwen3_8b_tool_curriculum_v1"
SCREEN_ROOT="${STORAGE_ROOT}/artifacts/paper_rollouts/qwen3_8b_structured_acquisition_matched_v2_threshold009"
SEED="${MUNO21_CURRICULUM_SEED:-20261401}"
SELECTOR_SEED="${MUNO21_SELECTOR_SEED:-20260811}"
GPU_R2="${GPU_R2:-1}"
GPU_R3="${GPU_R3:-2}"

mkdir -p "${CURRICULUM_ROOT}" "${RUN_ROOT}" "${ROLLOUT_ROOT}" "${LOG_ROOT}"
exec 9>"${RUN_ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${RUN_ROOT}/MATRIX_COMPLETED" ]] || exit 0

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

# Repair the earlier diagnostic's accidental 0.9 threshold before training.
MUNO21_SCREEN_ROOT="${SCREEN_ROOT}" \
MUNO21_V12_TOOL_NEED_THRESHOLD=0.09 \
GPU_NO_BELIEF="${GPU_R2}" GPU_RECURRENT="${GPU_R3}" \
bash scripts/launch_muno21_qwen8b_structured_acquisition_screen_nts.sh

build_curriculum() {
  local repeat="$1"
  local output="${CURRICULUM_ROOT}/tool_repeat${repeat}/train/sft_composed.jsonl"
  if [[ ! -s "${output}" ]]; then
    "${PYTHON}" scripts/balance_agent_sft.py \
      "${SOURCE_DATA}/train/sft_composed.jsonl" "${output}" \
      --acquire-repeat 1 --tool-repeat "${repeat}" --assume-train
  fi
}

build_curriculum 2
build_curriculum 3

run_variant() {
  local gpu="$1" repeat="$2"
  local label="tool_repeat${repeat}_seed${SEED}"
  local train_data="${CURRICULUM_ROOT}/tool_repeat${repeat}/train/sft_composed.jsonl"
  local run="${RUN_ROOT}/${label}"
  local rollout="${ROLLOUT_ROOT}/${label}"
  local log="${LOG_ROOT}/${label}.log"

  [[ ! -e "${run}" ]] || {
    echo "refusing existing curriculum run: ${run}" >&2
    return 4
  }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_agent_sft.py \
    "${MODEL}" "${train_data}" "${run}" \
    --eval-jsonl "${SOURCE_DATA}/val/sft_composed.jsonl" \
    --epochs 1 --learning-rate 0.0002 --batch-size 1 \
    --gradient-accumulation 4 --max-length 2048 \
    --logging-steps 5 --eval-steps 100 --save-steps 100 \
    --save-total-limit 4 --early-stopping-patience 2 \
    --early-stopping-threshold 0.0005 --lora-rank 16 --lora-alpha 32 \
    --seed "${SEED}" >"${log}" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_agent_actions.py \
    "${MODEL}" "${SOURCE_DATA}/val/sft_composed.jsonl" \
    "${run}/evaluation/static_val" --adapter "${run}/final" \
    --device cuda --batch-size 1 --max-length 2048 --max-new-tokens 64 \
    >>"${log}" 2>&1

  env \
    ACTIVEMAP_PROJECT_ROOT="${PROJECT_ROOT}" \
    ACTIVEMAP_STORAGE_ROOT="${STORAGE_ROOT}" \
    ACTIVEMAP_AGENT_ENV="$(dirname "${PYTHON}")/.." \
    MUNO21_V12_GPU="${gpu}" \
    MUNO21_AGENT_SEED="${SEED}" \
    MUNO21_SELECTOR_SEED="${SELECTOR_SEED}" \
    MUNO21_AGENT_MODEL="${MODEL}" \
    MUNO21_V12_ADAPTER="${run}/final" \
    MUNO21_V12_METHODS=qwen3_4b_sft_calibrated_tool_to_belief \
    MUNO21_V12_TOOL_NEED_THRESHOLD=0.09 \
    MUNO21_V12_ASSESS=0 \
    MUNO21_V12_ASSET_ROOT_MAP="/home/wh/ActiveMap=${STORAGE_ROOT}" \
    MUNO21_V12_ROLLOUT_ROOT="${rollout}" \
    bash scripts/run_muno21_v12_proactive_rollout.sh >>"${log}" 2>&1
}

run_variant "${GPU_R2}" 2 &
pid_r2=$!
run_variant "${GPU_R3}" 3 &
pid_r3=$!

status=0
wait "${pid_r2}" || status=1
wait "${pid_r3}" || status=1
if [[ "${status}" -eq 0 ]]; then
  touch "${RUN_ROOT}/MATRIX_COMPLETED"
else
  touch "${RUN_ROOT}/MATRIX_FAILED"
fi
exit "${status}"
