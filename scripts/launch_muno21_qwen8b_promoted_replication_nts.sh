#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORE="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-8B}"
DATA="${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
ROOT="${STORE}/runs/agent/muno21_qwen3_8b_gate005_replication_v1"
ROLLOUT_ROOT="${STORE}/artifacts/paper_rollouts/qwen3_8b_gate005_replication_v1"
LOG_ROOT="${STORE}/logs/qwen3_8b_gate005_replication_v1"
mkdir -p "${ROOT}/status" "${ROLLOUT_ROOT}" "${LOG_ROOT}"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/MATRIX_COMPLETED" ]] || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

run_seed() {
  local gpu="$1" seed="$2"
  local run="${STORE}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed${seed}"
  local rollout="${ROLLOUT_ROOT}/seed${seed}"
  local log="${LOG_ROOT}/seed${seed}.log"
  if [[ ! -e "${run}/TRAIN_COMPLETED" ]]; then
    [[ ! -e "${run}" ]] || { echo "partial existing run: ${run}" >&2; return 4; }
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_agent_sft.py \
      "${MODEL}" "${DATA}/train/sft_composed.jsonl" "${run}" \
      --eval-jsonl "${DATA}/val/sft_composed.jsonl" \
      --epochs 1 --learning-rate 0.0002 --batch-size 1 \
      --gradient-accumulation 4 --max-length 2048 \
      --logging-steps 5 --eval-steps 100 --save-steps 100 --save-total-limit 4 \
      --early-stopping-patience 2 --early-stopping-threshold 0.0005 \
      --lora-rank 16 --lora-alpha 32 --seed "${seed}" >"${log}" 2>&1
    touch "${run}/TRAIN_COMPLETED"
  fi
  if [[ ! -e "${rollout}/ROLLOUT_COMPLETED" ]]; then
    env ACTIVEMAP_PROJECT_ROOT="${PROJECT_ROOT}" ACTIVEMAP_STORAGE_ROOT="${STORE}" \
      ACTIVEMAP_AGENT_ENV="$(dirname "${PYTHON}")/.." MUNO21_V12_GPU="${gpu}" \
      MUNO21_AGENT_SEED="${seed}" MUNO21_SELECTOR_SEED=20260811 \
      MUNO21_AGENT_MODEL="${MODEL}" MUNO21_V12_ADAPTER="${run}/final" \
      MUNO21_V12_METHODS=qwen3_4b_sft_calibrated_tool_to_belief \
      MUNO21_V12_TOOL_NEED_THRESHOLD=0.05 MUNO21_V12_ASSESS=0 \
      MUNO21_V12_ASSET_ROOT_MAP="/home/wh/ActiveMap=${STORE}" \
      MUNO21_V12_ROLLOUT_ROOT="${rollout}" \
      bash scripts/run_muno21_v12_proactive_rollout.sh >>"${log}" 2>&1
    touch "${rollout}/ROLLOUT_COMPLETED"
  fi
  printf '{"seed":%s,"gpu":%s,"status":"complete","threshold":0.05,"test_assets_read":false}\n' \
    "${seed}" "${gpu}" >"${ROOT}/status/seed${seed}.json"
}

run_seed 0 20261411 & pid0=$!
sleep 10
run_seed 1 20261412 & pid1=$!
status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
if [[ "${status}" -eq 0 ]]; then touch "${ROOT}/MATRIX_COMPLETED"; else touch "${ROOT}/MATRIX_FAILED"; fi
exit "${status}"
