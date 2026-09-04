#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
QWEN_MODEL="${QWEN_MODEL:-/home/wh/hf_models/Qwen3-8B}"
GEMMA_MODEL="${GEMMA_MODEL:-/home/wh/hf_models/gemma-3-4b-it}"
DATA="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
WAIT_FOR="${WAIT_FOR:-${STORAGE_ROOT}/runs/sn7_active_catalog/p0_static_selector_baseline_matrix_v2_20260801/COMPLETE.json}"
ROOT="${STORAGE_ROOT}/runs/agent/muno21_overnight_capacity_matrix_20260802"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_overnight_capacity_matrix_20260802"
GPUS=(1 2 3 4 5 7)
QWEN_SEEDS=(20260909 20260910 20260911)
GEMMA_SEEDS=(20260904 20260905 20260906)

mkdir -p "${ROOT}/status" "${LOG_ROOT}"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${ROOT}/COMPLETE.json" ]] || exit 0

for path in \
  "${PYTHON}" "${QWEN_MODEL}/config.json" "${GEMMA_MODEL}/config.json" \
  "${DATA}/train/sft_composed.jsonl" "${DATA}/val/sft_composed.jsonl"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done

while [[ ! -s "${WAIT_FOR}" ]]; do
  sleep 30
done
for gpu in "${GPUS[@]}"; do
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 30
  done
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" - "${ROOT}/launch_manifest.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "muno21-overnight-capacity-matrix-v1",
    "split": "val",
    "test_assets_read": False,
    "qwen3_8b_seeds": [20260909, 20260910, 20260911],
    "gemma3_4b_seeds": [20260904, 20260905, 20260906],
    "physical_gpus": [1, 2, 3, 4, 5, 7],
    "purpose": "three-seed capacity and backbone control",
}, indent=2) + "\n", encoding="utf-8")
PY

run_qwen() {
  local gpu="$1" seed="$2"
  local label="qwen3_8b_seed${seed}"
  local output="${STORAGE_ROOT}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed${seed}"
  [[ ! -e "${output}" ]] || { echo "refusing existing output: ${output}" >&2; return 4; }
  printf '{"label":"%s","gpu":%s,"status":"running","stage":"train","split":"val","test_assets_read":false}\n' \
    "${label}" "${gpu}" >"${ROOT}/status/${label}.json"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_agent_sft.py \
    "${QWEN_MODEL}" "${DATA}/train/sft_composed.jsonl" "${output}" \
    --eval-jsonl "${DATA}/val/sft_composed.jsonl" \
    --epochs 1 --learning-rate 0.0002 --batch-size 1 \
    --gradient-accumulation 4 --max-length 2048 \
    --logging-steps 5 --eval-steps 100 --save-steps 100 --save-total-limit 4 \
    --early-stopping-patience 2 --early-stopping-threshold 0.0005 \
    --lora-rank 16 --lora-alpha 32 --seed "${seed}" \
    >"${LOG_ROOT}/${label}.log" 2>&1
  printf '{"label":"%s","gpu":%s,"status":"running","stage":"val_actions","split":"val","test_assets_read":false}\n' \
    "${label}" "${gpu}" >"${ROOT}/status/${label}.json"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_agent_actions.py \
    "${QWEN_MODEL}" "${DATA}/val/sft_composed.jsonl" "${output}/evaluation/overnight_val" \
    --adapter "${output}/final" --device cuda --batch-size 1 \
    --max-length 2048 --max-new-tokens 64 \
    >>"${LOG_ROOT}/${label}.log" 2>&1
  printf '{"label":"%s","gpu":%s,"status":"complete","stage":"done","exit_code":0,"split":"val","test_assets_read":false}\n' \
    "${label}" "${gpu}" >"${ROOT}/status/${label}.json"
}

run_gemma() {
  local gpu="$1" seed="$2"
  local label="gemma3_4b_seed${seed}"
  printf '{"label":"%s","gpu":%s,"status":"running","stage":"train_and_val","split":"val","test_assets_read":false}\n' \
    "${label}" "${gpu}" >"${ROOT}/status/${label}.json"
  GPU="${gpu}" SEED="${seed}" MODEL="${GEMMA_MODEL}" \
    RUN_FAMILY="muno21_direct_vlm_gemma3_4b_sft_v1" \
    bash scripts/queue_muno21_direct_vlm_gemma_sft.sh \
    >"${LOG_ROOT}/${label}.log" 2>&1
  printf '{"label":"%s","gpu":%s,"status":"complete","stage":"done","exit_code":0,"split":"val","test_assets_read":false}\n' \
    "${label}" "${gpu}" >"${ROOT}/status/${label}.json"
}

pids=()
for index in 0 1 2; do
  (set -e; run_qwen "${GPUS[${index}]}" "${QWEN_SEEDS[${index}]}") &
  pids+=("$!")
  sleep 10
done
for index in 0 1 2; do
  gpu_index=$((index + 3))
  (set -e; run_gemma "${GPUS[${gpu_index}]}" "${GEMMA_SEEDS[${index}]}") &
  pids+=("$!")
  sleep 10
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then
  printf '{"status":"failed","exit_code":1,"split":"val","test_assets_read":false}\n' >"${ROOT}/FAILED.json"
  exit 1
fi
printf '{"status":"complete","jobs":6,"split":"val","test_assets_read":false}\n' >"${ROOT}/COMPLETE.json"
