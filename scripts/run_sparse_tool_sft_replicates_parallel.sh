#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -n "${ACTIVEMAP_SERVER_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${ACTIVEMAP_SERVER_ENV}"
else
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
fi
export ACTIVEMAP_LAUNCHER_NAME="run_sparse_tool_sft_replicates_parallel.sh"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"

STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
SEED_LIST="${MUNO21_REPLICATE_SEEDS:-20260822 20260823}"
GPU_LIST="${MUNO21_REPLICATE_GPUS:-0 1}"
RUN_FAMILY="${MUNO21_AGENT_RUN_FAMILY:-muno21_qwen3_4b_sparse_tool_sft}"
START_SCRIPT="${MUNO21_AGENT_START_SCRIPT:-start_muno21_agent_sparse_tool_sft.sh}"
read -r -a seeds <<<"${SEED_LIST}"
read -r -a gpus <<<"${GPU_LIST}"

[[ "${#seeds[@]}" -eq 2 && "${#gpus[@]}" -eq 2 ]] || {
  echo "exactly two replicate seeds and two GPUs are required" >&2
  exit 2
}
[[ "${seeds[0]}" != "${seeds[1]}" && "${gpus[0]}" != "${gpus[1]}" ]] || {
  echo "replicate seeds and GPUs must be unique" >&2
  exit 2
}

allowed=",${ACTIVEMAP_GPU_IDS},"
for gpu in "${gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ && "${allowed}" == *",${gpu},"* ]] || {
    echo "GPU ${gpu} is outside ACTIVEMAP_GPU_IDS=${ACTIVEMAP_GPU_IDS}" >&2
    exit 2
  }
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} has active compute processes: ${pids}" >&2
    exit 3
  }
done

training_pids=()
for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  run="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}"
  promotion="${run}/evaluation/selection/promoted_adapter.json"
  if [[ -s "${promotion}" ]]; then
    echo "seed ${seed} already promoted; reusing"
    training_pids+=("")
    continue
  fi
  if [[ ! -s "${run}/final/adapter_config.json" ]]; then
    MUNO21_AGENT_SEED="${seed}" MUNO21_AGENT_GPU="${gpu}" \
    MUNO21_AGENT_RUN_DIR="${run}" \
      bash "${PROJECT_ROOT}/scripts/${START_SCRIPT}"
  fi
  [[ -s "${run}/train.pid" ]] || {
    echo "seed ${seed} has neither a promoted adapter nor a training PID" >&2
    exit 4
  }
  training_pids+=("$(cat "${run}/train.pid")")
done

while true; do
  active=0
  for index in "${!training_pids[@]}"; do
    pid="${training_pids[$index]}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[$(date --iso-8601=seconds)] seed ${seeds[$index]} training PID ${pid}"
      active=$((active + 1))
    fi
  done
  (( active > 0 )) || break
  sleep 60
done

evaluation_pids=()
for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  run="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}"
  [[ -s "${run}/final/adapter_config.json" ]] || {
    echo "SFT seed ${seed} failed before final adapter creation" >&2
    exit 5
  }
  MUNO21_AGENT_SEED="${seed}" MUNO21_AGENT_GPU="${gpu}" \
  MUNO21_AGENT_RUN_DIR="${run}" \
    bash "${PROJECT_ROOT}/scripts/evaluate_promote_sparse_tool_sft.sh" &
  evaluation_pids+=("$!")
done

failed=0
for index in "${!evaluation_pids[@]}"; do
  if ! wait "${evaluation_pids[$index]}"; then
    echo "validation promotion failed for seed ${seeds[$index]}" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 6

for seed in "${seeds[@]}"; do
  promotion="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}/evaluation/selection/promoted_adapter.json"
  [[ -s "${promotion}" ]] || { echo "missing promotion: ${promotion}" >&2; exit 7; }
done
echo "[$(date --iso-8601=seconds)] both MUNO21 replicate seeds promoted"
