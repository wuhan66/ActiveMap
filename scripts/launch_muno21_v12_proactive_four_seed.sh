#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
GPU_IDS="${MUNO21_V12_GPU_IDS:-1,3,4,5}"
SEEDS="${MUNO21_V12_SEEDS:-20260822,20260823,20260824,20260825}"
RUN_FAMILY="${MUNO21_V12_RUN_FAMILY:-agent_v12_proactive_tool_four_seed_v1}"
TOOL_NEED_THRESHOLD="${MUNO21_V12_TOOL_NEED_THRESHOLD:-}"
IFS=',' read -r -a gpus <<<"${GPU_IDS}"
IFS=',' read -r -a seeds <<<"${SEEDS}"

[[ "${#gpus[@]}" -eq "${#seeds[@]}" ]] || {
  echo "GPU and seed counts must match" >&2
  exit 2
}
[[ "${#gpus[@]}" -le 4 ]] || {
  echo "launcher is capped at four additional GPUs" >&2
  exit 2
}

cd "${PROJECT_ROOT}"
source scripts/assert_allowed_gpu.sh
mkdir -p "${STORAGE_ROOT}/logs/${RUN_FAMILY}"

for index in "${!seeds[@]}"; do
  gpu="${gpus[$index]}"
  seed="${seeds[$index]}"
  activemap_assert_allowed_gpu "${gpu}"
  run="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${seed}"
  promoted="${run}/evaluation/selection/promoted_adapter.json"
  [[ -s "${promoted}" ]] || {
    echo "missing promoted adapter record for seed ${seed}: ${promoted}" >&2
    exit 3
  }
  adapter="$(
    "${STORAGE_ROOT}/envs/activemap-agent/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["adapter_path"])' \
      "${promoted}"
  )"
  output="${STORAGE_ROOT}/artifacts/paper_rollouts/${RUN_FAMILY}/seed${seed}"
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite ${output}" >&2
    exit 3
  }
  log="${STORAGE_ROOT}/logs/${RUN_FAMILY}/seed${seed}.log"
  threshold_env=()
  if [[ -n "${TOOL_NEED_THRESHOLD}" ]]; then
    threshold_env+=(MUNO21_V12_TOOL_NEED_THRESHOLD="${TOOL_NEED_THRESHOLD}")
  fi
  nohup env \
    "${threshold_env[@]}" \
    MUNO21_V12_GPU="${gpu}" \
    MUNO21_AGENT_SEED="${seed}" \
    MUNO21_V12_ADAPTER="${adapter}" \
    MUNO21_V12_ROLLOUT_ROOT="${output}" \
    MUNO21_V12_METHODS="qwen3_4b_sft_calibrated_tool_to_belief,edit_conditioned_proactive_tools" \
    MUNO21_V12_ASSESS=0 \
    bash scripts/run_muno21_v12_proactive_rollout.sh \
    >"${log}" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "${pid}" >"${STORAGE_ROOT}/logs/${RUN_FAMILY}/seed${seed}.pid"
  echo "started seed=${seed} gpu=${gpu} pid=${pid}"
done
