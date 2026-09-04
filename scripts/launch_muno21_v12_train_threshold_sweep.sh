#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
GPU_IDS="${MUNO21_V12_GPU_IDS:-1,3,4,5}"
THRESHOLDS="${MUNO21_V12_THRESHOLDS:-0.03,0.05,0.07,0.09}"
SEED="${MUNO21_AGENT_SEED:-20260822}"
RUN_FAMILY="${MUNO21_V12_RUN_FAMILY:-agent_v12_proactive_train_threshold_sweep_v1}"
IFS=',' read -r -a gpus <<<"${GPU_IDS}"
IFS=',' read -r -a thresholds <<<"${THRESHOLDS}"

[[ "${#gpus[@]}" -eq "${#thresholds[@]}" ]] || {
  echo "GPU and threshold counts must match" >&2
  exit 2
}
[[ "${#gpus[@]}" -le 4 ]] || {
  echo "launcher is capped at four additional GPUs" >&2
  exit 2
}

cd "${PROJECT_ROOT}"
source scripts/assert_allowed_gpu.sh
run="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${SEED}"
promoted="${run}/evaluation/selection/promoted_adapter.json"
[[ -s "${promoted}" ]] || {
  echo "missing promoted adapter record: ${promoted}" >&2
  exit 3
}
adapter="$(
  "${STORAGE_ROOT}/envs/activemap-agent/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["adapter_path"])' \
    "${promoted}"
)"
mkdir -p "${STORAGE_ROOT}/logs/${RUN_FAMILY}"

for index in "${!thresholds[@]}"; do
  gpu="${gpus[$index]}"
  threshold="${thresholds[$index]}"
  tag="${threshold/./p}"
  activemap_assert_allowed_gpu "${gpu}"
  output="${STORAGE_ROOT}/artifacts/paper_rollouts/${RUN_FAMILY}/threshold_${tag}"
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite ${output}" >&2
    exit 3
  }
  log="${STORAGE_ROOT}/logs/${RUN_FAMILY}/threshold_${tag}.log"
  nohup env \
    MUNO21_V12_GPU="${gpu}" \
    MUNO21_AGENT_SEED="${SEED}" \
    MUNO21_V12_ADAPTER="${adapter}" \
    MUNO21_V12_SPLIT=train \
    MUNO21_V12_TOOL_NEED_THRESHOLD="${threshold}" \
    MUNO21_V12_ROLLOUT_ROOT="${output}" \
    MUNO21_V12_METHODS="qwen3_4b_sft_calibrated_tool_to_belief,edit_conditioned_proactive_tools" \
    MUNO21_V12_ASSESS=0 \
    bash scripts/run_muno21_v12_proactive_rollout.sh \
    >"${log}" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "${pid}" >"${STORAGE_ROOT}/logs/${RUN_FAMILY}/threshold_${tag}.pid"
  echo "started threshold=${threshold} gpu=${gpu} pid=${pid}"
done
