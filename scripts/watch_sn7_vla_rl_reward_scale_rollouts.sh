#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA_ROOT="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

names=(
  contextual_rl_diag_scale10_n512
  contextual_rl_diag_scale30_n512
  contextual_rl_diag_scale100_n512
  contextual_rl_diag_scale30_lr1e5_n512
)
gpus=(1 2 4 6)
outputs=(
  closed_loop_rl_scale10_n512_n256
  closed_loop_rl_scale30_n512_n256
  closed_loop_rl_scale100_n512_n256
  closed_loop_rl_scale30_lr1e5_n512_n256
)

for name in "${names[@]}"; do
  result="${RUN_ROOT}/${name}/process_result.json"
  until [[ -s "${result}" ]]; do
    echo "Waiting for ${name}."
    sleep "${POLL_SECONDS}"
  done
  "${PYTHON}" -c \
    "import json; assert json.load(open('${result}'))['status'] == 'completed'"
done

gpu_is_idle() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]]
}

run_rollout() {
  local index="$1"
  local gpu="${gpus[${index}]}"
  local name="${names[${index}]}"
  local output="${RUN_ROOT}/${outputs[${index}]}"
  until gpu_is_idle "${gpu}"; do
    sleep "${POLL_SECONDS}"
  done
  [[ ! -e "${output}" ]] || {
    echo "Refusing to overwrite rollout: ${output}" >&2
    return 1
  }
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${RUN_ROOT}/${name}/seed20260717/final" \
    "${DATA_ROOT}/closed_loop_v1/states_val_step0.jsonl" \
    "${DATA_ROOT}/closed_loop_v1/episodes_val.jsonl" \
    "${DATA_ROOT}/active_catalog_sft_v4/val.jsonl" \
    "${DATA_ROOT}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${output}" \
    --gpu "${gpu}" --seed 20260717 --max-candidates 16 \
    --max-acquisitions 2 --max-new-tokens 64 \
    --bootstrap-repetitions 500 --limit 256 --monitor-interval 5
}

run_rollout 0 &
pid1="$!"
run_rollout 1 &
pid2="$!"
run_rollout 2 &
pid3="$!"
run_rollout 3 &
pid4="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON}" scripts/summarize_sn7_vla_rl_reward_scale.py \
  "${RUN_ROOT}/closed_loop_sft_reference_n256/evaluation/summary.json" \
  "${RUN_ROOT}/closed_loop_rl_reward_scale_n256_20260727.json" \
  --candidate scale10 \
    "${RUN_ROOT}/closed_loop_rl_scale10_n512_n256/evaluation/summary.json" \
  --candidate scale30 \
    "${RUN_ROOT}/closed_loop_rl_scale30_n512_n256/evaluation/summary.json" \
  --candidate scale100 \
    "${RUN_ROOT}/closed_loop_rl_scale100_n512_n256/evaluation/summary.json" \
  --candidate scale30_lr1e5 \
    "${RUN_ROOT}/closed_loop_rl_scale30_lr1e5_n512_n256/evaluation/summary.json" \
  --output-markdown \
    "${RUN_ROOT}/closed_loop_rl_reward_scale_n256_20260727.md"
