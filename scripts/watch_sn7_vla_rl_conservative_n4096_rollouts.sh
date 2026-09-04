#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
SFT="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
POLL_SECONDS="${POLL_SECONDS:-60}"
SEED=20260718

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

names=(
  contextual_rl_conservative_lr2p5e6_kl005_bal50_n4096
  contextual_rl_conservative_lr2p5e6_kl010_bal50_n4096
  contextual_rl_conservative_lr5e6_kl010_bal50_n4096
  contextual_rl_conservative_lr5e6_kl005_bal25_n4096
)
labels=(lr2p5_kl005 lr2p5_kl010 lr5_kl010 lr5_bal25)
gpus=(0 1 2 3)
outputs=(
  closed_loop_rl_conservative_lr2p5e6_kl005_bal50_n4096_seed20260718_n512
  closed_loop_rl_conservative_lr2p5e6_kl010_bal50_n4096_seed20260718_n512
  closed_loop_rl_conservative_lr5e6_kl010_bal50_n4096_seed20260718_n512
  closed_loop_rl_conservative_lr5e6_kl005_bal25_n4096_seed20260718_n512
)

for name in "${names[@]}"; do
  result="${RUN}/${name}/process_result.json"
  until [[ -s "${result}" ]]; do
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
  local adapter="${RUN}/${names[${index}]}/seed${SEED}/final"
  local output="${RUN}/${outputs[${index}]}"
  until gpu_is_idle "${gpu}"; do
    sleep "${POLL_SECONDS}"
  done
  [[ ! -e "${output}" ]] || {
    echo "refusing existing RL rollout: ${output}" >&2
    return 1
  }
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${adapter}" \
    "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
    "${DATA}/closed_loop_v1/episodes_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${output}" --gpu "${gpu}" --seed "${SEED}" \
    --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
    --bootstrap-repetitions 500 --limit 512 --monitor-interval 5
}

pids=()
for index in 0 1 2 3; do
  run_rollout "${index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"

sft_output="${RUN}/closed_loop_sft_seed20260718_n512"
until gpu_is_idle 0; do
  sleep "${POLL_SECONDS}"
done
if [[ ! -e "${sft_output}" ]]; then
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${SFT}" \
    "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
    "${DATA}/closed_loop_v1/episodes_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${sft_output}" --gpu 0 --seed "${SEED}" \
    --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
    --bootstrap-repetitions 500 --limit 512 --monitor-interval 5
fi

summary_json="${RUN}/closed_loop_rl_conservative_n4096_seed20260718_n512.json"
summary_md="${RUN}/closed_loop_rl_conservative_n4096_seed20260718_n512.md"
"${PYTHON}" scripts/summarize_sn7_vla_rl_reward_scale.py \
  "${sft_output}/evaluation/summary.json" "${summary_json}" \
  --candidate lr2p5_kl005 "${RUN}/${outputs[0]}/evaluation/summary.json" \
  --candidate lr2p5_kl010 "${RUN}/${outputs[1]}/evaluation/summary.json" \
  --candidate lr5_kl010 "${RUN}/${outputs[2]}/evaluation/summary.json" \
  --candidate lr5_bal25 "${RUN}/${outputs[3]}/evaluation/summary.json" \
  --output-markdown "${summary_md}"

comparison="${RUN}/closed_loop_rl_conservative_n4096_seed20260718_n512_paired.json"
"${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
  "${comparison}" --reference sft --repetitions 2000 --seed "${SEED}" \
  --records "sft=${sft_output}/evaluation/traces.jsonl" \
  --records "lr2p5_kl005=${RUN}/${outputs[0]}/evaluation/traces.jsonl" \
  --records "lr2p5_kl010=${RUN}/${outputs[1]}/evaluation/traces.jsonl" \
  --records "lr5_kl010=${RUN}/${outputs[2]}/evaluation/traces.jsonl" \
  --records "lr5_bal25=${RUN}/${outputs[3]}/evaluation/traces.jsonl"
