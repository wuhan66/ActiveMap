#!/usr/bin/env bash
set -euo pipefail

cd /home/wh/projects/activemap-v1
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

PY=/home/wh/ActiveMap/envs/activemap-agent/bin/python
ADAPTER=/home/wh/ActiveMap/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808
ROOT=/home/wh/ActiveMap/runs/agent
OUT=${ROOT}/muno21_proxy_grpo_v2_temperature_ablation_v1
LOGROOT=/home/wh/ActiveMap/logs/muno21_proxy_grpo_v2_temperature_ablation_v1
mkdir -p "${OUT}" "${LOGROOT}"

launch_temperature() {
  local gpu=$1
  local tag=$2
  local source=$3
  local seed=$4
  local args=()
  for rollout in 0 1 2 3; do
    local run
    run=$(find "${source}" -maxdepth 1 -type d -name "rollout${rollout}_seed*" | head -n 1)
    [[ -n "${run}" ]]
    args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
      "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
  done
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${ADAPTER}" "${OUT}/${tag}_seed${seed}" \
    "${args[@]}" \
    --objective sequence-clip \
    --dynamic-sampling variable-only \
    --false-edit-lagrange 0.5 \
    --false-edit-target 0.0966 \
    --false-edit-dual-lr 1.0 \
    --learning-rate 2.5e-7 \
    --entropy-coef 0.001 \
    --gradient-accumulation 4 \
    --seed "${seed}" >"${LOGROOT}/${tag}_seed${seed}.log" 2>&1 &
  echo "$! GPU${gpu} ${tag}"
}

launch_temperature 0 t1p8 "${ROOT}/muno21_temperature_t1p8_clean_g4_v1" 20260951
launch_temperature 6 t2p2 "${ROOT}/muno21_temperature_t2p2_clean_g4_v1" 20260952
wait
