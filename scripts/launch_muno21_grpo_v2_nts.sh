#!/usr/bin/env bash
set -euo pipefail

cd /home/wh/projects/activemap-v1-joint-debug
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

PY=/home/wh/venvs/activemap/bin/python
ROOT=/mnt/mydisk/wh/ActiveMap/runs/agent
ADAPTER=${ROOT}/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808
BASE=${ROOT}/muno21_proxy_grpo_full_g8_t2p0_v1
SUP=${ROOT}/muno21_proxy_grpo_full_g8_t2p0_supplement_v1
OUT=${ROOT}/muno21_proxy_grpo_v2_sequence_clip_nts_v1
LOGROOT=/mnt/mydisk/wh/ActiveMap/logs/muno21_proxy_grpo_v2_sequence_clip_nts_v1
mkdir -p "${OUT}" "${LOGROOT}"

rollout_args=()
for run in \
  "${BASE}/rollout1_seed20260882" \
  "${BASE}/rollout2_seed20260883" \
  "${BASE}/rollout3_seed20260884" \
  "${BASE}/rollout4_seed20260885" \
  "${BASE}/rollout5_seed20260886" \
  "${BASE}/rollout7_seed20260888" \
  "${SUP}/supplement0_seed20260889" \
  "${SUP}/supplement1_seed20260890"; do
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

launch_one() {
  local gpu=$1
  local seed=$2
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${ADAPTER}" "${OUT}/constrained_seed${seed}" \
    "${rollout_args[@]}" \
    --objective sequence-clip \
    --dynamic-sampling variable-only \
    --false-edit-lagrange 0.5 \
    --false-edit-target 0.0966 \
    --false-edit-dual-lr 1.0 \
    --learning-rate 2.5e-7 \
    --entropy-coef 0.001 \
    --gradient-accumulation 4 \
    --seed "${seed}" >"${LOGROOT}/constrained_seed${seed}.log" 2>&1 &
  echo "$! GPU${gpu} constrained seed${seed}"
}

launch_one 1 20260947
launch_one 2 20260948
wait
