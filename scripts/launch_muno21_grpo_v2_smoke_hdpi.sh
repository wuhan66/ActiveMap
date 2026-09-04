#!/usr/bin/env bash
set -euo pipefail

cd /home/wh/projects/activemap-v1
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY=/home/wh/ActiveMap/envs/activemap-agent/bin/python
ADAPTER=/home/wh/ActiveMap/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808
ROOT=/home/wh/ActiveMap/runs/agent
BASE=${ROOT}/muno21_proxy_grpo_full_g8_t2p0_v1
SUP=${ROOT}/muno21_proxy_grpo_full_g8_t2p0_supplement_v1
OUT=${ROOT}/muno21_proxy_grpo_v2_sequence_clip_smoke_seed20260931

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
  rollout_args+=(
    --rollout
    "${run}/qwen3_4b_sft_tool_to_belief.jsonl"
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  )
done

"${PY}" scripts/train_recurrent_proxy_grpo.py \
  "${ADAPTER}" "${OUT}" \
  "${rollout_args[@]}" \
  --objective sequence-clip \
  --dynamic-sampling variable-only \
  --false-edit-lagrange 0.5 \
  --false-edit-target 0.0966 \
  --false-edit-dual-lr 1.0 \
  --learning-rate 2.5e-7 \
  --entropy-coef 0.001 \
  --gradient-accumulation 4 \
  --max-groups 8 \
  --seed 20260931
