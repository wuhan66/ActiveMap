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
BASE=${ROOT}/muno21_proxy_grpo_full_g8_t2p0_v1
SUP=${ROOT}/muno21_proxy_grpo_full_g8_t2p0_supplement_v1
MATRIX=${ROOT}/muno21_proxy_grpo_v2_sequence_clip_matrix_v1
LOGROOT=/home/wh/ActiveMap/logs/muno21_proxy_grpo_v2_sequence_clip_matrix_v1
mkdir -p "${MATRIX}" "${LOGROOT}"

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

launch_one() {
  local gpu=$1
  local variant=$2
  local seed=$3
  local lagrange=$4
  local output=${MATRIX}/${variant}_seed${seed}
  local log=${LOGROOT}/${variant}_seed${seed}.log
  local constraint_args=(--false-edit-lagrange "${lagrange}")
  if [[ "${variant}" == "constrained" ]]; then
    constraint_args+=(--false-edit-target 0.0966 --false-edit-dual-lr 1.0)
  fi
  if [[ -e "${output}" ]]; then
    echo "Refusing existing output: ${output}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${ADAPTER}" "${output}" \
    "${rollout_args[@]}" \
    --objective sequence-clip \
    --dynamic-sampling variable-only \
    "${constraint_args[@]}" \
    --learning-rate 2.5e-7 \
    --entropy-coef 0.001 \
    --gradient-accumulation 4 \
    --seed "${seed}" >"${log}" 2>&1 &
  echo "$! ${gpu} ${variant} ${seed}"
}

launch_one 1 constrained 20260941 0.5
launch_one 2 constrained 20260942 0.5
launch_one 3 constrained 20260943 0.5
launch_one 4 unconstrained 20260944 0.0
launch_one 5 unconstrained 20260945 0.0
launch_one 7 unconstrained 20260946 0.0
wait
