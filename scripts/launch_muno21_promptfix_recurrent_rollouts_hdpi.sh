#!/usr/bin/env bash
set -euo pipefail

cd /home/wh/projects/activemap-v1
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

PY=/home/wh/ActiveMap/envs/activemap-agent/bin/python
MODEL=/home/wh/hf_models/Qwen3-4B
STATES=/home/wh/ActiveMap/processed/muno21_v2/agent/selector_states_v1.jsonl
EPISODES=/home/wh/ActiveMap/processed/muno21_v2/agent/episodes_train_val_v1.jsonl
SFT_ADAPTER=/home/wh/ActiveMap/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808
TOOL_BELIEF=/home/wh/ActiveMap/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt
TOOL_SUPERVISION=/home/wh/ActiveMap/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl
ROOT=/home/wh/ActiveMap/runs/agent/muno21_promptfix_recurrent_rollouts_v1
LOGROOT=/home/wh/ActiveMap/logs/muno21_promptfix_recurrent_rollouts_v1
mkdir -p "${LOGROOT}"

launch_one() {
  local gpu=$1
  local index=$2
  local seed=$3
  local output="${ROOT}/rollout${index}_seed${seed}"
  local log="${LOGROOT}/rollout${index}_seed${seed}.log"
  if [[ -e "${output}" ]]; then
    echo "Refusing existing output: ${output}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" \
    --adapter "${SFT_ADAPTER}" \
    --checkpoint /home/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260811/best.pt \
    --checkpoint /home/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260812/best.pt \
    --checkpoint /home/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260813/best.pt \
    --episodes "${EPISODES}" \
    --asset-root-map /mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap \
    --tool-belief-checkpoint "${TOOL_BELIEF}" \
    --tool-supervision-jsonl "${TOOL_SUPERVISION}" \
    --max-tool-calls 2 \
    --tool-out-size 256 \
    --split train \
    --budgets 1.5,3.0,4.5 \
    --device cuda:0 \
    --selector-device cpu \
    --seed "${seed}" \
    --sample-order seeded-hash \
    --sample-seed 20260880 \
    --ensure-tool-positive \
    --limit 32 \
    --do-sample \
    --temperature 1.8 \
    --top-p 1.0 \
    --record-training-payload \
    --methods qwen3_4b_sft_tool_to_belief \
    >"${log}" 2>&1 &
  echo "$! GPU${gpu} rollout${index} seed${seed}"
}

launch_one 1 0 20261101
launch_one 4 1 20261102
launch_one 5 2 20261103
launch_one 7 3 20261104
wait
