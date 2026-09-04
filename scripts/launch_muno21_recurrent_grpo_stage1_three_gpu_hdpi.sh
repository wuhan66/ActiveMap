#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
ADAPTER="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR_ROOT="${STORE}/runs/selector"
ROOT="${STORE}/runs/agent/muno21_recurrent_grpo_stage1_g4_v4"
LOGS="${STORE}/logs/muno21_recurrent_grpo_stage1_g4_v4"
TOOL_SUPERVISION="${MUNO21_TOOL_SUPERVISION:-${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl}"
SAMPLE_LIMIT="${MUNO21_RECURRENT_SAMPLE_LIMIT:-32}"
SAMPLE_SEED="${MUNO21_RECURRENT_SAMPLE_SEED:-20260880}"
TEMPERATURE="${MUNO21_RECURRENT_TEMPERATURE:-2.0}"
TOP_P="${MUNO21_RECURRENT_TOP_P:-1.0}"
GPU_IDS="${MUNO21_RECURRENT_GPU_IDS:-4 7}"
read -r -a gpu_ids <<<"${GPU_IDS}"
[[ "${#gpu_ids[@]}" -ge 2 ]] || {
  echo "MUNO21_RECURRENT_GPU_IDS must contain at least two physical GPUs" >&2
  exit 2
}

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_actor() {
  local gpu="$1" rollout="$2" seed="$3"
  local output="${ROOT}/rollout${rollout}_seed${seed}"
  [[ ! -e "$output" ]] || { echo "Refusing existing rollout: $output" >&2; return 3; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" \
    --adapter "$ADAPTER" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --tool-supervision-jsonl "$TOOL_SUPERVISION" --ensure-tool-positive \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief \
    --split train --budgets 1.5,3.0,4.5 --limit "$SAMPLE_LIMIT" \
    --sample-order seeded-hash --sample-seed "$SAMPLE_SEED" \
    --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    --do-sample --temperature "$TEMPERATURE" --top-p "$TOP_P" \
    --record-training-payload >"${LOGS}/rollout${rollout}_seed${seed}.log" 2>&1
}

pids=()
run_actor "${gpu_ids[0]}" 0 20260831 & pids+=("$!")
run_actor "${gpu_ids[1]}" 1 20260832 & pids+=("$!")
run_actor "${gpu_ids[0]}" 2 20260833 & pids+=("$!")
run_actor "${gpu_ids[1]}" 3 20260834 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
