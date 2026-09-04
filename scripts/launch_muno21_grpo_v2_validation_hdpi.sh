#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PY=${STORE}/envs/activemap-agent/bin/python
MODEL=/home/wh/hf_models/Qwen3-4B
MATRIX=${STORE}/runs/agent/muno21_proxy_grpo_v2_sequence_clip_matrix_v1
TEMP=${STORE}/runs/agent/muno21_proxy_grpo_v2_temperature_ablation_v1
STATES=${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl
EPISODES=${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl
BELIEF=${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt
SELECTOR=${STORE}/runs/selector
cd "${PROJECT}"

validate() {
  local gpu=$1
  local run=$2
  local seed=$3
  local output=${run}/validation_full_gpu_selector
  CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=src:. \
    "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${run}/final" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 512 --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    >"${run}/validation_full_gpu_selector.log" 2>&1 &
  echo "$! GPU${gpu} $(basename "${run}")"
}

validate 0 "${TEMP}/t1p8_seed20260951" 20260951
validate 1 "${MATRIX}/constrained_seed20260941" 20260941
validate 2 "${MATRIX}/constrained_seed20260942" 20260942
validate 3 "${MATRIX}/constrained_seed20260943" 20260943
validate 4 "${MATRIX}/unconstrained_seed20260944" 20260944
validate 5 "${MATRIX}/unconstrained_seed20260945" 20260945
validate 6 "${TEMP}/t2p2_seed20260952" 20260952
validate 7 "${MATRIX}/unconstrained_seed20260946" 20260946
wait
