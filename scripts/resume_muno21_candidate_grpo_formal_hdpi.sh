#!/usr/bin/env bash
set -euo pipefail

# Recover the formal candidate-GRPO run after one training slot was occupied.
# Existing artifacts are never overwritten. GPU5 retries the missing seed after
# the matched SFT baselines finish; GPUs1/4 retain their original jobs.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807/sft_seed20260897/final}"
SOURCE_ROOT="${SOURCE_ROOT:-${STORE}/runs/agent/muno21_candidate_decoder_reentry_v2_20260807}"
SFT_REPLAY="${SFT_REPLAY:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_data_20260807_r1/train.jsonl}"
TOOL_VAL="${TOOL_VAL:-${STORE}/runs/agent/reachable_tool_sft_v1/val.jsonl}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_candidate_grpo_formal_3seed_v1_20260807}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_candidate_grpo_formal_3seed_v1_20260807}"
RETRY52="${ROOT}/grpo/train_seed20260952_retry1"

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${ROOT}/paired_reports" "${LOGROOT}"
date -Is >"${ROOT}/RECOVERY_STARTED"

wait_for_file() {
  local path="$1"
  while [[ ! -s "${path}" ]]; do
    sleep 30
  done
}

rollout_args=()
writeback_args=()
for index in 0 1 2 3; do
  seed=$((20260920 + index))
  run="${SOURCE_ROOT}/candidate_rollouts_g4/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
  writeback_args+=(--writeback "${run}/writeback/writeback.jsonl")
done

evaluate_policy() {
  local gpu="$1" name="$2" adapter="$3" eval_seed="$4"
  local output="${ROOT}/${name}/eval_seed${eval_seed}"
  if [[ -s "${output}/COMPLETED" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${adapter}" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --tool-belief-checkpoint "${BELIEF}" --tool-supervision-jsonl "${TOOL_VAL}" \
    --max-tool-calls 2 --tool-out-size 256 --split val --oracle-step 1 \
    --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu \
    --seed "${eval_seed}" --sample-order seeded-hash --sample-seed 20260882 \
    --ensure-tool-positive --limit 33 --do-sample --temperature 1.5 --top-p 0.95 \
    --action-decoder candidate-sample --candidate-score-batch-size 1 \
    --methods qwen3_4b_sft_tool_to_belief \
    >"${LOGROOT}/${name}_eval_seed${eval_seed}_recovery.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${output}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${output}/writeback" --device cuda:0 --split val --image-size 512 \
    --threshold 0.5 --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    >"${LOGROOT}/${name}_eval_seed${eval_seed}_writeback_recovery.log" 2>&1
  date -Is >"${output}/COMPLETED"
}

# GPU5 is released only after all three matched SFT evaluations complete.
for eval_seed in 20260961 20260962 20260963; do
  wait_for_file "${ROOT}/sft_baseline/eval_seed${eval_seed}/COMPLETED"
done

if [[ ! -s "${RETRY52}/final/adapter_model.safetensors" ]]; then
  CUDA_VISIBLE_DEVICES=5 "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT}" "${RETRY52}" "${rollout_args[@]}" "${writeback_args[@]}" \
    --reward-mode executable --objective candidate-clip \
    --candidate-score-batch-size 1 --dynamic-sampling variable-only \
    --minimum-reward-std 1e-6 --minimum-keep-trajectories 16 \
    --minimum-commit-trajectories 16 --minimum-tool-trajectories 16 \
    --minimum-false-edit-trajectories 16 --false-edit-lagrange 0.25 \
    --false-edit-target 0.10 --false-edit-dual-lr 0.5 \
    --learning-rate 2.5e-7 --epochs 1 --gradient-accumulation 2 \
    --entropy-coef 0.001 --sequence-kl-coef 0.01 \
    --sft-replay "${SFT_REPLAY}" --sft-replay-weight 0.10 \
    --sft-replay-probability 0.50 --sft-tool-replay-fraction 0.50 \
    --seed 20260952 >"${LOGROOT}/train_seed20260952_retry1.log" 2>&1
fi

wait_for_file "${ROOT}/grpo/train_seed20260951/final/adapter_model.safetensors"
wait_for_file "${ROOT}/grpo/train_seed20260953/final/adapter_model.safetensors"

evaluate_policy 1 grpo_seed20260951 "${ROOT}/grpo/train_seed20260951/final" 20260961 & e1=$!
evaluate_policy 5 grpo_seed20260952 "${RETRY52}/final" 20260962 & e2=$!
evaluate_policy 4 grpo_seed20260953 "${ROOT}/grpo/train_seed20260953/final" 20260963 & e3=$!
wait "${e1}" "${e2}" "${e3}"

for index in 0 1 2; do
  train_seed=$((20260951 + index))
  eval_seed=$((20260961 + index))
  bootstrap_seed=$((20261061 + index))
  "${PY}" scripts/compare_agent_writebacks.py \
    "${ROOT}/sft_baseline/eval_seed${eval_seed}/writeback/writeback.jsonl" \
    "${ROOT}/grpo_seed${train_seed}/eval_seed${eval_seed}/writeback/writeback.jsonl" \
    "${ROOT}/paired_reports/seed${train_seed}.json" \
    --bootstrap 5000 --seed "${bootstrap_seed}" --model-seed "${train_seed}" \
    --evaluation-seed "${eval_seed}" --group-key task_id --split val \
    >"${LOGROOT}/paired_seed${train_seed}.log" 2>&1
done

"${PY}" scripts/aggregate_muno21_reachable_tool_grpo_three_seed.py \
  "${ROOT}/paired_reports/seed20260951.json" \
  "${ROOT}/paired_reports/seed20260952.json" \
  "${ROOT}/paired_reports/seed20260953.json" \
  "${ROOT}/three_seed_aggregate.json" \
  >"${LOGROOT}/three_seed_aggregate.log" 2>&1

"${PY}" scripts/summarize_muno21_candidate_grpo_formal.py \
  "${ROOT}/three_seed_aggregate.json" \
  "${ROOT}/formal_result_summary.json" \
  "${ROOT}/formal_result.md" \
  >"${LOGROOT}/formal_result_summary.log" 2>&1

date -Is >"${ROOT}/RECOVERY_COMPLETED"
date -Is >"${ROOT}/FORMAL_RESULTS_READY"
