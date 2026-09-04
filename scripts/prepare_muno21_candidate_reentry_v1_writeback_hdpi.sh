#!/usr/bin/env bash
set -euo pipefail

# Score the already collected train-only categorical candidate rollouts.
# This queue is diagnostic: it performs no policy update and reads no test data.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
GPU="${GPU:-5}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_candidate_decoder_reentry_v1_20260807}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
SFT="${SFT:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807/sft_seed20260897/final}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_candidate_decoder_reentry_v1_20260807}"
SEEDS=(20260910 20260911 20260912 20260913)

[[ "${GPU}" != "0" && "${GPU}" != "2" ]] || { echo "GPU ${GPU} is excluded" >&2; exit 3; }
for path in "${PY}" "${UPDATER}" "${EPISODES}" "${SFT}/adapter_model.safetensors"; do test -s "${path}"; done
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
mkdir -p "${LOGROOT}"

rollout_args=()
writeback_args=()
for index in 0 1 2 3; do
  seed="${SEEDS[$index]}"
  run="${ROOT}/candidate_rollouts_g4/rollout${index}_seed${seed}"
  rollout="${run}/qwen3_4b_sft_tool_to_belief.jsonl"
  calls="${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  output="${run}/writeback"
  test -s "${rollout}"; test -s "${calls}"
  if [[ ! -s "${output}/writeback.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/evaluate_agent_map_writeback.py \
      "${UPDATER}" "${EPISODES}" "${rollout}" "${output}" \
      --device cuda:0 --split train --image-size 512 --threshold 0.5 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
      >"${LOGROOT}/writeback${index}_seed${seed}.log" 2>&1
  fi
  rollout_args+=(--rollout "${rollout}" "${calls}")
  writeback_args+=(--writeback "${output}/writeback.jsonl")
done

"${PY}" scripts/train_recurrent_proxy_grpo.py "${SFT}" \
  "${ROOT}/executable_reward_audit" "${rollout_args[@]}" "${writeback_args[@]}" \
  --reward-mode executable --objective candidate-clip --dynamic-sampling variable-only \
  --minimum-keep-trajectories 16 --minimum-commit-trajectories 16 \
  --minimum-tool-trajectories 16 --audit-only \
  >"${LOGROOT}/executable_reward_audit.log" 2>&1
date -Is >"${ROOT}/READY_FOR_CANDIDATE_GRPO_SMOKE"
echo "candidate re-entry executable reward audit complete"
