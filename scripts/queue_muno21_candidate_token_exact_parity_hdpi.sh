#!/usr/bin/env bash
set -euo pipefail

# Wait for the token-exact reward audit, then recompute the behavior policy
# distribution under the trainable adapter. This queue never updates weights.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
GPU="${GPU:-5}"
SOURCE="${SOURCE:-${STORE}/runs/agent/muno21_candidate_token_exact_gpu5_20260808}"
ADAPTER="${ADAPTER:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807/sft_seed20260897/final}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_candidate_token_exact_parity_gpu5_20260808}"
LOG="${LOG:-${STORE}/logs/muno21_candidate_token_exact_parity_gpu5_20260808.log}"
POLL_SECONDS="${POLL_SECONDS:-30}"

[[ "${GPU}" != "0" && "${GPU}" != "2" ]] || {
  echo "GPU ${GPU} is excluded" >&2
  exit 3
}
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

while [[ ! -s "${SOURCE}/READY_FOR_CANDIDATE_GRPO_SMOKE" ]]; do
  if ! pgrep -f "queue_muno21_candidate_token_exact_gpu5_hdpi.sh" >/dev/null; then
    echo "source queue exited without readiness marker" >&2
    exit 21
  fi
  sleep "${POLL_SECONDS}"
done

[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing existing parity output: ${OUTPUT}" >&2
  exit 22
}

rollout_args=()
writeback_args=()
seeds=(20260930 20260931 20260932 20260933)
for index in 0 1 2 3; do
  seed="${seeds[$index]}"
  root="${SOURCE}/candidate_rollouts_g4/rollout${index}_seed${seed}"
  rollout="${root}/qwen3_4b_sft_tool_to_belief.jsonl"
  calls="${root}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  writeback="${root}/writeback/writeback.jsonl"
  for path in "${rollout}" "${calls}" "${writeback}"; do
    test -s "${path}"
  done
  rollout_args+=(--rollout "${rollout}" "${calls}")
  writeback_args+=(--writeback "${writeback}")
done

mkdir -p "$(dirname "${LOG}")"
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/train_recurrent_proxy_grpo.py \
  "${ADAPTER}" "${OUTPUT}" \
  "${rollout_args[@]}" "${writeback_args[@]}" \
  --reward-mode executable --objective candidate-clip \
  --candidate-score-batch-size 1 --dynamic-sampling variable-only \
  --minimum-keep-trajectories 16 --minimum-commit-trajectories 16 \
  --minimum-tool-trajectories 16 --audit-only --verify-candidate-parity \
  >"${LOG}" 2>&1

test -s "${OUTPUT}/PARITY_VERIFIED"
cp "${OUTPUT}/PARITY_VERIFIED" "${OUTPUT}/READY_FOR_CONSTRAINED_GRPO_SMOKE"
echo "candidate behavior/update parity verified; no weights updated"
