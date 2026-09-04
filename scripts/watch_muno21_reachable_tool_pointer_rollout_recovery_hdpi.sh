#!/usr/bin/env bash
set -euo pipefail

# Recover the diversity audit after an individual rollout worker was restarted.
# This script never launches policy inference or training by itself.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_rollouts_v3}"
OUTPUT="${OUTPUT:-${ROOT}/diversity_audit.json}"
LOG="${LOG:-${STORE}/logs/muno21_reachable_tool_pointer_rollouts_v3/diversity_audit.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT}"
mkdir -p "$(dirname "${LOG}")"
[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing existing diversity audit: ${OUTPUT}" >&2
  exit 2
}

for index in 0 1 2 3; do
  seed=$((20261251 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  until [[ -s "${run}/qwen3_4b_sft_tool_to_belief.jsonl" ]] && \
    [[ -s "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" ]]; do
    echo "Waiting for recovered rollout${index} seed${seed}"
    sleep "${POLL_SECONDS}"
  done
done

args=()
for index in 0 1 2 3; do
  seed=$((20261251 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

"${PY}" scripts/audit_recurrent_rollout_diversity.py "${OUTPUT}" \
  "${args[@]}" --minimum-variable-group-rate 0.20 --minimum-nonstop-rate 0.05 \
  >"${LOG}" 2>&1

echo "Recovered pointer-action diversity audit completed: ${OUTPUT}"
