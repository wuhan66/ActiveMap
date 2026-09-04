#!/usr/bin/env bash
set -euo pipefail

# CPU-only gate combiner. It records whether the pointer-action branch can
# proceed; it never starts GRPO by itself.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_rollouts_v3}"
VALIDATION_ROOT="${VALIDATION_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_validation_v3}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_preflight_v3.json}"
LOG="${LOG:-${STORE}/logs/muno21_reachable_tool_pointer_preflight_v3.log}"

[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 2; }
mkdir -p "$(dirname "${LOG}")"
cd "${PROJECT}"
until [[ -s "${ROLLOUT_ROOT}/diversity_audit.json" ]] && \
  [[ -s "${VALIDATION_ROOT}/summary.json" ]]; do sleep 60; done

"${PY}" scripts/audit_pointer_action_validation.py \
  "${VALIDATION_ROOT}/summary.json" "${VALIDATION_ROOT}/pointer_action_audit.json" \
  >"${LOG}" 2>&1
"${PY}" scripts/combine_pointer_action_gates.py \
  "${ROLLOUT_ROOT}/diversity_audit.json" \
  "${VALIDATION_ROOT}/pointer_action_audit.json" "${OUTPUT}" \
  >>"${LOG}" 2>&1

echo "Pointer-action preflight complete: ${OUTPUT}"
