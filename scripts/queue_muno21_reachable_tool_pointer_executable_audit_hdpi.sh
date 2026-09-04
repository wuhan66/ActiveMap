#!/usr/bin/env bash
set -euo pipefail

# Prepare executable rewards for the corrected pointer-action branch. This is
# deliberately an audit-only stage: it never updates the policy or reads test.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_rollouts_v3}"
PREFLIGHT="${PREFLIGHT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_preflight_v3.json}"
ADAPTER="${ADAPTER:-${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_pointer_sft_v3_seed20260842/final}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_executable_audit_v3}"
LOG="${LOG:-${STORE}/logs/muno21_reachable_tool_pointer_executable_audit_v3.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT}"
mkdir -p "$(dirname "${LOG}")"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

[[ ! -e "${AUDIT_OUTPUT}" ]] || {
  echo "Refusing existing audit output: ${AUDIT_OUTPUT}" >&2
  exit 2
}
for path in "${PY}" "${UPDATER}" "${EPISODES}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done

echo "Waiting for pointer-action preflight: ${PREFLIGHT}"
until [[ -s "${PREFLIGHT}" ]]; do sleep "${POLL_SECONDS}"; done
"${PY}" -c 'import json,sys; payload=json.load(open(sys.argv[1])); sys.exit(0 if payload.get("ready_for_executable_grpo") is True and payload.get("test_assets_read") is False else 1)' "${PREFLIGHT}" || {
  echo "Pointer-action preflight failed; executable audit was not started." >&2
  exit 4
}

launch_writeback() {
  local gpu="$1" index="$2" seed="$3"
  local rollout="${ROOT}/rollout${index}_seed${seed}/qwen3_4b_sft_tool_to_belief.jsonl"
  local output="${ROOT}/rollout${index}_seed${seed}/writeback"
  [[ -s "${rollout}" ]] || { echo "Missing rollout: ${rollout}" >&2; return 5; }
  if [[ -s "${output}/summary.json" ]]; then
    echo "Reusing complete writeback: ${output}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${rollout}" "${output}" \
    --device cuda:0 --split train --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    >"${output}.log" 2>&1 &
  echo "$! GPU${gpu} rollout${index}"
}

launch_writeback 1 0 20261251
launch_writeback 2 1 20261252
launch_writeback 3 2 20261253
launch_writeback 4 3 20261254
wait

rollout_args=()
writeback_args=()
for index in 0 1 2 3; do
  seed=$((20261251 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  trajectories="${run}/qwen3_4b_sft_tool_to_belief.jsonl"
  calls="${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  writeback="${run}/writeback/writeback.jsonl"
  for path in "${trajectories}" "${calls}" "${writeback}"; do
    [[ -s "${path}" ]] || { echo "Missing executable-audit input: ${path}" >&2; exit 6; }
  done
  rollout_args+=(--rollout "${trajectories}" "${calls}")
  writeback_args+=(--writeback "${writeback}")
done

"${PY}" scripts/train_recurrent_proxy_grpo.py "${ADAPTER}" "${AUDIT_OUTPUT}" \
  "${rollout_args[@]}" "${writeback_args[@]}" \
  --reward-mode executable --objective sequence-clip \
  --dynamic-sampling variable-only --false-edit-lagrange 0.25 \
  --false-edit-target 0.0966 --false-edit-dual-lr 1.0 --audit-only \
  >"${LOG}" 2>&1

echo "Pointer-action executable reward audit completed: ${AUDIT_OUTPUT}"
