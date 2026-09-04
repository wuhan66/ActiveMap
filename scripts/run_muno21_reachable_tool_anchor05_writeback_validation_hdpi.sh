#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
STABILITY_ROOT="${STABILITY_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_grpo_stability_ablation_validation_v1}"
REPLICATION_ROOT="${REPLICATION_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_grpo_anchor05_replication_validation_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_anchor05_writeback_validation_v1}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_reachable_tool_support_anchor05_writeback_validation_v1}"

for path in "${PY}" "${UPDATER}" "${EPISODES}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}" "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

launch_one() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local rollout_root="$4"
  local rollout="${rollout_root}/${variant}_seed${seed}/qwen3_4b_sft_tool_to_belief.jsonl"
  local output="${OUTPUT_ROOT}/${variant}_seed${seed}"
  local log="${LOGROOT}/${variant}_seed${seed}.log"
  if [[ -s "${output}/summary.json" && -s "${output}/writeback.jsonl" ]]; then
    echo "writeback validation already complete: ${variant} seed${seed}"
    return 0
  fi
  [[ -s "${rollout}" ]] || { echo "Missing rollout: ${rollout}" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    nohup "${PY}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${rollout}" "${output}" \
    --device cuda:0 --split val --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --protocol-name "muno21-anchor05-real-writeback-val-v1" \
    >"${log}" 2>&1 &
  echo "$! GPU${gpu} writeback ${variant} seed${seed}"
}

# GPU0/6 are intentionally excluded. The validation is executable map quality,
# so all candidate/control rows use the same frozen updater and val episodes.
launch_one 1 anchor05 20261230 "${STABILITY_ROOT}"
launch_one 2 sft_control 20261230 "${STABILITY_ROOT}"
launch_one 3 anchor05 20261232 "${REPLICATION_ROOT}"
launch_one 4 sft_control 20261232 "${REPLICATION_ROOT}"
launch_one 5 anchor05 20261233 "${REPLICATION_ROOT}"
launch_one 7 sft_control 20261233 "${REPLICATION_ROOT}"
wait

for variant in anchor05 sft_control; do
  for seed in 20261230 20261232 20261233; do
    output="${OUTPUT_ROOT}/${variant}_seed${seed}"
    [[ -s "${output}/summary.json" && -s "${output}/writeback.jsonl" ]] || {
      echo "writeback validation did not complete: ${variant} seed${seed}" >&2
      exit 4
    }
  done
done

echo "anchor05 real validation writebacks completed"
