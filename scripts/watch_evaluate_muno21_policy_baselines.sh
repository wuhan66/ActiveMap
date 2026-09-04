#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
GPU="${MUNO21_SELECTOR_GPU:-3}"
DPO_RUN="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821}"
RUN_ROOT="${MUNO21_BASELINE_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_policy_baselines_v1}"
DATA_ROOT="/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent"
CHECKPOINT_ROOT="/mnt/mydisk/wh/ActiveMap/runs/selector"
UPDATER="/mnt/mydisk/wh/ActiveMap/runs/updater/muno21_road_v4_explicit_change_scratch_seed20260726/best_val_loss.pt"
EPISODES="${DATA_ROOT}/episodes_train_val_v1.jsonl"
ROLLOUTS="${RUN_ROOT}/rollouts"
STATUS="${RUN_ROOT}/watcher.exit_code"

edit_checkpoints=(
  "${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt"
  "${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt"
  "${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt"
)
generic_checkpoints=(
  "${CHECKPOINT_ROOT}/muno21_evidence_generic_v5_seed20260811/best.pt"
  "${CHECKPOINT_ROOT}/muno21_evidence_generic_v5_seed20260812/best.pt"
  "${CHECKPOINT_ROOT}/muno21_evidence_generic_v5_seed20260813/best.pt"
)

mkdir -p "$RUN_ROOT"
rm -f "$STATUS"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS"' EXIT

required=("${DPO_RUN}/evaluation/watcher.exit_code")
for checkpoint in "${generic_checkpoints[@]}"; do
  required+=("$checkpoint")
done
while true; do
  ready=true
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || ready=false
  done
  [[ "$ready" == "true" ]] && break
  echo "[$(date --iso-8601=seconds)] waiting for DPO evaluation and generic selectors"
  sleep 30
done
[[ "$(cat "${DPO_RUN}/evaluation/watcher.exit_code")" == "0" ]] || exit 2

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -s "${ROLLOUTS}/summary.json" ]]; then
  checkpoint_args=()
  for checkpoint in "${edit_checkpoints[@]}"; do
    checkpoint_args+=(--checkpoint "$checkpoint")
  done
  generic_args=()
  for checkpoint in "${generic_checkpoints[@]}"; do
    generic_args+=(--generic-checkpoint "$checkpoint")
  done
  "$PYTHON" scripts/evaluate_agent_rollouts.py \
    /home/wh/hf_models/Qwen3-4B "${DATA_ROOT}/selector_states_v1.jsonl" "$ROLLOUTS" \
    "${checkpoint_args[@]}" "${generic_args[@]}" \
    --split val --budgets 1.5,3.0,4.5 --selector-device cpu \
    --methods oracle,generic_selector,edit_conditioned_selector,uncertainty
fi

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
  sleep 30
done
export CUDA_VISIBLE_DEVICES="$GPU"
for method in generic_selector edit_conditioned_selector oracle; do
  output="${RUN_ROOT}/writeback/${method}"
  if [[ ! -s "${output}/summary.json" ]]; then
    "$PYTHON" scripts/evaluate_agent_map_writeback.py \
      "$UPDATER" "$EPISODES" "${ROLLOUTS}/${method}.jsonl" "$output" \
      --device cuda --split val --image-size 512 --threshold 0.5
  fi
done

echo "[$(date --iso-8601=seconds)] policy baseline rollout and writeback complete"
