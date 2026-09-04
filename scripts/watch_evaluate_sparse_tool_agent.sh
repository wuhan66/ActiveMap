#!/usr/bin/env bash
set -euo pipefail

if [[ "${ACTIVEMAP_ALLOW_LEGACY_PIPELINE:-0}" != "1" ]]; then
  echo "Legacy single-seed watcher is disabled on hdpi-sys1." >&2
  echo "Use run_sparse_tool_sft_three_seeds.sh then evaluate_agent_three_seeds.sh." >&2
  exit 64
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
AGENT_SITE_PACKAGES="${ACTIVEMAP_AGENT_SITE_PACKAGES:-/mnt/mydisk/wh/ActiveMap/envs/agent_peft_overlay}"
GPU="${MUNO21_AGENT_GPU:-3}"
RUN_DIR="${MUNO21_AGENT_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_natural_sparse_tool_sft_seed20260821}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-4B}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent}"
TOOL_DATA="${MUNO21_TOOL_DATA:-${DATA_ROOT}/agent_data_v9_natural_sparse_tools}"
REACHABILITY_AUDIT="${MUNO21_AGENT_REACHABILITY_AUDIT:-${TOOL_DATA}/val/reachability_audit.json}"
OPPORTUNITY_MANIFEST="${MUNO21_TOOL_OPPORTUNITY_MANIFEST:-/mnt/mydisk/wh/ActiveMap/runs/agent/tool_belief_anchored_v4_seed20260821_sequence_eval_detailed/tool_opportunity_strata_val.json}"
EPISODES="${DATA_ROOT}/episodes_train_val_v1.jsonl"
STATES="${DATA_ROOT}/selector_states_v1.jsonl"
TOOL_BELIEF="${MUNO21_TOOL_BELIEF:-/mnt/mydisk/wh/ActiveMap/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt}"
UPDATER="${MUNO21_UPDATER:-/mnt/mydisk/wh/ActiveMap/runs/updater/muno21_road_v4_explicit_change_scratch_seed20260726/best_val_loss.pt}"
EVALUATION="${RUN_DIR}/evaluation"
STATUS="${EVALUATION}/watcher.exit_code"

edit_checkpoints=(
  /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260811/best.pt
  /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260812/best.pt
  /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260813/best.pt
)
generic_checkpoints=(
  /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_generic_v5_seed20260811/best.pt
  /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_generic_v5_seed20260812/best.pt
  /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_generic_v5_seed20260813/best.pt
)

mkdir -p "${EVALUATION}/selection"
rm -f "$STATUS"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS"' EXIT

train_pid="$(cat "${RUN_DIR}/train.pid")"
while kill -0 "$train_pid" 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] waiting for sparse-tool SFT PID ${train_pid}"
  sleep 60
done
[[ -s "${RUN_DIR}/final/adapter_config.json" ]] || {
  echo "Sparse-tool SFT ended without a final adapter" >&2
  exit 2
}
[[ -s "${RUN_DIR}/eval_metrics.json" ]] || {
  echo "Sparse-tool SFT ended without final validation metrics" >&2
  exit 2
}
[[ -s "$REACHABILITY_AUDIT" ]] || {
  echo "Missing integrated-trajectory reachability audit: $REACHABILITY_AUDIT" >&2
  exit 3
}

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
  sleep 30
done

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PROJECT_ROOT}/src:${AGENT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

mapfile -t adapters < <(find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 \
  -type d -name 'checkpoint-*' | sort -V)
if [[ "${#adapters[@]}" -eq 0 ]]; then
  adapters=("${RUN_DIR}/final")
fi
labels=()
for adapter in "${adapters[@]}"; do
  label="$(basename "$adapter")"
  labels+=("$label")
  output="${EVALUATION}/${label}/actions"
  if [[ ! -s "${output}/summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] static evaluation: ${label}"
    "$PYTHON" scripts/evaluate_agent_actions.py \
      "$MODEL" "${TOOL_DATA}/val/sft_composed.jsonl" "$output" \
      --adapter "$adapter" --device cuda --batch-size 2 \
      --max-length 2048 --max-new-tokens 96
  fi
  if [[ -s "$OPPORTUNITY_MANIFEST" && ! -s "${output}/tool_opportunity_strata.json" ]]; then
    "$PYTHON" scripts/evaluate_tool_opportunity_strata.py \
      "$OPPORTUNITY_MANIFEST" "${output}/predictions.jsonl" \
      "${output}/tool_opportunity_strata.json"
  fi
done

labels_csv="$(IFS=,; echo "${labels[*]}")"
"$PYTHON" scripts/select_sparse_tool_sft_checkpoint.py \
  "$EVALUATION" "${EVALUATION}/selection/static_checkpoint_decision.json" \
  --labels "$labels_csv"
selected="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["selected_checkpoint"] or d["best_observed_checkpoint"])' \
  "${EVALUATION}/selection/static_checkpoint_decision.json")"
selection_passed="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d["selection_passed"] else "0")' \
  "${EVALUATION}/selection/static_checkpoint_decision.json")"
if [[ "$selection_passed" != "1" ]]; then
  printf '%s\n' "No checkpoint passed the frozen static gates; closed-loop promotion evaluation was not run." \
    > "${EVALUATION}/selection/static_rejection.txt"
  echo "[$(date --iso-8601=seconds)] sparse-tool SFT rejected by static gates"
  exit 0
fi
[[ -n "$selected" && "$selected" != "None" ]] || exit 3
adapter="${RUN_DIR}/checkpoints/${selected}"
[[ -d "$adapter" ]] || adapter="${RUN_DIR}/final"
printf '%s\n' "$adapter" > "${EVALUATION}/selection/evaluated_adapter.txt"

rollouts="${EVALUATION}/${selected}/rollouts"
checkpoint_args=()
for checkpoint in "${edit_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "$checkpoint")
done
generic_args=()
for checkpoint in "${generic_checkpoints[@]}"; do
  generic_args+=(--generic-checkpoint "$checkpoint")
done
if [[ ! -s "${rollouts}/summary.json" ]]; then
  "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$rollouts" \
    "${checkpoint_args[@]}" "${generic_args[@]}" \
    --adapter "$adapter" --episodes "$EPISODES" \
    --tool-belief-checkpoint "$TOOL_BELIEF" \
    --tool-artifact-root "${EVALUATION}/${selected}/tool_artifacts" \
    --tool-supervision-jsonl "${TOOL_DATA}/val/sft_composed.jsonl" \
    --split val --budgets 1.5,3.0,4.5 --device cuda --selector-device cpu \
    --max-tool-calls 6 --tool-out-size 512 \
    --methods generic_selector,edit_conditioned_selector,qwen3_4b_sft,forced_tools,qwen3_4b_sft_tools_no_belief,qwen3_4b_sft_tool_to_belief
fi

"$PYTHON" scripts/assess_agent_rl_readiness.py \
  "${EVALUATION}/selection/static_checkpoint_decision.json" \
  "${rollouts}/summary.json" \
  "${EVALUATION}/selection/rl_readiness_decision.json" \
  --reachability-audit "$REACHABILITY_AUDIT"

comparisons="${EVALUATION}/${selected}/comparisons"
mkdir -p "$comparisons"
full="${rollouts}/qwen3_4b_sft_tool_to_belief.jsonl"
"$PYTHON" scripts/compare_agent_rollouts.py \
  "${rollouts}/edit_conditioned_selector.jsonl" "$full" \
  "${comparisons}/full_vs_selector.json" --bootstrap 2000 --seed 20260821
"$PYTHON" scripts/compare_agent_rollouts.py \
  "${rollouts}/qwen3_4b_sft_tools_no_belief.jsonl" "$full" \
  "${comparisons}/full_vs_no_belief.json" --bootstrap 2000 --seed 20260821
"$PYTHON" scripts/compare_agent_rollouts.py \
  "${rollouts}/forced_tools.jsonl" "$full" \
  "${comparisons}/full_vs_forced.json" --bootstrap 2000 --seed 20260821

for method in generic_selector edit_conditioned_selector qwen3_4b_sft forced_tools \
  qwen3_4b_sft_tools_no_belief qwen3_4b_sft_tool_to_belief; do
  output="${EVALUATION}/${selected}/writeback/${method}"
  [[ -s "${output}/summary.json" ]] && continue
  "$PYTHON" scripts/evaluate_agent_map_writeback.py \
    "$UPDATER" "$EPISODES" "${rollouts}/${method}.jsonl" "$output" \
    --device cuda --split val --image-size 512 --threshold 0.5
done

"$PYTHON" scripts/compare_agent_writebacks.py \
  "${EVALUATION}/${selected}/writeback/edit_conditioned_selector/writeback.jsonl" \
  "${EVALUATION}/${selected}/writeback/qwen3_4b_sft_tool_to_belief/writeback.jsonl" \
  "${comparisons}/full_vs_selector_writeback.json" --bootstrap 2000 --seed 20260821

"$PYTHON" scripts/assess_tool_agent_promotion.py \
  "${rollouts}/summary.json" "${EVALUATION}/selection/single_seed_tool_agent_decision.json" \
  --selector-comparison "${comparisons}/full_vs_selector.json" \
  --no-belief-comparison "${comparisons}/full_vs_no_belief.json" \
  --forced-comparison "${comparisons}/full_vs_forced.json" \
  --writeback-comparison "${comparisons}/full_vs_selector_writeback.json"

echo "[$(date --iso-8601=seconds)] sparse-tool SFT evaluation complete"
