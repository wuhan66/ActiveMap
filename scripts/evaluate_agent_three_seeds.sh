#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
export ACTIVEMAP_LAUNCHER_NAME="evaluate_agent_three_seeds.sh"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
GPU="${MUNO21_AGENT_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
MODEL="${MUNO21_AGENT_MODEL:-${ACTIVEMAP_MODEL_ROOT}/Qwen3-4B}"
AGENT_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent"
SPLIT="${MUNO21_AGENT_SPLIT:-val}"
TOOL_DATA="${MUNO21_TOOL_DATA:-${AGENT_ROOT}/agent_data_v9_natural_sparse_tools}"
TOOL_BELIEF="${MUNO21_TOOL_BELIEF:-${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt}"
SELECTOR_ROOT="${MUNO21_SELECTOR_ROOT:-${STORAGE_ROOT}/runs/selector}"
SEED_LIST="${MUNO21_AGENT_SEEDS:-20260821 20260822 20260823}"
SELECTOR_SEED_LIST="${MUNO21_SELECTOR_SEEDS:-20260811 20260812 20260813}"
MAX_TOOL_CALLS="${MUNO21_MAX_TOOL_CALLS:-2}"
RUN_FAMILY="${MUNO21_AGENT_RUN_FAMILY:-muno21_qwen3_4b_sparse_tool_sft}"
INCLUDE_HEURISTICS_FIRST_SEED="${MUNO21_INCLUDE_HEURISTICS_FIRST_SEED:-1}"
ASSET_ROOT_MAP="${MUNO21_ASSET_ROOT_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"

[[ "${MAX_TOOL_CALLS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MUNO21_MAX_TOOL_CALLS must be a positive integer" >&2
  exit 2
}
[[ "${INCLUDE_HEURISTICS_FIRST_SEED}" =~ ^[01]$ ]] || {
  echo "MUNO21_INCLUDE_HEURISTICS_FIRST_SEED must be 0 or 1" >&2
  exit 2
}

if [[ "$SPLIT" == "test" ]]; then
  PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -c "from scripts.frozen_test_access import assert_frozen_test_access; assert_frozen_test_access()"
  STATES="${MUNO21_SELECTOR_STATES:-${AGENT_ROOT}/selector_states_test_v1.jsonl}"
  EPISODES="${MUNO21_EPISODES:-${AGENT_ROOT}/episodes_test_v1.jsonl}"
  TOOL_SUPERVISION="${MUNO21_TOOL_SUPERVISION:-${AGENT_ROOT}/sparse_tool_labels_test_v1/sft.jsonl}"
  OUTPUT_ROOT="${MUNO21_AGENT_ROLLOUT_ROOT:-${STORAGE_ROOT}/artifacts/frozen_test/agent_three_seed}"
elif [[ "$SPLIT" == "val" ]]; then
  STATES="${MUNO21_SELECTOR_STATES:-${AGENT_ROOT}/selector_states_v1.jsonl}"
  EPISODES="${MUNO21_EPISODES:-${AGENT_ROOT}/episodes_train_val_v1.jsonl}"
  TOOL_SUPERVISION="${MUNO21_TOOL_SUPERVISION:-${TOOL_DATA}/val/sft_composed.jsonl}"
  OUTPUT_ROOT="${MUNO21_AGENT_ROLLOUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_rollouts/agent_three_seed_val}"
else
  echo "MUNO21_AGENT_SPLIT must be val or test" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/assert_allowed_gpu.sh"
activemap_assert_allowed_gpu "${GPU}"
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "$STORAGE_ROOT"
for path in "$STATES" "$EPISODES" "$TOOL_SUPERVISION" "$TOOL_BELIEF"; do
  [[ -s "$path" ]] || { echo "Missing Agent rollout input: $path" >&2; exit 3; }
done

read -r -a seeds <<<"$SEED_LIST"
read -r -a selector_seeds <<<"$SELECTOR_SEED_LIST"
[[ "${#seeds[@]}" -eq "${#selector_seeds[@]}" ]] || {
  echo "Agent and selector seed lists must have equal length" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT"
"$PYTHON" - "$OUTPUT_ROOT/seed_pairing.json" "$SEED_LIST" "$SELECTOR_SEED_LIST" "$SPLIT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
agent_seeds = [int(value) for value in sys.argv[2].split()]
selector_seeds = [int(value) for value in sys.argv[3].split()]
split = sys.argv[4]
record = {
    "schema_version": "activemap-agent-selector-seed-pairing-v1",
    "split": split,
    "test_assets_read": False,
    "pairs": [
        {"agent_seed": agent, "selector_seed": selector}
        for agent, selector in zip(agent_seeds, selector_seeds, strict=True)
    ],
}
if output.exists():
    if json.loads(output.read_text(encoding="utf-8")) != record:
        raise SystemExit(f"refusing changed seed pairing: {output}")
else:
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY
for seed_index in "${!seeds[@]}"; do
  seed="${seeds[$seed_index]}"
  selector_seed="${selector_seeds[$seed_index]}"
  run="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}"
  promotion="${run}/evaluation/selection/promoted_adapter.json"
  edit_checkpoint="${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed${selector_seed}/best.pt"
  generic_checkpoint="${SELECTOR_ROOT}/muno21_evidence_generic_v5_seed${selector_seed}/best.pt"
  output="${OUTPUT_ROOT}/seed${seed}"
  for path in "$promotion" "$edit_checkpoint" "$generic_checkpoint"; do
    [[ -s "$path" ]] || { echo "Missing seed ${seed} rollout input: $path" >&2; exit 3; }
  done
  PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" scripts/verify_sparse_tool_sft_adapter.py "$promotion"
  adapter="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_path"])' "$promotion")"
  if [[ -s "${output}/summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] seed ${seed} rollouts already complete; skipping"
    continue
  fi
  mkdir -p "$(dirname "$output")"
  methods="oracle,generic_selector,edit_conditioned_selector,qwen3_4b_sft,forced_tools,qwen3_4b_sft_tools_no_belief,qwen3_4b_sft_tool_to_belief"
  if [[ "$seed_index" -eq 0 && "${INCLUDE_HEURISTICS_FIRST_SEED}" == "1" ]]; then
    methods="${methods},random,cheapest,quality_first,uncertainty,mapex,greedy_utility"
  fi
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" \
    --adapter "$adapter" \
    --checkpoint "$edit_checkpoint" \
    --generic-checkpoint "$generic_checkpoint" \
    --episodes "$EPISODES" \
    --tool-belief-checkpoint "$TOOL_BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" \
    --tool-supervision-jsonl "$TOOL_SUPERVISION" \
    --asset-root-map "$ASSET_ROOT_MAP" \
    --max-tool-calls "$MAX_TOOL_CALLS" \
    --split "$SPLIT" --budgets 1.5,3.0,4.5 --device cuda --selector-device cpu \
    --seed "$seed" \
    --methods "$methods" \
    >"${OUTPUT_ROOT}/seed${seed}.log" 2>&1
done

echo "[$(date --iso-8601=seconds)] three-seed ${SPLIT} rollouts complete"
