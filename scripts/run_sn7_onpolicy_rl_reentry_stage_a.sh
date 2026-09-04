#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PYTHON="$STORE/envs/activemap-agent/bin/python"
MODEL=/home/wh/hf_models/Qwen3-VL-4B-Instruct
DATA="$STORE/processed/sn7_v1/agent/sequential_selector_v1"
FULL="$DATA/full"
RUN="${RUN:-$STORE/runs/sn7_active_catalog/onpolicy_rl_reentry_stage_a_eps035_20260731}"
ADAPTER="$STORE/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
TOOL_BELIEF="$STORE/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt"
TOOL_GATE="$STORE/runs/sn7_active_catalog/step0_tool_need_gate_seed20260730_benefit_gate_v1"
GPU="${GPU:-5}"
ROLLOUTS="${ROLLOUTS:-4}"
TRAIN_STATES="${TRAIN_STATES:-128}"
VAL_STATES="${VAL_STATES:-64}"
SELECTOR_EPSILON="${SELECTOR_EPSILON:-0.35}"

cd "$PROJECT"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for path in \
  "$MODEL/config.json" "$ADAPTER/adapter_config.json" \
  "$DATA/selector_states_train_val.jsonl" \
  "$FULL/episodes_train_val.jsonl" \
  "$FULL/active_catalog_sft_v4/train.jsonl" \
  "$FULL/active_catalog_sft_v4/val.jsonl" \
  "$FULL/active_catalog_sft_v4/train_evaluation_index.jsonl" \
  "$FULL/active_catalog_sft_v4/val_evaluation_index.jsonl" \
  "$TOOL_BELIEF" "$TOOL_GATE/gate.joblib" "$TOOL_GATE/summary.json"; do
  test -f "$path" || { echo "missing Stage-A input: $path" >&2; exit 3; }
done

test ! -e "$RUN" || { echo "refusing existing Stage-A run: $RUN" >&2; exit 4; }
test -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)" ||
  { echo "physical GPU $GPU is occupied" >&2; exit 5; }
mkdir -p "$RUN/states" "$RUN/preferences_v2"

"$PYTHON" scripts/sample_active_catalog_online_states.py \
  "$DATA/selector_states_train_val.jsonl" "$RUN/states/train.jsonl" \
  --split train --count "$TRAIN_STATES" --seed 20260731
"$PYTHON" scripts/sample_active_catalog_online_states.py \
  "$DATA/selector_states_train_val.jsonl" "$RUN/states/val.jsonl" \
  --split val --count "$VAL_STATES" --seed 20260731

run_rollout() {
  local split="$1" rollout="$2" seed="$3"
  local output="$RUN/${split}_rollout${rollout}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/evaluate_active_catalog_closed_loop.py \
    "$MODEL" "$ADAPTER" "$RUN/states/${split}.jsonl" \
    "$FULL/episodes_train_val.jsonl" \
    "$FULL/active_catalog_sft_v4/${split}.jsonl" \
    "$FULL/active_catalog_sft_v4/${split}_evaluation_index.jsonl" \
    "$output/evaluation" --device cuda:0 --split "$split" \
    --seed "$seed" --max-candidates 16 --max-acquisitions 2 \
    --max-new-tokens 64 --bootstrap-repetitions 0 \
    --tool-mode selective --belief-mode recurrent \
    --tool-belief-checkpoint "$TOOL_BELIEF" \
    --tool-gate "$TOOL_GATE/gate.joblib" \
    --tool-gate-summary "$TOOL_GATE/summary.json" \
    --tool-artifact-root "$output/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=$STORE" \
    --do-sample --temperature 0.9 --top-p 0.95 \
    --selector-epsilon "$SELECTOR_EPSILON"
}

for split in train val; do
  for ((rollout=0; rollout<ROLLOUTS; rollout++)); do
    run_rollout "$split" "$rollout" "$((20260731 + rollout))"
  done
done

for split in train val; do
  trace_args=()
  for ((rollout=0; rollout<ROLLOUTS; rollout++)); do
    trace_args+=(--traces "$RUN/${split}_rollout${rollout}/evaluation/traces.jsonl")
  done
  "$PYTHON" scripts/build_active_catalog_onpolicy_preferences.py \
    "$FULL/active_catalog_sft_v4/${split}.jsonl" \
    "$FULL/active_catalog_sft_v4/${split}_evaluation_index.jsonl" \
    "$RUN/preferences_v2/${split}.jsonl" "${trace_args[@]}" \
    --expected-split "$split" --minimum-margin 0.001
done

"$PYTHON" - "$RUN" "$TRAIN_STATES" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source_states = int(sys.argv[2])
summaries = {}
for split in ("train", "val"):
    path = root / "preferences_v2" / f"{split}.jsonl.summary.json"
    summaries[split] = json.loads(path.read_text())
train = summaries["train"]
multi_action_rate = train["preferences"] / source_states
passed = (
    train["preferences"] > 0
    and multi_action_rate >= 0.20
    and len(train["family_counts"]) >= 2
)
payload = {
    "schema_version": "sn7-onpolicy-rl-reentry-stage-a-v2",
    "status": "pass" if passed else "fail",
    "promotion_authorized": bool(passed),
    "minimum_multi_action_rate": 0.20,
    "observed_preference_rate_proxy": multi_action_rate,
    "train": train,
    "val": summaries["val"],
    "test_assets_read": False,
}
(root / "STAGE_A_DECISION.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if not passed:
    raise SystemExit(12)
PY
