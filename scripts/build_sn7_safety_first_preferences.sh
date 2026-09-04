#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PYTHON="$STORE/envs/activemap-agent/bin/python"
FULL="$STORE/processed/sn7_v1/agent/sequential_selector_v1/full"
RUN="${RUN:-$STORE/runs/sn7_active_catalog/onpolicy_rl_reentry_stage_a_eps035_20260731}"
ROLLOUTS="${ROLLOUTS:-4}"
OUTPUT="$RUN/preferences_safety_v1"

cd "$PROJECT"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 4; }
mkdir -p "$OUTPUT"

for split in train val; do
  trace_args=()
  for ((rollout=0; rollout<ROLLOUTS; rollout++)); do
    trace_args+=(--traces "$RUN/${split}_rollout${rollout}/evaluation/traces.jsonl")
  done
  "$PYTHON" scripts/build_active_catalog_onpolicy_preferences.py \
    "$FULL/active_catalog_sft_v4/${split}.jsonl" \
    "$FULL/active_catalog_sft_v4/${split}_evaluation_index.jsonl" \
    "$OUTPUT/${split}.jsonl" "${trace_args[@]}" \
    --expected-split "$split" --minimum-margin 0.001 \
    --preference-mode safety_first --false-edit-tolerance 0.02
done
