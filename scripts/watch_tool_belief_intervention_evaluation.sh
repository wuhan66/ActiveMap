#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 DATA_ROOT RUN_DIR OUTPUT_DIR [POLL_SECONDS]" >&2
  exit 2
fi

DATA_ROOT="$1"
RUN_DIR="$2"
OUTPUT_DIR="$3"
POLL_SECONDS="${4:-30}"
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_SUMMARY="$RUN_DIR/summary.json"
CHECKPOINT="$RUN_DIR/best.pt"
VAL="$DATA_ROOT/val.jsonl"

echo "waiting_for=$TRAINING_SUMMARY poll_seconds=$POLL_SECONDS"
while [[ ! -s "$TRAINING_SUMMARY" || ! -s "$CHECKPOINT" ]]; do
  sleep "$POLL_SECONDS"
done
if [[ -e "$OUTPUT_DIR/summary.json" ]]; then
  echo "refusing_evaluation=existing summary at $OUTPUT_DIR/summary.json" >&2
  exit 3
fi

echo "evaluation_start=$(date --iso-8601=seconds) checkpoint=$CHECKPOINT"
cd "$PROJECT_ROOT"
PYTHONPATH=src /home/wh/venvs/activemap/bin/python \
  scripts/evaluate_tool_belief_interventions.py \
  "$VAL" "$CHECKPOINT" "$OUTPUT_DIR" --device cpu
echo "evaluation_complete=$(date --iso-8601=seconds) output=$OUTPUT_DIR"
