#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 DATA_ROOT [POLL_SECONDS]" >&2
  exit 2
fi

DATA_ROOT="$1"
POLL_SECONDS="${2:-30}"
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY="$DATA_ROOT/summary.json"
TRAIN="$DATA_ROOT/train.jsonl"
VAL="$DATA_ROOT/val.jsonl"
AUDIT="$DATA_ROOT/audit.json"

echo "waiting_for=$SUMMARY poll_seconds=$POLL_SECONDS"
while [[ ! -s "$SUMMARY" ]]; do
  sleep "$POLL_SECONDS"
done

echo "summary_detected=$SUMMARY"
cd "$PROJECT_ROOT"
PYTHONPATH=src /home/wh/venvs/activemap/bin/python \
  scripts/audit_tool_belief_data.py "$TRAIN" "$VAL" --output "$AUDIT"
echo "audit_passed=$AUDIT"
