#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 FULL NO_OPERATION NO_TEACHER OUTPUT [POLL_SECONDS]" >&2
  exit 2
fi

FULL="$1"
NO_OPERATION="$2"
NO_TEACHER="$3"
OUTPUT="$4"
POLL_SECONDS="${5:-30}"
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "waiting_for=$FULL poll_seconds=$POLL_SECONDS"
while [[ ! -s "$FULL" ]]; do
  sleep "$POLL_SECONDS"
done
if ! /home/wh/venvs/activemap/bin/python - "$FULL" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("gates", {}).get("passed") is True else 1)
PY
then
  echo "comparison_skipped=full intervention gates failed"
  exit 0
fi

echo "waiting_for=$NO_OPERATION,$NO_TEACHER"
while [[ ! -s "$NO_OPERATION" || ! -s "$NO_TEACHER" ]]; do
  sleep "$POLL_SECONDS"
done
if [[ -e "$OUTPUT" ]]; then
  echo "refusing_comparison=existing output $OUTPUT" >&2
  exit 3
fi

cd "$PROJECT_ROOT"
PYTHONPATH=src /home/wh/venvs/activemap/bin/python \
  scripts/compare_tool_belief_interventions.py \
  --candidate "full=$FULL" \
  --candidate "no_operation=$NO_OPERATION" \
  --candidate "no_teacher=$NO_TEACHER" \
  --reference full --output "$OUTPUT"
echo "comparison_complete=$(date --iso-8601=seconds) output=$OUTPUT"
