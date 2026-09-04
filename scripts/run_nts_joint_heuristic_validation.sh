#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/server_nts_env.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
LEDGER="${ACTIVEMAP_UNIFIED_PROGRESS:-${ACTIVEMAP_STORAGE_ROOT}/artifacts/cluster_progress/unified.json}"
STATES="${MUNO21_SELECTOR_STATES:-${ACTIVEMAP_PROCESSED_ROOT}/muno21_v2/agent/selector_states_v1.jsonl}"
CHECKPOINT_LIST="${MUNO21_EDIT_SELECTOR_CHECKPOINTS:?Set MUNO21_EDIT_SELECTOR_CHECKPOINTS}"
OUTPUT="${MUNO21_HEURISTIC_OUTPUT:-${ACTIVEMAP_RUN_ROOT}/muno21_heuristic_budget_filling_v1}"
LOG="${ACTIVEMAP_LOG_ROOT}/muno21_heuristic_budget_filling_v1.log"

[[ "${ACTIVEMAP_DISABLE_FROZEN_TEST:-}" == "1" ]] || {
  echo "Joint-debug launcher requires frozen-test disablement" >&2
  exit 2
}
for path in "$LEDGER" "$STATES"; do
  [[ -s "$path" ]] || { echo "Missing joint-debug input: $path" >&2; exit 3; }
done
read -r -a checkpoints <<<"$CHECKPOINT_LIST"
[[ "${#checkpoints[@]}" -ge 1 ]] || { echo "No selector checkpoints" >&2; exit 3; }
checkpoint_args=()
for checkpoint in "${checkpoints[@]}"; do
  [[ -s "$checkpoint" ]] || { echo "Missing selector checkpoint: $checkpoint" >&2; exit 3; }
  checkpoint_args+=(--checkpoint "$checkpoint")
done
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 4; }

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
PYTHONPATH=src:. "$PYTHON" scripts/assert_joint_progress.py \
  "$LEDGER" "$PROJECT_ROOT" --cluster-id nts --expected-role validation_debug \
  >"${LOG}.authorization.json"

PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
  "$ACTIVEMAP_MODEL_ROOT/Qwen3-4B" "$STATES" "$OUTPUT" \
  "${checkpoint_args[@]}" \
  --split val --budgets 1.5,3.0,4.5 --device cpu --selector-device cpu \
  --seed 20260821 \
  --methods random,cheapest,quality_first,uncertainty,mapex,greedy_utility \
  >"$LOG" 2>&1

PYTHONPATH=src:. "$PYTHON" - "$LEDGER" "$OUTPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

ledger, output = map(Path, sys.argv[1:])
summary = output / "summary.json"
payload = {
    "schema_version": "activemap-joint-validation-run-v1",
    "cluster_id": "nts",
    "role": "validation_debug",
    "split": "val",
    "unified_progress": str(ledger.resolve()),
    "unified_progress_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "summary": str(summary.resolve()),
    "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    "methods": [
        "random", "cheapest", "quality_first", "uncertainty", "mapex", "greedy_utility"
    ],
    "test_assets_read": False,
}
(output / "joint_validation_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

echo "[$(date --iso-8601=seconds)] nts joint heuristic validation complete"
