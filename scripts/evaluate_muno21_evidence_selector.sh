#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
STATES="${MUNO21_SELECTOR_STATES:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/selector_states_v1.jsonl}"
RUN_DIR="${1:?usage: evaluate_muno21_evidence_selector.sh RUN_DIR}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m activemap.cli evaluate-selector \
  "$STATES" "$RUN_DIR/validation/evaluation.csv" \
  --split val --budgets 1.5,3.0,4.5 --checkpoint "$RUN_DIR/best.pt" --device cpu
"$PYTHON" -m activemap.cli rollout-selector \
  "$STATES" "$RUN_DIR/validation/rollout" \
  --split val --budgets 1.5,3.0,4.5 --checkpoint "$RUN_DIR/best.pt" \
  --device cpu --max-steps 3
echo "Validation-only evidence-selector evaluation complete: $RUN_DIR/validation"
