#!/usr/bin/env bash
set -euo pipefail

ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${ACTIVEMAP_REPOSITORY:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-$ROOT/envs/activemap-agent/bin/python}"
OUTPUT="${1:-$ROOT/artifacts/sn7_active_catalog/gate_ranker_round_20260720}"
POLL_SECONDS="${POLL_SECONDS:-180}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-64800}"

declare -a RUNS=(
  "t05_seed1,0.05,20260720,$ROOT/runs/sn7_active_catalog/qwen3vl4b_gate_only_t05_seed1/seed20260720/active_catalog_gate_ranker_val/traces.jsonl"
  "t10_seed1,0.10,20260720,$ROOT/runs/sn7_active_catalog/qwen3vl4b_gate_only_t10_seed1/seed20260720/active_catalog_gate_ranker_val/traces.jsonl"
  "t10_seed2,0.10,20260721,$ROOT/runs/sn7_active_catalog/qwen3vl4b_gate_only_t10_seed2/seed20260721/active_catalog_gate_ranker_val/traces.jsonl"
  "t10_seed3,0.10,20260722,$ROOT/runs/sn7_active_catalog/qwen3vl4b_gate_only_t10_seed3/seed20260722/active_catalog_gate_ranker_val/traces.jsonl"
  "t15_seed1,0.15,20260720,$ROOT/runs/sn7_active_catalog/qwen3vl4b_gate_only_t15_seed1/seed20260720/active_catalog_gate_ranker_val/traces.jsonl"
  "t20_seed1,0.20,20260720,$ROOT/runs/sn7_active_catalog/qwen3vl4b_gate_only_t20_seed1/seed20260720/active_catalog_gate_ranker_val/traces.jsonl"
)

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "refusing existing output: $OUTPUT" >&2; exit 3; }

started="$(date +%s)"
while true; do
  missing=0
  for declaration in "${RUNS[@]}"; do
    trace="${declaration#*,*,*,}"
    summary="${trace%/traces.jsonl}/summary.json"
    [[ -s "$trace" && -s "$summary" ]] || missing=$((missing + 1))
  done
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  elapsed=$(( $(date +%s) - started ))
  if [[ "$elapsed" -ge "$TIMEOUT_SECONDS" ]]; then
    echo "timed out after ${elapsed}s with $missing incomplete runs" >&2
    exit 4
  fi
  echo "$(date --iso-8601=seconds) waiting for $missing/6 gate-ranker runs"
  sleep "$POLL_SECONDS"
done

args=()
for declaration in "${RUNS[@]}"; do
  args+=(--run "$declaration")
done
cd "$REPO"
export PYTHONPATH="src:."
exec "$PYTHON" scripts/finalize_active_catalog_gate_ranker_round.py \
  "$OUTPUT" "${args[@]}" --tuning-seed 20260720 \
  --repetitions 2000 --bootstrap-seed 20260723
