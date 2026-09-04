#!/usr/bin/env bash
set -euo pipefail

# Aggregate three predeclared SELECT calibration variants after their generated
# action evaluations are complete. It records a choice; it never launches a
# terminal controller or accesses test data.
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_calibration_v1.json}"
LOG="${LOG:-${STORE}/logs/muno21_qwen3vl4b_sequential_select_calibration_v1.watch.log}"
POLL_SECONDS="${POLL_SECONDS:-90}"

runs=(
  "muno21_qwen3vl4b_sequential_select_v1_seed20260862_eval"
  "muno21_qwen3vl4b_sequential_select_cal15_v1_seed20260862_eval"
  "muno21_qwen3vl4b_sequential_select_natural_v1_seed20260862_eval"
)
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing aggregate: ${OUTPUT}" >&2; exit 2; }
mkdir -p "$(dirname "${LOG}")"
for run in "${runs[@]}"; do
  until [[ -s "${STORE}/runs/agent/${run}/summary.json" ]]; do
    echo "Waiting for ${run}"
    sleep "${POLL_SECONDS}"
  done
done

"${PY}" - "${OUTPUT}" "${STORE}" "${runs[@]}" >"${LOG}" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

output, store, *runs = sys.argv[1:]
records = []
required = (
    "valid_action_rate_1",
    "nonzero_calls",
    "call_rate_at_most_0_50",
    "acquire_recall_at_least_0_10",
    "precision_at_least_prevalence",
    "utility_positive",
    "macro_f1_above_always_stop",
    "utility_bootstrap_ci_above_zero",
)
for run in runs:
    summary = json.loads((Path(store) / "runs" / "agent" / run / "summary.json").read_text())
    gate = summary["promotion_gate"]
    records.append({
        "run": run,
        "passed": bool(gate.get("passed")),
        "passed_gate_count": sum(bool(gate.get(name)) for name in required),
        "promotion_gate": gate,
        "metrics": summary["metrics"],
    })
records.sort(key=lambda item: (item["passed"], item["passed_gate_count"], item["metrics"]["realized_utility_mean"], item["metrics"]["macro_f1"]), reverse=True)
result = {
    "schema_version": "activemap-sequential-select-calibration-v1",
    "variants": records,
    "recommended_run": records[0]["run"],
    "recommended_run_passed": records[0]["passed"],
    "test_assets_read": False,
    "next_action": "train joint curriculum only if the recommended run passes all predeclared gates",
}
Path(output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "Sequential SELECT calibration aggregate completed: ${OUTPUT}"
