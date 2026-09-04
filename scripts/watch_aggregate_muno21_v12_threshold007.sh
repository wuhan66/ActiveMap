#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_FAMILY="${MUNO21_V12_RUN_FAMILY:-agent_v12_proactive_tool_four_seed_threshold007_v1}"
ROLLOUT_ROOT="${STORAGE_ROOT}/artifacts/paper_rollouts/${RUN_FAMILY}"
OUTPUT="${MUNO21_V12_AGGREGATE_OUTPUT:-${STORAGE_ROOT}/artifacts/paper_evidence/muno21_v12_threshold007_controller_4seed_v1}"
SEEDS=(20260822 20260823 20260824 20260825)
SELECTOR_SEEDS=(20260812 20260813 20260811 20260812)

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
exec >>"${OUTPUT}/watcher.log" 2>&1
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0

echo "[$(date -Is)] waiting for threshold-0.07 rollouts"
for seed in "${SEEDS[@]}"; do
  summary="${ROLLOUT_ROOT}/seed${seed}/summary.json"
  log="${STORAGE_ROOT}/logs/${RUN_FAMILY}/seed${seed}.log"
  while [[ ! -s "${summary}" ]]; do
    if [[ -s "${log}" ]] && grep -Eqi 'traceback|out of memory|error:' "${log}"; then
      printf '{"status":"failed","seed":%s,"log":"%s","test_assets_read":false}\n' \
        "${seed}" "${log}" >"${OUTPUT}/FAILED.json"
      exit 3
    fi
    sleep 60
  done
done

"${PYTHON}" - "${ROLLOUT_ROOT}/seed_pairing.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema_version": "activemap-agent-selector-seed-pairing-v1",
    "split": "val",
    "test_assets_read": False,
    "pairs": [
        {"agent_seed": 20260822, "selector_seed": 20260812},
        {"agent_seed": 20260823, "selector_seed": 20260813},
        {"agent_seed": 20260824, "selector_seed": 20260811},
        {"agent_seed": 20260825, "selector_seed": 20260812},
    ],
}
if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
    raise SystemExit(f"refusing changed seed pairing: {path}")
path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" scripts/aggregate_agent_three_seeds.py \
  "${ROLLOUT_ROOT}" "${OUTPUT}/controller_four_seed.json" \
  --seeds "20260822,20260823,20260824,20260825" \
  --candidate qwen3_4b_sft_calibrated_tool_to_belief \
  --baselines edit_conditioned_proactive_tools \
  --bootstrap 10000 --rng-seed 20260912

OUTPUT="${OUTPUT}" "${PYTHON}" - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
source = root / "controller_four_seed.json"
data = json.loads(source.read_text(encoding="utf-8"))
comparison = data["comparisons"]["edit_conditioned_proactive_tools"]
metrics = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_cost",
    "mean_tool_calls",
    "mean_tool_belief_l1_delta",
    "terminal_edit_flip_rate",
    "quality_cost_utility_auc",
    "episode_utility_v2_proxy_balanced_auc",
    "episode_utility_v2_proxy_safety_auc",
)
rows = []
for metric in metrics:
    if metric not in comparison["paired_delta"]:
        continue
    interval = comparison["paired_delta"][metric]
    baseline = sum(
        float(row["baseline"][metric]) for row in comparison["per_seed"]
    ) / len(comparison["per_seed"])
    candidate = sum(
        float(row["candidate"][metric]) for row in comparison["per_seed"]
    ) / len(comparison["per_seed"])
    rows.append(
        {
            "metric": metric,
            "edit_conditioned": baseline,
            "qwen4b_tool_to_belief": candidate,
            "delta": interval["delta"],
            "ci95_low": interval["ci95_low"],
            "ci95_high": interval["ci95_high"],
        }
    )

with (root / "table.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# MUNO21 threshold=0.07 four-seed controller aggregate",
    "",
    "| Metric | Edit-conditioned | Qwen-4B Tool-to-Belief | Delta | 95% CI |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['metric']} | {row['edit_conditioned']:.6f} | "
        f"{row['qwen4b_tool_to_belief']:.6f} | {row['delta']:+.6f} | "
        f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
    )
(root / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

files = {}
for name in ("controller_four_seed.json", "table.csv", "table.md"):
    path = root / name
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-v12-threshold007-controller-4seed-v1",
            "status": "complete",
            "model_seeds": [20260822, 20260823, 20260824, 20260825],
            "bootstrap_repetitions": 10000,
            "files": files,
            "split": "val",
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "[$(date -Is)] threshold-0.07 four-seed aggregation complete"
