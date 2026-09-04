#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
NATIVE="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes/val.jsonl"
FEATURES="${STORE}/runs/argotweak/safety_listwise_selector_screen_v1/features/fd050"
ROOT="${STORE}/runs/argotweak/safety_listwise_selector_replication_v1"
mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/.lock"; flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

run_seed() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_argotweak_listwise_selector.py \
    --features "${FEATURES}/train" --output-dir "${ROOT}/models/seed${seed}" \
    --seed "${seed}" --device cuda >"${ROOT}/logs/train_seed${seed}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${FEATURES}/val" --checkpoint "${ROOT}/models/seed${seed}/best.pt" \
    --output-dir "${ROOT}/validation/seed${seed}" --device cuda \
    >"${ROOT}/logs/val_seed${seed}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_executable_policies.py \
    --episodes "${NATIVE}" --rankings "${ROOT}/validation/seed${seed}/rankings.jsonl" \
    --budgets 1,3,5,10 --commit-confidence 0.50 \
    --output-dir "${ROOT}/executable/seed${seed}" \
    >"${ROOT}/logs/executable_seed${seed}.log" 2>&1
  touch "${ROOT}/status/seed${seed}.done"
}

run_seed 0 20261532 & p0=$!
run_seed 1 20261533 & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
if [[ "${status}" -ne 0 ]]; then touch "${ROOT}/QUEUE_FAILED"; exit 1; fi

"${PYTHON}" - <<PY
import json, statistics
from pathlib import Path
source = Path("${STORE}/runs/argotweak/safety_listwise_selector_screen_v1/executable/fd050/summary.json")
paths = [source, *(Path("${ROOT}/executable") / f"seed{seed}" / "summary.json" for seed in (20261532, 20261533))]
seeds = (20261531, 20261532, 20261533)
rows = []
for seed, path in zip(seeds, paths):
    result = json.loads(path.read_text())
    for budget in (1, 3, 5, 10):
        row = next(item for item in result if item["policy"] == f"learned_top{budget}")
        rows.append({"seed": seed, "budget": budget, **row})
aggregate = []
for budget in (1, 3, 5, 10):
    subset = [row for row in rows if row["budget"] == budget]
    entry = {"budget": budget, "seeds": len(subset)}
    for key in ("atomic_edit_f1", "atomic_edit_precision", "atomic_edit_recall", "false_edit_rate", "false_edit_discovery_rate", "missed_edit_rate", "balanced_utility"):
        values = [float(row[key]) for row in subset]
        entry[f"{key}_mean"] = statistics.mean(values)
        entry[f"{key}_std"] = statistics.stdev(values)
    aggregate.append(entry)
(Path("${ROOT}") / "seed_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(Path("${ROOT}") / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
PY
touch "${ROOT}/QUEUE_COMPLETED"
