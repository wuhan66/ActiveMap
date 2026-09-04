#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
NATIVE="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes"
FEATURES="${STORE}/runs/argotweak/map_conditioned_selector_screen_v1/features/qwen_structured"
ROOT="${STORE}/runs/argotweak/safety_listwise_selector_screen_v1"
mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/.lock"; flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

run_setting() {
  local gpu="$1" tag="$2" weight="$3"
  for split in train val; do
    "${PYTHON}" scripts/relabel_argotweak_map_utility.py \
      --features "${FEATURES}/${split}" --episodes "${NATIVE}/${split}.jsonl" \
      --commit-confidence 0.50 --false-discovery-weight "${weight}" \
      --output-dir "${ROOT}/features/${tag}/${split}" \
      >"${ROOT}/logs/relabel_${tag}_${split}.log" 2>&1
  done
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_argotweak_listwise_selector.py \
    --features "${ROOT}/features/${tag}/train" --output-dir "${ROOT}/models/${tag}" \
    --seed 20261531 --device cuda >"${ROOT}/logs/train_${tag}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${ROOT}/features/${tag}/val" --checkpoint "${ROOT}/models/${tag}/best.pt" \
    --output-dir "${ROOT}/validation/${tag}" --device cuda >"${ROOT}/logs/val_${tag}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_executable_policies.py \
    --episodes "${NATIVE}/val.jsonl" --rankings "${ROOT}/validation/${tag}/rankings.jsonl" \
    --budgets 1,3,5,10 --commit-confidence 0.50 --output-dir "${ROOT}/executable/${tag}" \
    >"${ROOT}/logs/executable_${tag}.log" 2>&1
  touch "${ROOT}/status/${tag}.done"
}

run_setting 0 fd025 0.25 & p0=$!
run_setting 1 fd050 0.50 & p1=$!
run_setting 2 fd100 1.00 & p2=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
wait "${p2}" || status=1
if [[ "${status}" -ne 0 ]]; then touch "${ROOT}/QUEUE_FAILED"; exit 1; fi

"${PYTHON}" - <<PY
import json
from pathlib import Path
root = Path("${ROOT}")
rows = []
for tag in ("fd025", "fd050", "fd100"):
    result = json.loads((root / "executable" / tag / "summary.json").read_text())
    top5 = next(row for row in result if row["policy"] == "learned_top5")
    rows.append({"setting": tag, **top5})
eligible = [row for row in rows if row["atomic_edit_f1"] > 0.09091 and row["atomic_edit_precision"] >= 0.70]
payload = {"rows": rows, "promoted": bool(eligible), "winner": max(eligible, key=lambda row: row["atomic_edit_f1"]) if eligible else None}
(root / "promotion.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
touch "${ROOT}/QUEUE_COMPLETED"
