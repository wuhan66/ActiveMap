#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
NATIVE="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes"
FEATURES="${STORE}/runs/argotweak/map_conditioned_selector_screen_v1/features/qwen_structured"
ROOT="${STORE}/runs/argotweak/map_utility_listwise_selector_v1"
mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/.lock"; flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

prepare_gate() {
  local gate="$1" tag="$2"
  for split in train val; do
    "${PYTHON}" scripts/relabel_argotweak_map_utility.py \
      --features "${FEATURES}/${split}" --episodes "${NATIVE}/${split}.jsonl" \
      --commit-confidence "${gate}" --output-dir "${ROOT}/features/${tag}/${split}" \
      >"${ROOT}/logs/relabel_${tag}_${split}.log" 2>&1
  done
}
prepare_gate 0.35 gate035
prepare_gate 0.50 gate050

run_seed() {
  local gpu="$1" tag="$2" gate="$3" seed="$4"
  local model="${ROOT}/models/${tag}_seed${seed}"
  local validation="${ROOT}/validation/${tag}_seed${seed}"
  local executable="${ROOT}/executable/${tag}_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_argotweak_listwise_selector.py \
    --features "${ROOT}/features/${tag}/train" --output-dir "${model}" \
    --seed "${seed}" --device cuda >"${ROOT}/logs/train_${tag}_seed${seed}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${ROOT}/features/${tag}/val" --checkpoint "${model}/best.pt" \
    --output-dir "${validation}" --device cuda >"${ROOT}/logs/val_${tag}_seed${seed}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_executable_policies.py \
    --episodes "${NATIVE}/val.jsonl" --rankings "${validation}/rankings.jsonl" \
    --budgets 1,3,5,10 --commit-confidence "${gate}" --output-dir "${executable}" \
    >"${ROOT}/logs/executable_${tag}_seed${seed}.log" 2>&1
  touch "${ROOT}/status/${tag}_seed${seed}.done"
}

run_seed 0 gate035 0.35 20261521 & p0=$!
run_seed 1 gate050 0.50 20261521 & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
if [[ "${status}" -ne 0 ]]; then touch "${ROOT}/QUEUE_FAILED"; exit 1; fi

"${PYTHON}" - <<PY
import json
from pathlib import Path
root = Path("${ROOT}")
candidates = []
for tag, gate in (("gate035", 0.35), ("gate050", 0.50)):
    rows = json.loads((root / "executable" / f"{tag}_seed20261521" / "summary.json").read_text())
    top5 = next(row for row in rows if row["policy"] == "learned_top5")
    candidates.append({"tag": tag, "gate": gate, **top5})
winner = max(candidates, key=lambda row: (row["atomic_edit_f1"], row["atomic_edit_precision"]))
promotion = winner["atomic_edit_f1"] > 0.09091 and winner["atomic_edit_precision"] >= 0.70
(root / "promotion.json").write_text(json.dumps({"candidates": candidates, "winner": winner, "promoted": promotion}, indent=2) + "\n")
PY

if "${PYTHON}" -c 'import json; raise SystemExit(0 if json.load(open("'"${ROOT}"'/promotion.json"))["promoted"] else 1)'; then
  winner=$("${PYTHON}" -c 'import json; print(json.load(open("'"${ROOT}"'/promotion.json"))["winner"]["tag"])')
  gate=$("${PYTHON}" -c 'import json; print(json.load(open("'"${ROOT}"'/promotion.json"))["winner"]["gate"])')
  run_seed 0 "${winner}" "${gate}" 20261522 & p2=$!
  run_seed 1 "${winner}" "${gate}" 20261523 & p3=$!
  wait "${p2}"; wait "${p3}"
else
  touch "${ROOT}/NOT_PROMOTED"
fi
touch "${ROOT}/QUEUE_COMPLETED"
