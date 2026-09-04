#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
EPISODES="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes/val.jsonl"
RANKINGS="${STORE}/runs/argotweak/map_conditioned_selector_screen_v1/validation/qwen_structured_seed20261511/rankings.jsonl"
ROOT="${STORE}/runs/argotweak/executable_map_policy_matrix_v2"
mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/.lock"; flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

run_threshold() {
  local gpu="$1" threshold="$2" tag="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_executable_policies.py \
    --episodes "${EPISODES}" \
    --rankings "${RANKINGS}" \
    --budgets 1,3,5,10 \
    --commit-confidence "${threshold}" \
    --output-dir "${ROOT}/${tag}" \
    >"${ROOT}/logs/${tag}.log" 2>&1
  touch "${ROOT}/status/${tag}.done"
}

run_threshold 0 0.35 gate035 & p0=$!
run_threshold 1 0.50 gate050 & p1=$!
run_threshold 2 0.65 gate065 & p2=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
wait "${p2}" || status=1
if [[ "${status}" -ne 0 ]]; then touch "${ROOT}/QUEUE_FAILED"; exit 1; fi

"${PYTHON}" -c 'import json,pathlib; root=pathlib.Path("'"${ROOT}"'"); rows=[]; [(rows.extend({**r,"gate":d.name} for r in json.loads((d/"summary.json").read_text()))) for d in sorted(root.glob("gate*"))]; (root/"paper_table.json").write_text(json.dumps(rows,indent=2)+"\n")'
touch "${ROOT}/QUEUE_COMPLETED"
