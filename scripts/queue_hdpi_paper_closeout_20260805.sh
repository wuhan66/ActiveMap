#!/usr/bin/env bash
set -euo pipefail

# Validation-only closeout. This queue aggregates completed official runs;
# it deliberately does not launch inference, training, or frozen-test access.
PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORE}/envs/activemap-agent/bin/python}"
ROOT="${STORE}/runs/queues/hdpi_paper_closeout_20260805"
OFFICIAL_ROOT="${STORE}/runs/muno21_v12_official_three_seed_20260802"
OUTPUT_ROOT="${STORE}/runs/paper_evidence/muno21_official_three_seed_20260805"
STOP_BASELINE="${STORE}/runs/muno21_active_catalog_transfer_v1/official_always_stop/official_metrics.jsonl"
OLD_BASELINE="${STORE}/runs/muno21_evidence_value_official_val_v2_20260725/old_generic/official_metrics.jsonl"

mkdir -p "${ROOT}/logs" "${ROOT}/status" "${OUTPUT_ROOT}"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

for path in "${PYTHON}" "${STOP_BASELINE}" "${OLD_BASELINE}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; touch "${ROOT}/QUEUE_FAILED"; exit 3; }
done

candidate_args=()
for seed in 20260822 20260823 20260824; do
  path="${OFFICIAL_ROOT}/seed${seed}/edit_conditioned_proactive_tools_safe_delta/official_metrics.jsonl"
  [[ -s "${path}" ]] || { echo "missing candidate: ${path}" >&2; touch "${ROOT}/QUEUE_FAILED"; exit 3; }
  candidate_args+=(--candidate "seed${seed}=${path}")
done

aggregate_one() {
  local label="$1" baseline="$2" output="$3"
  if [[ ! -s "${output}" ]]; then
    "${PYTHON}" -m scripts.aggregate_muno21_official_three_seeds \
      "${baseline}" "${output}" "${candidate_args[@]}" \
      --repetitions 10000 --seed 20260805 \
      >"${ROOT}/logs/aggregate_${label}.json" 2>&1
  fi
}

cat >"${OUTPUT_ROOT}/protocol.json" <<EOF
{"schema_version":"activemap-paper-closeout-v1","split":"val","test_assets_read":false,"official_policy":"edit_conditioned_proactive_tools_safe_delta","seeds":[20260822,20260823,20260824],"bootstrap_repetitions":10000,"purpose":"aggregate completed official MUNO21 validation metrics; no new inference or training"}
EOF

aggregate_one vs_always_stop "${STOP_BASELINE}" "${OUTPUT_ROOT}/official_vs_always_stop.json"
aggregate_one vs_old_selector "${OLD_BASELINE}" "${OUTPUT_ROOT}/official_vs_old_selector.json"

printf '%s\n' "validation-only aggregation completed" >"${ROOT}/status/aggregation.done"
date -Is >"${ROOT}/QUEUE_COMPLETED"
