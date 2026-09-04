#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v2.yaml}"
LEDGER="${LEDGER:-${STORAGE_ROOT}/artifacts/paper_results/frozen_test_access/sn7_step0_v2.json}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_results/sn7_step0_frozen_test_v2_20260807}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/sn7_step0_frozen_test_v2}"
POLL_SECONDS="${POLL_SECONDS:-120}"
EXPORTER="${EXPORTER:-scripts/export_sn7_step0_frozen_test_tables.py}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}"

ledger_status() {
  if [[ ! -s "${LEDGER}" ]]; then
    printf '%s\n' "missing"
    return 0
  fi
  "${PYTHON}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("status","missing"))' \
    "${LEDGER}"
}

ledger_returncode() {
  "${PYTHON}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("returncode",""))' \
    "${LEDGER}"
}

echo "[$(date --iso-8601=seconds)] waiting for immutable SN7 frozen test"
while true; do
  status="$(ledger_status)"
  if [[ "${status}" == "complete" ]]; then
    break
  fi
  if [[ "${status}" == "failed" || "${status}" == "command_failed" || "${status}" == "launcher_error" ]]; then
    echo "frozen test ledger reports failure" >&2
    exit 21
  fi
  sleep "${POLL_SECONDS}"
done

[[ "$(ledger_returncode)" == "0" ]] || {
  echo "frozen test completed with a non-zero return code" >&2
  exit 22
}
[[ -s "${RUN_ROOT}/COMPLETE.json" ]] || {
  echo "frozen test ledger completed without COMPLETE.json" >&2
  exit 23
}

if [[ -e "${OUTPUT_ROOT}" ]]; then
  [[ -s "${OUTPUT_ROOT}/manifest.json" ]] || {
    echo "refusing to reuse incomplete paper output: ${OUTPUT_ROOT}" >&2
    exit 24
  }
  echo "paper output already exists and is complete: ${OUTPUT_ROOT}"
  exit 0
fi

"${PYTHON}" "${EXPORTER}" \
  "${REGISTRY}" "${LEDGER}" "${RUN_ROOT}" "${OUTPUT_ROOT}" \
  >"${LOG_ROOT}/paper_export.log" 2>&1

sha256sum "${OUTPUT_ROOT}"/* >"${OUTPUT_ROOT}/SHA256SUMS"
echo "[$(date --iso-8601=seconds)] SN7 frozen closeout complete: ${OUTPUT_ROOT}"
