#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v3_cap20.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v3_cap20}"
LEDGER="${LEDGER:-${STORAGE_ROOT}/artifacts/paper_results/frozen_test_access/sn7_step0_v3_cap20_postprocess_retry4.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_results/sn7_step0_frozen_test_v3_cap20_physical_gpu_20260808}"
SLICE_ROOT="${SLICE_ROOT:-${STORAGE_ROOT}/artifacts/paper_results/sn7_step0_frozen_test_operation_slices_v3_cap20_20260808}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/sn7_step0_frozen_test_v3_cap20}"
GPUS="${GPUS:-1 3 4}"
PURPOSE="${PURPOSE:-iclr2027_sn7_step0_postprocess_retry4}"

cd "${PROJECT_ROOT}"
export PROJECT_ROOT STORAGE_ROOT PYTHON REGISTRY RUN_ROOT GPUS
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}" "$(dirname "${LEDGER}")" "$(dirname "${OUTPUT_ROOT}")"

read -r -a gpu_ids <<<"${GPUS}"
(( ${#gpu_ids[@]} == 3 )) || { echo "this closeout requires exactly three mapped GPUs" >&2; exit 61; }
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" != "0" && "${gpu}" != "2" ]] || { echo "GPU ${gpu} is reserved" >&2; exit 62; }
done
[[ ! -e "${LEDGER}" ]] || { echo "immutable retry ledger already exists: ${LEDGER}" >&2; exit 63; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "paper export already exists: ${OUTPUT_ROOT}" >&2; exit 64; }
[[ ! -e "${SLICE_ROOT}" ]] || { echo "operation-slice export already exists: ${SLICE_ROOT}" >&2; exit 65; }
[[ -s "${RUN_ROOT}/three_policy_summary.json" ]] || { echo "missing completed controller summary" >&2; exit 66; }
for seed in 20260730 20260731 20260801; do
  for variant in notool forced benefit; do
    result="${RUN_ROOT}/seed${seed}/${variant}/writeback/process_result.json"
    [[ -s "${result}" ]] || { echo "missing writeback result: ${result}" >&2; exit 67; }
    "${PYTHON}" - "${result}" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
assert row["status"] == "completed" and int(row["returncode"]) == 0
PY
  done
done

"${PYTHON}" scripts/run_frozen_paper_test.py \
  --purpose "${PURPOSE}" --confirm-frozen \
  "${REGISTRY}" "${STORAGE_ROOT}" "${LEDGER}" -- \
  bash scripts/resume_sn7_step0_frozen_test_v3_cap20_after_split_fix.sh

"${PYTHON}" scripts/export_sn7_step0_frozen_test_tables_v3_cap20.py \
  "${REGISTRY}" "${LEDGER}" "${RUN_ROOT}" "${OUTPUT_ROOT}" \
  >"${LOG_ROOT}/paper_export_postprocess_retry4.log" 2>&1
sha256sum "${OUTPUT_ROOT}"/* >"${OUTPUT_ROOT}/SHA256SUMS"

RUN_ROOT="${RUN_ROOT}" \
CANONICAL_ROOT="${OUTPUT_ROOT}" \
OUTPUT_ROOT="${SLICE_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
POLL_SECONDS=5 \
bash scripts/queue_sn7_frozen_operation_slices_hdpi.sh

printf '%s\n' "SN7_RETRY4_PAPER_EXPORT_COMPLETE" >"${OUTPUT_ROOT}/PIPELINE_COMPLETE"
echo "completed SN7 frozen postprocess retry4 and paper exports"
