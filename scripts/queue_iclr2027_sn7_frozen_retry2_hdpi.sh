#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v3_cap20.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v3_cap20}"
LEDGER="${LEDGER:-${STORAGE_ROOT}/artifacts/paper_results/frozen_test_access/sn7_step0_v3_cap20_physical_gpu_retry2.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_results/sn7_step0_frozen_test_v3_cap20_physical_gpu_20260808}"
SLICE_ROOT="${SLICE_ROOT:-${STORAGE_ROOT}/artifacts/paper_results/sn7_step0_frozen_test_operation_slices_v3_cap20_20260808}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/sn7_step0_frozen_test_v3_cap20}"
GPUS="${GPUS:-1 3 4}"

cd "${PROJECT_ROOT}"
export PROJECT_ROOT STORAGE_ROOT PYTHON REGISTRY RUN_ROOT GPUS
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}" "$(dirname "${LEDGER}")" "$(dirname "${OUTPUT_ROOT}")"

read -r -a gpu_ids <<<"${GPUS}"
(( ${#gpu_ids[@]} == 3 )) || { echo "this retry requires exactly three GPUs" >&2; exit 61; }
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" != "0" && "${gpu}" != "2" ]] || { echo "GPU ${gpu} is reserved" >&2; exit 62; }
done
[[ ! -e "${LEDGER}" ]] || { echo "immutable retry ledger already exists: ${LEDGER}" >&2; exit 63; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "paper export already exists: ${OUTPUT_ROOT}" >&2; exit 64; }

"${PYTHON}" scripts/run_frozen_paper_test.py \
  --purpose iclr2027_sn7_step0_physical_gpu_interruption_retry2 --confirm-frozen \
  "${REGISTRY}" "${STORAGE_ROOT}" "${LEDGER}" -- \
  bash scripts/resume_sn7_step0_frozen_test_v3_cap20_after_split_fix.sh

"${PYTHON}" scripts/export_sn7_step0_frozen_test_tables_v3_cap20.py \
  "${REGISTRY}" "${LEDGER}" "${RUN_ROOT}" "${OUTPUT_ROOT}" \
  >"${LOG_ROOT}/paper_export_physical_gpu_retry2.log" 2>&1
sha256sum "${OUTPUT_ROOT}"/* >"${OUTPUT_ROOT}/SHA256SUMS"

RUN_ROOT="${RUN_ROOT}" \
CANONICAL_ROOT="${OUTPUT_ROOT}" \
OUTPUT_ROOT="${SLICE_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
POLL_SECONDS=5 \
bash scripts/queue_sn7_frozen_operation_slices_hdpi.sh

printf '%s\n' "SN7_RETRY2_PAPER_EXPORT_COMPLETE" >"${OUTPUT_ROOT}/PIPELINE_COMPLETE"
echo "completed SN7 frozen retry2 and paper exports"
