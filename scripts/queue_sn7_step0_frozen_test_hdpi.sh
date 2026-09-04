#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v2.yaml}"
TEST_MANIFEST="${TEST_MANIFEST:-${STORAGE_ROOT}/manifests/sn7_frozen_v2/test_release_v1/sn7_test_path_manifest.parquet}"
LEDGER="${LEDGER:-${STORAGE_ROOT}/artifacts/paper_results/frozen_test_access/sn7_step0_v2.json}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v2}"
GPUS="${GPUS:-1 4 5}"

read -r -a gpu_ids <<<"${GPUS}"
(( ${#gpu_ids[@]} > 0 && ${#gpu_ids[@]} <= 4 )) || {
  echo "frozen test requires one to four GPUs" >&2
  exit 51
}
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" != "0" && "${gpu}" != "2" ]] || {
    echo "GPU ${gpu} is reserved and cannot be used" >&2
    exit 52
  }
done

export PROJECT_ROOT STORAGE_ROOT PYTHON REGISTRY RUN_ROOT GPUS
export SN7_TEST_MANIFEST="${TEST_MANIFEST}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON}" "${PROJECT_ROOT}/scripts/run_frozen_paper_test.py" \
  --purpose iclr2027_sn7_step0_one_time_test --confirm-frozen \
  "${REGISTRY}" "${STORAGE_ROOT}" "${LEDGER}" -- \
  bash "${PROJECT_ROOT}/scripts/run_sn7_step0_frozen_test_v2.sh"
