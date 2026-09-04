#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1}"
MORPHOLOGY="${MORPHOLOGY:?set MORPHOLOGY to erode or dilate}"
RADIUS="${RADIUS:?set RADIUS to a positive pixel radius}"

case "${MORPHOLOGY}" in erode|dilate) ;; *) exit 2 ;; esac
[[ "${RADIUS}" =~ ^[1-9][0-9]*$ ]] || exit 2

CHECKPOINT="${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
FROZEN_STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/states_val_step0.jsonl"
OUTPUT_ROOT="${RUN_ROOT}/${MORPHOLOGY}${RADIUS}"
RAW_STATES="${OUTPUT_ROOT}/raw_states.jsonl"
ANCHORED_STATES="${OUTPUT_ROOT}/states_val_step0.jsonl"
PARITY_AUDIT="${OUTPUT_ROOT}/PARITY_AUDIT.json"
LOG="${OUTPUT_ROOT}/run.log"

for path in "${CHECKPOINT}" "${EPISODES}" "${FROZEN_STATES}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 4
}
mkdir -p "${OUTPUT_ROOT}"
FAILURE_MARKER="${OUTPUT_ROOT}/FAILED.json"
rm -f "${FAILURE_MARKER}"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"state_generation","exit_code":%d,"test_assets_read":false}\n' \
      "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

cd "${PROJECT_ROOT}"
{
  date -Is
  sha256sum "${CHECKPOINT}" "${EPISODES}" "${FROZEN_STATES}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" -m activemap.cli build-selector-oracle \
      "${CHECKPOINT}" "${EPISODES}" "${RAW_STATES}" \
      --device cuda:0 --image-size 512 \
      --cost-weight 0.18 --false-edit-weight 0.35 \
      --utility-mode executable --utility-profile balanced \
      --writeback-threshold 0.5 --writeback-delta-margin 0.15 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
      --budgets 1.5,3.0,4.5 --operation-update-threshold 0.5 \
      --splits val \
      --prior-input-morphology "${MORPHOLOGY}" \
      --prior-input-morphology-pixels "${RADIUS}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/anchor_sn7_corrupted_states_to_frozen_step0.py \
      "${FROZEN_STATES}" "${RAW_STATES}" "${ANCHORED_STATES}"
  resolved_parity_audit="${PARITY_AUDIT:-${OUTPUT_ROOT}/PARITY_AUDIT.json}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/audit_sn7_frozen_anchor_parity.py \
      "${FROZEN_STATES}" "${ANCHORED_STATES}" "${resolved_parity_audit}"
  sha256sum "${RAW_STATES}" "${ANCHORED_STATES}" "${resolved_parity_audit}"
  date -Is
} >"${LOG}" 2>&1

printf \
  '{"status":"complete","morphology":"%s","radius":%s,"test_assets_read":false}\n' \
  "${MORPHOLOGY}" "${RADIUS}" >"${OUTPUT_ROOT}/COMPLETE.json"
trap - EXIT
