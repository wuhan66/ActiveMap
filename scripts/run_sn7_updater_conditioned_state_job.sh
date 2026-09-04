#!/usr/bin/env bash
# Generate one immutable counterfactual-state shard for the C5 updater-conditioned audit.
set -euo pipefail

STATE="${STATE:?set STATE to f0 or f1}"
SPLIT="${SPLIT:?set SPLIT to train or val}"
GPU="${GPU:?set physical GPU ID}"
RUN_ROOT="${RUN_ROOT:?set immutable output root}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"

case "${STATE}" in f0|f1) ;; *) echo "invalid STATE: ${STATE}" >&2; exit 2 ;; esac
case "${SPLIT}" in train|val) ;; *) echo "invalid SPLIT: ${SPLIT}" >&2; exit 2 ;; esac
case "${GPU}" in ''|*[!0-9]*) echo "invalid GPU: ${GPU}" >&2; exit 2 ;; esac

F0_CHECKPOINT="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260719/best.pt"
F1_CHECKPOINT="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260722/best.pt"
if [[ "${STATE}" == "f0" ]]; then
  CHECKPOINT="${F0_CHECKPOINT}"
else
  CHECKPOINT="${F1_CHECKPOINT}"
fi

if [[ "${SPLIT}" == "train" ]]; then
  EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl"
else
  EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
fi
JOB_ROOT="${RUN_ROOT}/states/${STATE}/${SPLIT}"
OUTPUT="${JOB_ROOT}/states.jsonl"
LOG="${JOB_ROOT}/run.log"
INPUT_CACHE="${INPUT_CACHE:-}"

for path in "${CHECKPOINT}" "${EPISODES}"; do
  test -f "${path}" || { echo "missing input: ${path}" >&2; exit 2; }
done
test ! -e "${OUTPUT}" || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
if [[ -n "${INPUT_CACHE}" ]]; then
  test -f "${INPUT_CACHE}" || { echo "missing INPUT_CACHE: ${INPUT_CACHE}" >&2; exit 2; }
  CACHE_ARGS=(--input-cache "${INPUT_CACHE}")
else
  CACHE_ARGS=()
fi
mkdir -p "${JOB_ROOT}"

failure="${JOB_ROOT}/FAILED.json"
record_failure() {
  local status=$?
  trap - EXIT
  if (( status != 0 )); then
    printf '{"status":"failed","stage":"counterfactual_state_generation","state":"%s","split":"%s","exit_code":%d,"test_assets_read":false}\n' \
      "${STATE}" "${SPLIT}" "${status}" > "${failure}"
  fi
  exit "${status}"
}
trap record_failure EXIT

{
  date -Is
  printf 'state=%s split=%s physical_gpu=%s\n' "${STATE}" "${SPLIT}" "${GPU}"
  sha256sum "${CHECKPOINT}" "${EPISODES}"
  cd "${PROJECT_ROOT}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" -m activemap.cli build-selector-oracle \
      "${CHECKPOINT}" "${EPISODES}" "${OUTPUT}" \
      --device cuda:0 --image-size 512 --cost-weight 0.18 --false-edit-weight 0.35 \
      --utility-mode executable --utility-profile balanced \
      --writeback-threshold 0.5 --writeback-delta-margin 0.15 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
      --budgets 1.5,3.0,4.5 --operation-update-threshold 0.5 \
      --initial-evidence-strategy min_cost --splits "${SPLIT}" \
      "${CACHE_ARGS[@]}"
  sha256sum "${OUTPUT}"
  date -Is
} > "${LOG}" 2>&1

printf '{"status":"complete","state":"%s","split":"%s","checkpoint":"%s","episodes":"%s","test_assets_read":false}\n' \
  "${STATE}" "${SPLIT}" "${CHECKPOINT}" "${EPISODES}" > "${JOB_ROOT}/COMPLETE.json"
trap - EXIT
