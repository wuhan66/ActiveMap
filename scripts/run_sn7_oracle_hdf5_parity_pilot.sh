#!/usr/bin/env bash
# Compare raw and HDF5-backed oracle inputs without reading frozen test assets.
set -euo pipefail

GPU="${GPU:?set a physical GPU ID}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/system_validation/sn7_oracle_hdf5_pilot8}"
CACHE="${CACHE:-${STORAGE_ROOT}/cache/oracle_input_v1/sn7_train_512_pilot8.h5}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
CHECKPOINT="${CHECKPOINT:-${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260719/best.pt}"
MAX_EPISODES="${MAX_EPISODES:-8}"

for path in "${CACHE}" "${EPISODES}" "${CHECKPOINT}"; do
  test -f "${path}" || { echo "missing input: ${path}" >&2; exit 2; }
done
test ! -e "${RUN_ROOT}" || { echo "refusing to overwrite ${RUN_ROOT}" >&2; exit 3; }
mkdir -p "${RUN_ROOT}/raw" "${RUN_ROOT}/cached"

common_args=(
  --device cuda:0 --image-size 512 --cost-weight 0.18 --false-edit-weight 0.35
  --utility-mode executable --utility-profile balanced
  --writeback-threshold 0.5 --writeback-delta-margin 0.15
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  --budgets 1.5,3.0,4.5 --operation-update-threshold 0.5
  --initial-evidence-strategy min_cost --splits train --max-episodes "${MAX_EPISODES}"
)

run_oracle() {
  local label="$1"
  shift
  local output="${RUN_ROOT}/${label}/states.jsonl"
  local start end
  start="$(date +%s)"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" -m activemap.cli build-selector-oracle \
      "${CHECKPOINT}" "${EPISODES}" "${output}" "${common_args[@]}" "$@" \
      >"${RUN_ROOT}/${label}/run.log" 2>&1
  end="$(date +%s)"
  printf '%s\n' "$((end - start))" >"${RUN_ROOT}/${label}/elapsed_seconds.txt"
}

run_oracle raw
run_oracle cached --input-cache "${CACHE}"

raw_sha="$(sha256sum "${RUN_ROOT}/raw/states.jsonl" | awk '{print $1}')"
cached_sha="$(sha256sum "${RUN_ROOT}/cached/states.jsonl" | awk '{print $1}')"
raw_seconds="$(cat "${RUN_ROOT}/raw/elapsed_seconds.txt")"
cached_seconds="$(cat "${RUN_ROOT}/cached/elapsed_seconds.txt")"
if [[ "${raw_sha}" != "${cached_sha}" ]]; then
  printf '{"status":"failed","reason":"state_sha256_mismatch","raw_sha256":"%s","cached_sha256":"%s"}\n' \
    "${raw_sha}" "${cached_sha}" >"${RUN_ROOT}/PARITY.json"
  exit 4
fi
printf '{"status":"passed","episodes":%s,"raw_seconds":%s,"cached_seconds":%s,"raw_sha256":"%s","cached_sha256":"%s","test_assets_read":false}\n' \
  "${MAX_EPISODES}" "${raw_seconds}" "${cached_seconds}" "${raw_sha}" "${cached_sha}" \
  >"${RUN_ROOT}/PARITY.json"
