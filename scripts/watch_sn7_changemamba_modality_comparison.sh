#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
POLL_SECONDS="${POLL_SECONDS:-120}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

full="${RUN_ROOT}/full_weight5_three_seed_20260726.json"
image="${RUN_ROOT}/full_weight5_image_only_three_seed_20260726.json"
prior="${RUN_ROOT}/full_weight5_prior_only_three_seed_20260726.json"
output="${RUN_ROOT}/full_weight5_modality_comparison_20260726.json"

if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing comparison: ${output}" >&2
  exit 1
fi
until [[ -s "${full}" && -s "${image}" && -s "${prior}" ]]; do
  echo "Waiting for all three modality aggregates."
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" scripts/compare_sn7_changemamba_modalities.py \
  "${full}" "${image}" "${prior}" "${output}" \
  --output-markdown \
  "${RUN_ROOT}/full_weight5_modality_comparison_20260726.md"
