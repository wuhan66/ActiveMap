#!/usr/bin/env bash
set -euo pipefail

# Independent visual-backend matrix. It never reads the frozen SN7 test set.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORE="${STORE:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${STORE}/runs/updater/nts_visual_matrix_20260808"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

mkdir -p "${RUN_ROOT}/configs" "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
exec 9>"${RUN_ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${RUN_ROOT}/COMPLETE.json" ]] || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for path in "${PYTHON}" \
  "${STORE}/processed/inria_v1/segmentation/updater_samples.jsonl" \
  "${STORE}/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl" \
  "${STORE}/processed/muno21_v2/updater/updater_samples.jsonl"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 2; }
done
IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
[[ "${#GPUS[@]}" -eq 4 ]] || { echo "GPU_IDS must contain four devices" >&2; exit 2; }

labels=(inria_seed20260821 sn7_v4_seed20260821 muno_v7_seed20260832 muno_v7_prior_roi_seed20260833)
seeds=(20260821 20260821 20260832 20260833)
templates=(
  configs/updater/inria_pretrain_server.yaml
  configs/updater/sn7_v4_hierarchical_vector_change_scratch_server.yaml
  configs/updater/muno21_road_v7_topology_scratch_server.yaml
  configs/updater/muno21_road_v7_topology_scratch_server.yaml
)

prepare_config() {
  local template="$1" output="$2" seed="$3" roi="$4" target="$5"
  sed \
    -e "s|/home/wh/ActiveMap|${STORE}|g" \
    -e "s|^seed: .*|seed: ${seed}|" \
    -e "s|^output_dir: .*|output_dir: ${output}|" \
    -e "s|^archive_dir: .*|archive_dir: ${output}/archive|" \
    -e 's|resume: true|resume: false|' \
    "${template}" >"${target}"
  if [[ "${roi}" == "true" ]]; then
    if grep -q '^  prior_guided_roi:' "${target}"; then
      sed -i 's|prior_guided_roi: false|prior_guided_roi: true|' "${target}"
    else
      sed -i '/^  vector_change_encoder: true$/a\  prior_guided_roi: true' "${target}"
    fi
  fi
}

run_one() {
  local index="$1"
  local label="${labels[$index]}"
  local seed="${seeds[$index]}"
  local template="${templates[$index]}"
  local gpu="${GPUS[$index]}"
  local run="${RUN_ROOT}/${label}"
  local config="${RUN_ROOT}/configs/${label}.yaml"
  local log="${RUN_ROOT}/logs/${label}.log"
  local roi=false
  [[ "${label}" == *prior_roi* ]] && roi=true
  [[ ! -e "${run}" ]] || { echo "refusing partial run: ${run}" >&2; return 3; }
  prepare_config "${template}" "${run}" "${seed}" "${roi}" "${config}"
  {
    echo "$(date -Is) START ${label} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap.cli train-updater "${config}"
    test -s "${run}/state.json"
    date -Is >"${RUN_ROOT}/status/${label}.done"
    echo "$(date -Is) DONE ${label}"
  } >"${log}" 2>&1
}

pids=()
for index in 0 1 2 3; do
  run_one "${index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if ((status == 0)); then
  printf '{"status":"complete","schema_version":"nts-visual-updater-matrix-v1","split":"train-validation-only","test_assets_read":false}\n' >"${RUN_ROOT}/COMPLETE.json"
else
  printf '{"status":"failed","split":"train-validation-only","test_assets_read":false}\n' >"${RUN_ROOT}/FAILED.json"
  exit 1
fi
