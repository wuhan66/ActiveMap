#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
OPENCD_ROOT="${STORAGE_ROOT}/external/open_cd"
CONFIG="${OPENCD_ROOT}/configs/ban/ban_vit-b16-clip_mit-b0_512x512_40k_levircd.py"
MODEL_ROOT="${STORAGE_ROOT}/models/opencd_ban"
CLIP="${MODEL_ROOT}/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
SIDE="${MODEL_ROOT}/mit_b0_20220624-7e0fe6dd.pth"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
LOG_ROOT="${STORAGE_ROOT}/logs/opencd_ban_formal_20260726"
GATE="${RUN_ROOT}/gate_summary_20260726.json"
BELIEF_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_belief_ablation_v1/severity4/ungated"
UPSTREAM_SEED25="${BAN_UPSTREAM_SEED25:-${BELIEF_ROOT}/seed20260730/COMPLETE.json}"
UPSTREAM_SEED26="${BAN_UPSTREAM_SEED26:-${BELIEF_ROOT}/seed20260731/COMPLETE.json}"
UPSTREAM_SEED27="${BAN_UPSTREAM_SEED27:-${BELIEF_ROOT}/seed20260801/COMPLETE.json}"
POLL_SECONDS="${POLL_SECONDS:-120}"
GPU_SEED25="${BAN_GPU_SEED25:-1}"
GPU_SEED26="${BAN_GPU_SEED26:-2}"
GPU_SEED27="${BAN_GPU_SEED27:-3}"
LOCK_FILE="${RUN_ROOT}/.formal.lock"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "BAN formal watcher is already active."
  exit 0
fi
if [[ -s "${RUN_ROOT}/full_weight5_three_seed_20260727.json" &&
      -s "${RUN_ROOT}/ban_vs_changemamba_aoi_bootstrap_20260727.json" ]]; then
  echo "BAN formal three-seed comparison is already complete."
  exit 0
fi
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
until [[ -s "${GATE}" ]] && \
  "${PYTHON}" -c "import json; assert json.load(open('${GATE}'))['passed']"; do
  echo "Waiting for BAN smoke and overfit gates."
  sleep "${POLL_SECONDS}"
done

gpu_is_idle() {
  local gpu="$1"
  local memory
  local utilization
  IFS=, read -r memory utilization < <(
    nvidia-smi --id="${gpu}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  memory="${memory// /}"
  utilization="${utilization// /}"
  [[ "${memory}" -lt 1024 && "${utilization}" -lt 15 ]]
}

wait_for_gpu() {
  local gpu="$1"
  until gpu_is_idle "${gpu}"; do
    echo "GPU ${gpu} is occupied; waiting ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
}

wait_for_marker() {
  local marker="$1"
  while [[ -n "${marker}" && ! -s "${marker}" ]]; do
    echo "Waiting for upstream experiment: ${marker}"
    sleep "${POLL_SECONDS}"
  done
}

run_seed() {
  local gpu="$1"
  local seed="$2"
  local run="${RUN_ROOT}/full_weight5_seed${seed}_v1"
  local audit="${run}/val_audit"
  if [[ -e "${run}" ]]; then
    echo "Refusing to overwrite existing run: ${run}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_sn7_opencd_ban.py \
    "${MANIFEST}" "${OPENCD_ROOT}" "${CONFIG}" "${run}" \
    --clip-checkpoint "${CLIP}" \
    --side-checkpoint "${SIDE}" \
    --device cuda:0 --image-size 128 --batch-size 8 --workers 8 \
    --epochs 40 --min-epochs 10 --patience 8 \
    --learning-rate 0.0001 --positive-class-weight 5 \
    --seed "${seed}" \
    > "${LOG_ROOT}/seed${seed}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_sn7_opencd_ban.py \
    "${run}/best.pt" "${MANIFEST}" "${OPENCD_ROOT}" "${CONFIG}" "${audit}" \
    --split val --device cuda:0 --batch-size 8 --workers 8 \
    > "${LOG_ROOT}/seed${seed}_audit.log" 2>&1
}

wait_for_marker "${UPSTREAM_SEED25}"
wait_for_gpu "${GPU_SEED25}"
run_seed "${GPU_SEED25}" 20260725 &
pid1="$!"
wait_for_marker "${UPSTREAM_SEED26}"
wait_for_gpu "${GPU_SEED26}"
run_seed "${GPU_SEED26}" 20260726 &
pid2="$!"
wait_for_marker "${UPSTREAM_SEED27}"
wait_for_gpu "${GPU_SEED27}"
run_seed "${GPU_SEED27}" 20260727 &
pid3="$!"
status=0
for pid in "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON}" scripts/aggregate_sn7_opencd_ban.py \
  "${RUN_ROOT}/full_weight5_three_seed_20260726.json" \
  "${RUN_ROOT}/full_weight5_seed20260725_v1" \
  "${RUN_ROOT}/full_weight5_seed20260726_v1" \
  "${RUN_ROOT}/full_weight5_seed20260727_v1" \
  --output-markdown "${RUN_ROOT}/full_weight5_three_seed_20260726.md"

"${PYTHON}" scripts/bootstrap_sn7_updater_comparison.py \
  --baseline \
    "${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba/full_weight5_seed20260725_v1/val_audit" \
    "${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba/full_weight5_seed20260726_v1/val_audit" \
    "${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba/full_weight5_seed20260727_v1/val_audit" \
  --candidate \
    "${RUN_ROOT}/full_weight5_seed20260725_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_seed20260726_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_seed20260727_v1/val_audit" \
  --baseline-name changemamba \
  --candidate-name ban \
  --output "${RUN_ROOT}/ban_vs_changemamba_aoi_bootstrap_20260726.json" \
  --output-markdown "${RUN_ROOT}/ban_vs_changemamba_aoi_bootstrap_20260726.md" \
  --draws 5000 --seed 20260726
