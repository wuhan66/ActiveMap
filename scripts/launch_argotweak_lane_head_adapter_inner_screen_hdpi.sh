#!/usr/bin/env bash
set -euo pipefail

# One-seed diagnostic only. The outer eight-log validation split is deliberately
# excluded from training and checkpoint selection.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
OFFICIAL="${STORE}/external/ArgoTweak_baselines"
PYTHON="${STORE}/envs/argotweak-legacy/bin/python"
DATA="${STORE}/datasets/argotweak/tbv_balanced_24_8_v1/official_full"
SOURCE_SUBSET="${STORE}/datasets/argotweak/subsets/balanced_24_8_v1/selected_train24_val8.json"
ROOT="${STORE}/runs/argotweak/lane_head_adapter_inner_screen_v1"
WORK_DIR="${ROOT}/seed20260841"
GPU_IDS="${GPU_IDS:-1,3,4,5}"
MASTER_PORT="${MASTER_PORT:-29541}"

for path in \
  "${PROJECT_ROOT}/scripts/build_argotweak_inner_adapter_split.py" \
  "${PROJECT_ROOT}/scripts/build_argotweak_official_pilot.py" \
  "${PROJECT_ROOT}/scripts/run_argotweak_official_train.py" \
  "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
  "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
  "${SOURCE_SUBSET}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
if [[ -e "${ROOT}/TRAINING_COMPLETED" ]]; then
  echo "inner screen already completed: ${ROOT}"
  exit 0
fi
if compgen -G "${WORK_DIR}/epoch_*.pth" >/dev/null; then
  echo "refusing to restart after a partial training epoch: ${WORK_DIR}" >&2
  exit 4
fi

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
[[ "${#GPUS[@]}" -eq 4 ]] || { echo "GPU_IDS requires four physical GPU ids" >&2; exit 5; }
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" != "0" && "${gpu}" != "2" ]] || { echo "GPU ${gpu} is reserved" >&2; exit 5; }
done

mkdir -p "${ROOT}/official_inner" "${WORK_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${OFFICIAL}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${ROOT}/inner_split.json" ]]; then
  "${PYTHON}" scripts/build_argotweak_inner_adapter_split.py \
    --subset "${SOURCE_SUBSET}" --output "${ROOT}/inner_split.json" --holdout-count 6
fi
if [[ ! -s "${ROOT}/official_inner/train_inner18.pkl" ]]; then
  "${PYTHON}" scripts/build_argotweak_official_pilot.py \
    --official-root "${OFFICIAL}" --tbv-root "${STORE}/datasets/argotweak/tbv_balanced_24_8_v1/segments" \
    --extension-root "${STORE}/datasets/argotweak/raw/extension_train_val" \
    --output-dir "${ROOT}/official_inner" --split train --subset-file "${ROOT}/inner_split.json" \
    --output-name train_inner18
fi
if [[ ! -s "${ROOT}/official_inner/val_inner6.pkl" ]]; then
  "${PYTHON}" scripts/build_argotweak_official_pilot.py \
    --official-root "${OFFICIAL}" --tbv-root "${STORE}/datasets/argotweak/tbv_balanced_24_8_v1/segments" \
    --extension-root "${STORE}/datasets/argotweak/raw/extension_train_val" \
    --output-dir "${ROOT}/official_inner" --split val --subset-file "${ROOT}/inner_split.json" \
    --output-name val_inner6
fi

cat >"${ROOT}/protocol.json" <<EOF
{
  "schema_version": "activemap-argotweak-lane-head-inner-screen-v1",
  "seed": 20260841,
  "train_logs": 18,
  "inner_holdout_logs": 6,
  "outer_confirmation_logs": 8,
  "epochs": 8,
  "optimizer_lr": 0.00002,
  "effective_global_batch": 4,
  "freeze_prefixes": ["img_backbone", "img_neck", "map_encoder", "bev_constructor"],
  "outer_validation_used_for_selection": false,
  "test_assets_read": false
}
EOF

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
"${PYTHON}" -m torch.distributed.launch --nproc_per_node=4 --master_port="${MASTER_PORT}" \
  scripts/run_argotweak_official_train.py \
  --official-root "${OFFICIAL}" --config "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
  --work-dir "${WORK_DIR}" --train-ann "${ROOT}/official_inner/train_inner18.pkl" \
  --val-ann "${ROOT}/official_inner/val_inner6.pkl" \
  --checkpoint "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
  --epochs 8 --workers 2 --seed 20260841 --launcher pytorch \
  --freeze-prefix img_backbone --freeze-prefix img_neck --freeze-prefix map_encoder \
  --freeze-prefix bev_constructor --optimizer-lr 0.00002 \
  --auto-scale-base-batch-size 4 --evaluation-interval 1 --max-keep-checkpoints 8 \
  >"${ROOT}/train.log" 2>&1

date -Is >"${ROOT}/TRAINING_COMPLETED"
echo "Inner screen complete. Select a checkpoint using only ${ROOT}/seed20260841/ logs."
