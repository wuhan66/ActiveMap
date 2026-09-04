#!/usr/bin/env bash
set -euo pipefail

# A single, hypothesis-driven repair: retain visual features and adapt only the
# map-prior encoder plus output head. The previous lane-head-only run showed
# marginal all-mAP improvement but worse changed-mAP on this exact inner split.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
OFFICIAL="${STORE}/external/ArgoTweak_baselines"
PYTHON="${STORE}/envs/argotweak-legacy/bin/python"
SOURCE_ROOT="${STORE}/runs/argotweak/lane_head_adapter_inner_screen_v1"
ROOT="${STORE}/runs/argotweak/map_encoder_adapter_inner_screen_v1"
WORK_DIR="${ROOT}/seed20260842"
GPU_IDS="${GPU_IDS:-1,3,4,5}"
MASTER_PORT="${MASTER_PORT:-29542}"

for path in \
  "${PROJECT_ROOT}/scripts/run_argotweak_official_train.py" \
  "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
  "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
  "${SOURCE_ROOT}/official_inner/train_inner18.pkl" \
  "${SOURCE_ROOT}/official_inner/val_inner6.pkl" \
  "${SOURCE_ROOT}/inner_split.json"; do
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

mkdir -p "${ROOT}" "${WORK_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${OFFICIAL}${PYTHONPATH:+:${PYTHONPATH}}"

cat >"${ROOT}/protocol.json" <<EOF
{
  "schema_version": "activemap-argotweak-map-encoder-inner-screen-v1",
  "seed": 20260842,
  "train_logs": 18,
  "inner_holdout_logs": 6,
  "outer_confirmation_logs": 8,
  "epochs": 4,
  "optimizer_lr": 0.00001,
  "effective_global_batch": 4,
  "freeze_prefixes": ["img_backbone", "img_neck", "bev_constructor"],
  "split_source": "lane_head_adapter_inner_screen_v1/inner_split.json",
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
  --work-dir "${WORK_DIR}" --train-ann "${SOURCE_ROOT}/official_inner/train_inner18.pkl" \
  --val-ann "${SOURCE_ROOT}/official_inner/val_inner6.pkl" \
  --checkpoint "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
  --epochs 4 --workers 2 --seed 20260842 --launcher pytorch \
  --freeze-prefix img_backbone --freeze-prefix img_neck --freeze-prefix bev_constructor \
  --optimizer-lr 0.00001 --auto-scale-base-batch-size 4 \
  --evaluation-interval 1 --max-keep-checkpoints 4 \
  >"${ROOT}/train.log" 2>&1

date -Is >"${ROOT}/TRAINING_COMPLETED"
echo "Inner screen complete. Select a checkpoint using only the frozen inner holdout."
