#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
OFFICIAL="${OFFICIAL:-${STORE}/external/ArgoTweak_baselines}"
PYTHON="${PYTHON:-${STORE}/envs/argotweak-legacy/bin/python}"
DATA="${STORE}/datasets/argotweak/tbv_balanced_24_8_v1/official_full"
RUN="${STORE}/runs/argotweak/full_domain_adaptation_balanced24_v1/four_gpu_seed20260831"
LOG="${STORE}/logs/argotweak_full_domain_adaptation_balanced24_seed20260831.log"

for path in \
  "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
  "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
  "${DATA}/train_argotweak_balanced24.pkl" \
  "${DATA}/val_argotweak_balanced8.pkl"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
[[ ! -e "${RUN}" ]] || { echo "refusing existing output: ${RUN}" >&2; exit 4; }
mkdir -p "${RUN}" "$(dirname "${LOG}")"

cat >"${RUN}/protocol.json" <<EOF
{
  "schema_version": "activemap-argotweak-full-domain-adaptation-v1",
  "train_logs": 24,
  "val_logs": 8,
  "official_frames": 3615,
  "initial_checkpoint_sha256": "c42c29a88223db900464cfa88ef2a6f6c5575e5baf899b9c01a07bd674c31ce9",
  "epochs": 10,
  "physical_gpus": [1, 2, 3, 4],
  "samples_per_gpu": 1,
  "auto_scale_lr_base_batch_size": 8,
  "seed": 20260831,
  "split": "train+validation",
  "test_assets_read": false
}
EOF

export PYTHONPATH="${OFFICIAL}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1,2,3,4
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

"${PYTHON}" -m torch.distributed.launch \
  --nproc_per_node=4 --master_port=29531 \
  "${PROJECT_ROOT}/scripts/run_argotweak_official_train.py" \
  --official-root "${OFFICIAL}" \
  --config "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
  --work-dir "${RUN}" \
  --train-ann "${DATA}/train_argotweak_balanced24.pkl" \
  --val-ann "${DATA}/val_argotweak_balanced8.pkl" \
  --checkpoint "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
  --epochs 10 --workers 4 --seed 20260831 \
  --launcher pytorch --autoscale-lr >"${LOG}" 2>&1

date -Is >"${RUN}/TRAINING_COMPLETED"
