#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
ENCODER="${STORAGE_ROOT}/models/change_mamba/vssm_tiny_0230_ckpt_epoch_262.pth"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
LOG_ROOT="${STORAGE_ROOT}/logs/changemamba_proposal_jitter_20260726"
BAN_RESULT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban/ban_vs_changemamba_aoi_bootstrap_20260726.json"
AUDIT_RESULT="${RUN_ROOT}/candidate_localization_audit_20260726.json"
POLL_SECONDS="${POLL_SECONDS:-120}"
GPU_IMAGE_PRIOR="${JITTER_GPU_IMAGE_PRIOR:-1}"
GPU_PRIOR_ONLY="${JITTER_GPU_PRIOR_ONLY:-2}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
while [[ ! -s "${BAN_RESULT}" || ! -s "${AUDIT_RESULT}" ]]; do
  echo "Waiting for BAN comparison and candidate-localization audit."
  sleep "${POLL_SECONDS}"
done

run_one() {
  local gpu="$1"
  local mode="$2"
  local output="${RUN_ROOT}/proposal_jitter16_${mode}_seed20260725_v1"
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite proposal-jitter run: ${output}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_sn7_changemamba.py \
    "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${output}" \
    --encoder-checkpoint "${ENCODER}" \
    --device cuda:0 --image-size 128 --batch-size 16 --workers 8 \
    --epochs 40 --min-epochs 10 --patience 8 \
    --learning-rate 0.0001 --positive-class-weight 5 \
    --max-translation-pixels 16 --input-mode "${mode}" --seed 20260725 \
    > "${LOG_ROOT}/${mode}_seed20260725.log" 2>&1
}

run_one "${GPU_IMAGE_PRIOR}" image_prior &
pid1="$!"
run_one "${GPU_PRIOR_ONLY}" prior_only &
pid2="$!"
status=0
for pid in "${pid1}" "${pid2}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
full = json.loads(
    (root / "proposal_jitter16_image_prior_seed20260725_v1/summary.json").read_text()
)
prior = json.loads(
    (root / "proposal_jitter16_prior_only_seed20260725_v1/summary.json").read_text()
)
visual_gain = full["best_committed_map_iou"] - prior["best_committed_map_iou"]
payload = {
    "schema_version": "sn7-proposal-jitter-pilot-v1",
    "max_translation_pixels": 16,
    "full_best_committed_map_iou": full["best_committed_map_iou"],
    "prior_only_best_committed_map_iou": prior["best_committed_map_iou"],
    "visual_gain_over_prior_only": visual_gain,
    "three_seed_promotion": (
        full["best_committed_map_iou"] > 0.6420103859798584
        and visual_gain > 0.03
    ),
    "test_assets_read": False,
}
path = root / "proposal_jitter16_pilot_20260726.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
