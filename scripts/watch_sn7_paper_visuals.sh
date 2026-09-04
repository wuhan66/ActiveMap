#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
OPENCD_ROOT="${STORAGE_ROOT}/external/open_cd"
OPENCD_CONFIG="${OPENCD_ROOT}/configs/ban/ban_vit-b16-clip_mit-b0_512x512_40k_levircd.py"
CHANGE_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
BAN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
CHANGE_RUN="${CHANGE_ROOT}/full_weight5_seed20260725_v1"
BAN_RUN="${BAN_ROOT}/full_weight5_seed20260725_v1"
VISUAL_ROOT="${STORAGE_ROOT}/runs/paper_visuals/sn7_updater_20260726"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_paper_visuals_20260726"
POLL_SECONDS="${POLL_SECONDS:-120}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
while [[ ! -s "${BAN_ROOT}/full_weight5_three_seed_20260726.json" ]] || \
  [[ ! -s "${BAN_RUN}/best.pt" ]] || \
  [[ ! -s "${CHANGE_RUN}/val_audit/predicted_change_masks.npz" ]]; do
  echo "Waiting for formal BAN aggregate and auditable ChangeMamba masks."
  sleep "${POLL_SECONDS}"
done

BAN_AUDIT="${BAN_RUN}/val_audit"
if [[ ! -s "${BAN_AUDIT}/predicted_change_masks.npz" ]]; then
  BAN_AUDIT="${BAN_RUN}/val_visual_audit"
  if [[ ! -s "${BAN_AUDIT}/predicted_change_masks.npz" ]]; then
    CUDA_VISIBLE_DEVICES="${VISUAL_GPU:-3}" "${PYTHON}" \
      scripts/evaluate_sn7_opencd_ban.py \
      "${BAN_RUN}/best.pt" "${MANIFEST}" "${OPENCD_ROOT}" \
      "${OPENCD_CONFIG}" "${BAN_AUDIT}" \
      --split val --device cuda:0 --batch-size 8 --workers 8 \
      > "${LOG_ROOT}/ban_seed20260725_visual_audit.log" 2>&1
  fi
fi

if [[ ! -e "${VISUAL_ROOT}/qualitative" ]]; then
  "${PYTHON}" scripts/render_sn7_updater_qualitative.py \
    "${MANIFEST}" "${VISUAL_ROOT}/qualitative" \
    --audit "changemamba=${CHANGE_RUN}/val_audit" \
    --audit "ban=${BAN_AUDIT}" \
    --per-edit 3 --failures-per-edit 2 --disagreements-per-edit 2 \
    > "${LOG_ROOT}/render_qualitative.log" 2>&1
fi

if [[ ! -e "${VISUAL_ROOT}/training_curves" ]]; then
  "${PYTHON}" scripts/plot_sn7_updater_training.py \
    "${VISUAL_ROOT}/training_curves" \
    --history "changemamba=${CHANGE_RUN}/history.jsonl" \
    --history "ban=${BAN_RUN}/history.jsonl" \
    > "${LOG_ROOT}/plot_training.log" 2>&1
fi

"${PYTHON}" - "${VISUAL_ROOT}" "${BAN_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": "sn7-paper-visual-bundle-v1",
    "qualitative_summary": str(root / "qualitative/summary.json"),
    "training_summary": str(root / "training_curves/summary.json"),
    "ban_audit": sys.argv[2],
    "individual_unlabeled_panels": True,
    "test_assets_read": False,
}
(root / "bundle_summary.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
