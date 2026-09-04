#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
ENV_PREFIX="${STORAGE_ROOT}/envs/activemap-opencd"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
OPENCD_ROOT="${STORAGE_ROOT}/external/open_cd"
CONFIG="${OPENCD_ROOT}/configs/ban/ban_vit-b16-clip_mit-b0_512x512_40k_levircd.py"
MODEL_ROOT="${STORAGE_ROOT}/models/opencd_ban"
CLIP="${MODEL_ROOT}/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
SIDE="${MODEL_ROOT}/mit_b0_20220624-7e0fe6dd.pth"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
LOG_ROOT="${STORAGE_ROOT}/logs"
RUNTIME_AUDIT="${LOG_ROOT}/opencd_setup_20260726/runtime_audit.json"
PIP_CHECK="${LOG_ROOT}/opencd_setup_20260726/pip_check.txt"
UPSTREAM_MARKER="${BAN_UPSTREAM_MARKER:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_belief_ablation_v1/severity4/SUMMARY_COMPLETE.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU="${BAN_GATE_GPU:-1}"
LOCK_FILE="${RUN_ROOT}/.gate.lock"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "BAN gate watcher is already active."
  exit 0
fi

while [[ ! -s "${RUNTIME_AUDIT}" ]] || \
  ! grep -qx "No broken requirements found." "${PIP_CHECK}" 2>/dev/null; do
  echo "Waiting for clean Open-CD runtime and pip audit."
  sleep "${POLL_SECONDS}"
done
bash scripts/setup_opencd_ban_pretrained.sh \
  > "${LOG_ROOT}/opencd_ban_pretrained_20260726.log" 2>&1

while [[ -n "${UPSTREAM_MARKER}" && ! -s "${UPSTREAM_MARKER}" ]]; do
  echo "Waiting for upstream experiment: ${UPSTREAM_MARKER}"
  sleep "${POLL_SECONDS}"
done

gpu_is_idle() {
  local memory
  local utilization
  IFS=, read -r memory utilization < <(
    nvidia-smi --id="${GPU}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  memory="${memory// /}"
  utilization="${utilization// /}"
  [[ "${memory}" -lt 1024 && "${utilization}" -lt 15 ]]
}

wait_for_gpu() {
  until gpu_is_idle; do
    echo "GPU ${GPU} is occupied; waiting ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
}

SMOKE="${RUN_ROOT}/smoke_two_sample_20260726"
OVERFIT="${RUN_ROOT}/overfit20_seed20260725_20260726"
if [[ ! -s "${SMOKE}/summary.json" ]]; then
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" "${ENV_PREFIX}/bin/python" \
    scripts/train_sn7_opencd_ban.py \
    "${MANIFEST}" "${OPENCD_ROOT}" "${CONFIG}" "${SMOKE}" \
    --clip-checkpoint "${CLIP}" \
    --side-checkpoint "${SIDE}" \
    --device cuda:0 --image-size 128 --batch-size 2 --workers 0 \
    --epochs 1 --train-limit 2 --overfit --seed 20260725 \
    > "${LOG_ROOT}/opencd_ban_smoke_20260726.log" 2>&1
fi
if [[ ! -s "${OVERFIT}/summary.json" ]]; then
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" "${ENV_PREFIX}/bin/python" \
    scripts/train_sn7_opencd_ban.py \
    "${MANIFEST}" "${OPENCD_ROOT}" "${CONFIG}" "${OVERFIT}" \
    --clip-checkpoint "${CLIP}" \
    --side-checkpoint "${SIDE}" \
    --device cuda:0 --image-size 128 --batch-size 4 --workers 4 \
    --epochs 20 --limit-per-edit 5 --overfit --seed 20260725 \
    > "${LOG_ROOT}/opencd_ban_overfit20_20260726.log" 2>&1
fi

"${ENV_PREFIX}/bin/python" - "${SMOKE}" "${OVERFIT}" <<'PY'
import json
import math
import sys
from pathlib import Path

smoke = Path(sys.argv[1])
overfit = Path(sys.argv[2])
smoke_history = [
    json.loads(line)
    for line in (smoke / "history.jsonl").read_text().splitlines()
    if line.strip()
]
overfit_history = [
    json.loads(line)
    for line in (overfit / "history.jsonl").read_text().splitlines()
    if line.strip()
]
losses = [row["train_loss"] for row in smoke_history + overfit_history]
best_delta = max(row["val"]["map_iou_delta"] for row in overfit_history)
payload = {
    "schema_version": "sn7-opencd-ban-gate-v1",
    "smoke_completed": len(smoke_history) == 1,
    "all_losses_finite": all(math.isfinite(value) for value in losses),
    "overfit_epoch_count": len(overfit_history),
    "overfit_best_map_iou_delta": best_delta,
    "passed": (
        len(smoke_history) == 1
        and all(math.isfinite(value) for value in losses)
        and len(overfit_history) == 20
        and best_delta > 0.0
    ),
    "test_assets_read": False,
}
path = overfit.parent / "gate_summary_20260726.json"
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
print(json.dumps(payload, indent=2))
if not payload["passed"]:
    raise SystemExit(1)
PY
