#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ARGOTWEAK_PYTHON:-${STORAGE_ROOT}/envs/argotweak-legacy/bin/python}"
OFFICIAL_ROOT="${STORAGE_ROOT}/external/ArgoTweak_baselines"
CONFIG="${OFFICIAL_ROOT}/projects/configs/argotweak_explainable.py"
ANNOTATIONS="${STORAGE_ROOT}/datasets/argotweak/tbv_balanced_24_8_v1/official_full/val_argotweak_balanced8.pkl"
TRAIN_RUN="${STORAGE_ROOT}/runs/argotweak/full_domain_adaptation_balanced24_v1/four_gpu_seed20260831"
QUEUE_ROOT="${STORAGE_ROOT}/runs/argotweak/posttrain_checkpoint_curve_v1"
LOG_ROOT="${STORAGE_ROOT}/logs/argotweak_posttrain_checkpoint_curve_v1"
GPUS=(1 2 3 4)
LABELS=(official_baseline epoch6 epoch8 epoch10)
CHECKPOINTS=(
  "${OFFICIAL_ROOT}/checkpoints/argotweak_baseline.pth"
  "${TRAIN_RUN}/epoch_6.pth"
  "${TRAIN_RUN}/epoch_8.pth"
  "${TRAIN_RUN}/epoch_10.pth"
)

mkdir -p "${QUEUE_ROOT}" "${LOG_ROOT}"
exec 9>"${QUEUE_ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${QUEUE_ROOT}/MATRIX_COMPLETED" ]] || exit 0

while pgrep -f "run_argotweak_official_train.py.*${TRAIN_RUN}" >/dev/null; do
  sleep 60
done
for checkpoint in "${CHECKPOINTS[@]}"; do
  [[ -s "${checkpoint}" ]] || {
    echo "missing queued checkpoint: ${checkpoint}" >&2
    touch "${QUEUE_ROOT}/MATRIX_FAILED"
    exit 3
  }
done
for gpu in "${GPUS[@]}"; do
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 30
  done
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_eval() {
  local gpu="$1" label="$2" checkpoint="$3"
  local output="${QUEUE_ROOT}/${label}"
  local log="${LOG_ROOT}/${label}.log"
  mkdir -p "${output}"
  if [[ ! -s "${output}/results.pkl" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/run_argotweak_official_test.py \
      --official-root "${OFFICIAL_ROOT}" --config "${CONFIG}" \
      --annotations "${ANNOTATIONS}" --checkpoint "${checkpoint}" \
      --output-dir "${output}" --workers 4 --seed 20260831 \
      >"${log}" 2>&1
  fi
  if [[ ! -s "${output}/atomic_edit_proposals.jsonl" ]]; then
    "${PYTHON}" scripts/export_argotweak_official_proposals.py \
      --results "${output}/results.pkl" --annotations "${ANNOTATIONS}" \
      --output "${output}/atomic_edit_proposals.jsonl" --object-threshold 0.3 \
      >>"${log}" 2>&1
  fi
}

pids=()
for index in 0 1 2 3; do
  run_eval "${GPUS[${index}]}" "${LABELS[${index}]}" "${CHECKPOINTS[${index}]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  touch "${QUEUE_ROOT}/MATRIX_FAILED"
  exit 1
fi

"${PYTHON}" - "${QUEUE_ROOT}" "${ANNOTATIONS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
annotations = Path(sys.argv[2])
rows = []
for label in ("official_baseline", "epoch6", "epoch8", "epoch10"):
    result = root / label / "results.pkl"
    proposals = root / label / "atomic_edit_proposals.jsonl"
    rows.append({
        "label": label,
        "results_bytes": result.stat().st_size,
        "results_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        "proposal_bytes": proposals.stat().st_size,
        "proposal_sha256": hashlib.sha256(proposals.read_bytes()).hexdigest(),
    })
manifest = {
    "schema_version": "argotweak-posttrain-checkpoint-curve-v1",
    "split": "val",
    "annotations": str(annotations),
    "object_threshold": 0.3,
    "rows": rows,
    "test_assets_read": False,
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
touch "${QUEUE_ROOT}/MATRIX_COMPLETED"
