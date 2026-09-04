#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:?Set ACTIVEMAP_AGENT_PYTHON explicitly}"
OVERLAY="${ACTIVEMAP_AGENT_SITE_PACKAGES:-/mnt/mydisk/wh/ActiveMap/envs/agent_peft_overlay}"
GPU="${MUNO21_AGENT_GPU:-3}"
MODEL="${MUNO21_DPO_MODEL:?Set MUNO21_DPO_MODEL to the selected SFT adapter}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/agent_data_v6_anonymized}"
TRAIN_FILE="${MUNO21_DPO_TRAIN_FILE:-${DATA_ROOT}/train/safety_preferences.jsonl}"
EVAL_FILE="${MUNO21_DPO_EVAL_FILE:-${DATA_ROOT}/val/safety_preferences.jsonl}"
RUN_DIR="${MUNO21_DPO_RUN_DIR:?Set MUNO21_DPO_RUN_DIR explicitly}"

for path in \
  "${MODEL}/adapter_model.safetensors" \
  "${MODEL}/adapter_config.json" \
  "${TRAIN_FILE}" \
  "${EVAL_FILE}"; do
  [[ -s "${path}" ]] || { echo "Required input is missing: ${path}" >&2; exit 1; }
done
mkdir -p "${RUN_DIR}/control"
PID_FILE="${RUN_DIR}/train.pid"
if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Agent safety DPO already running: PID $(cat "${PID_FILE}")" >&2
  exit 1
fi
if [[ -f "${RUN_DIR}/control/STOP" ]]; then
  echo "Remove ${RUN_DIR}/control/STOP before launching" >&2
  exit 1
fi
GPU_PIDS="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU ${GPU} has active compute processes: ${GPU_PIDS}" >&2
  exit 1
fi
PYTHONPATH="${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -c "import datasets, peft, trl, transformers" || {
  echo "Agent DPO environment is incomplete; install environments/agent-requirements.txt" >&2
  exit 1
}

cd "${PROJECT_ROOT}"
printf 'physical_gpu=%s\npython=%s\nmodel=%s\ntrain_file=%s\neval_file=%s\n' \
  "${GPU}" "${PYTHON}" "${MODEL}" "${TRAIN_FILE}" "${EVAL_FILE}" \
  > "${RUN_DIR}/launch.txt"
nohup env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  PYTHONPATH="${PROJECT_ROOT}/src:${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/train_agent_dpo.py" \
  "${MODEL}" "${TRAIN_FILE}" "${RUN_DIR}" \
  --eval-jsonl "${EVAL_FILE}" \
  --epochs 1 --learning-rate 1e-5 --batch-size 1 \
  --gradient-accumulation 32 --max-length 2048 --beta 0.1 \
  --logging-steps 5 --eval-steps 25 --save-steps 25 --seed 20260821 \
  > "${RUN_DIR}/train.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
sleep 3
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Agent safety DPO failed to start; inspect ${RUN_DIR}/train.log" >&2
  exit 1
fi
echo "Agent safety DPO started: PID ${pid}, physical GPU ${GPU}, run ${RUN_DIR}"
