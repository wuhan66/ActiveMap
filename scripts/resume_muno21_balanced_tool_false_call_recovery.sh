#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: resume_muno21_balanced_tool_false_call_recovery.sh GPU SEED SELECTOR_SEED [CHECKPOINT]}"
SEED="${2:?usage: resume_muno21_balanced_tool_false_call_recovery.sh GPU SEED SELECTOR_SEED [CHECKPOINT]}"
SELECTOR_SEED="${3:?usage: resume_muno21_balanced_tool_false_call_recovery.sh GPU SEED SELECTOR_SEED [CHECKPOINT]}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
MODEL="${ACTIVEMAP_MODEL_ROOT:-/home/wh/hf_models}/Qwen3-4B"
DATA="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
RUN="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${SEED}"
DECISION="${RUN}/evaluation/selection/static_checkpoint_decision.json"
CONTROL="${RUN}/control/false_call_recovery_v1"
LOG="${RUN}/false_call_recovery_v1.log"

cd "${PROJECT_ROOT}"
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "${GPU}"
for path in \
  "${MODEL}/config.json" \
  "${DATA}/train/sft_composed.jsonl" \
  "${DATA}/val/sft_composed.jsonl" \
  "${DECISION}"; do
  [[ -s "${path}" ]] || { echo "missing recovery input: ${path}" >&2; exit 3; }
done
LATEST_CHECKPOINT="$(
  find "${RUN}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -1
)"
RESUME_LABEL="${4:-${LATEST_CHECKPOINT}}"
[[ "${RESUME_LABEL}" =~ ^checkpoint-[1-9][0-9]*$ ]] || {
  echo "invalid recovery checkpoint: ${RESUME_LABEL}" >&2
  exit 3
}
[[ "${RESUME_LABEL}" == "${LATEST_CHECKPOINT}" ]] || {
  echo "recovery must resume the latest optimizer checkpoint ${LATEST_CHECKPOINT}" >&2
  exit 3
}
RESUME="${RUN}/checkpoints/${RESUME_LABEL}"
for path in "${RESUME}/adapter_config.json" "${RESUME}/trainer_state.json"; do
  [[ -s "${path}" ]] || { echo "missing recovery checkpoint input: ${path}" >&2; exit 3; }
done
[[ ! -s "${RUN}/evaluation/selection/promoted_adapter.json" ]] || {
  echo "refusing recovery of an already promoted run" >&2
  exit 4
}
if [[ -s "${RUN}/train.pid" ]] && kill -0 "$(cat "${RUN}/train.pid")" 2>/dev/null; then
  echo "training is already active for seed ${SEED}" >&2
  exit 4
fi

mkdir -p "${CONTROL}"
DECISION="${DECISION}" CONTROL="${CONTROL}" SEED="${SEED}" \
RESUME_LABEL="${RESUME_LABEL}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

decision_path = Path(os.environ["DECISION"])
control = Path(os.environ["CONTROL"])
decision = json.loads(decision_path.read_text(encoding="utf-8"))
if decision.get("selection_passed") is not False:
    raise SystemExit("recovery requires a failed checkpoint selection")
labels = {row["label"] for row in decision["checkpoints"]}
if os.environ["RESUME_LABEL"] not in labels:
    raise SystemExit("recovery checkpoint is absent from the failed selection audit")
failed = {
    gate
    for checkpoint in decision["checkpoints"]
    for gate in checkpoint.get("failed_gates", [])
}
if failed != {"false_call_rate"}:
    raise SystemExit(f"recovery permits only false-call gate failures, got {failed}")
snapshot = control / "pre_recovery_checkpoint_decision.json"
snapshot.write_bytes(decision_path.read_bytes())
record = {
    "schema_version": "muno21-balanced-tool-false-call-recovery-v1",
    "seed": int(os.environ["SEED"]),
    "resume_checkpoint": os.environ["RESUME_LABEL"],
    "reason": "eval-loss early stopping preceded the false-call safety optimum",
    "controlled_change": {
        "early_stopping_patience": [2, 0],
        "maximum_training_epochs": [4, 4],
        "training_data_changed": False,
        "optimizer_state_resumed": True,
    },
    "pre_recovery_decision_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "split": "val",
    "test_assets_read": False,
}
(control / "protocol.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

nohup env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  ACTIVEMAP_TRAINING_SLOT_HELD=1 \
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -u scripts/train_agent_sft.py \
  "${MODEL}" \
  "${DATA}/train/sft_composed.jsonl" \
  "${RUN}" \
  --eval-jsonl "${DATA}/val/sft_composed.jsonl" \
  --epochs 4 \
  --learning-rate 0.0002 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-length 2048 \
  --logging-steps 5 \
  --eval-steps 100 \
  --save-steps 100 \
  --save-total-limit 6 \
  --early-stopping-patience 0 \
  --early-stopping-threshold 0.0005 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --seed "${SEED}" \
  --resume-from-checkpoint "${RESUME}" \
  >"${LOG}" 2>&1 &
train_pid=$!
printf '%s\n' "${train_pid}" >"${RUN}/train.pid"
printf '%s\n' "${train_pid}" >"${CONTROL}/train.pid"

setsid -f bash scripts/watch_muno21_balanced_tool_posttrain.sh \
  "${GPU}" "${SEED}" "${SELECTOR_SEED}" </dev/null >/dev/null 2>&1

echo "started false-call recovery seed=${SEED} gpu=${GPU} pid=${train_pid}"
