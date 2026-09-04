#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-3}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/agent/muno21_acquire_all_baseline_hdpi_v1}"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent"
SELECTOR="${STORAGE_ROOT}/runs/selector/muno21_evidence_conservative_v5_seed20260811/best.pt"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
MODEL="/home/wh/hf_models/Qwen3-4B"
EPISODES="${DATA_ROOT}/episodes_train_val_v1.jsonl"
STATES="${DATA_ROOT}/selector_states_v1.jsonl"
ROLLOUTS="${RUN_ROOT}/rollouts"
WRITEBACK="${RUN_ROOT}/writeback/acquire_all"
LOG="${RUN_ROOT}/run.log"

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0
rm -f "${RUN_ROOT}/FAILED.json"
trap 'rc=$?; ((rc == 0)) || printf "{\"status\":\"failed\",\"exit_code\":%d,\"split\":\"val\",\"test_assets_read\":false}\n" "${rc}" >"${RUN_ROOT}/FAILED.json"' EXIT

for path in "${MODEL}" "${STATES}" "${EPISODES}" "${SELECTOR}" "${UPDATER}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 3; }
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
printf '%s run_root=%s gpu=%s\n' "$(date -Is)" "${RUN_ROOT}" "${GPU}" >>"${LOG}"

if [[ ! -s "${ROLLOUTS}/summary.json" ]]; then
  "${PYTHON}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${ROLLOUTS}" \
    --checkpoint "${SELECTOR}" \
    --split val --budgets 1.5,3.0,4.5 --selector-device cpu \
    --methods acquire_all --seed 20260821 >>"${LOG}" 2>&1
fi

# Preserve the five-card cap. The CPU rollout can finish immediately; GPU
# writeback starts only after this card and one global project slot stay free.
idle_polls=0
while ((idle_polls < 6)); do
  mapfile -t memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  active=0
  for used in "${memory[@]}"; do ((used >= 512)) && active=$((active + 1)); done
  if ((memory[GPU] < 512 && active <= 4)); then
    idle_polls=$((idle_polls + 1))
  else
    idle_polls=0
  fi
  sleep 10
done

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
  "${UPDATER}" "${EPISODES}" "${ROLLOUTS}/acquire_all.jsonl" "${WRITEBACK}" \
  --device cuda:0 --split val --image-size 512 --threshold 0.5 \
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${LOG}" 2>&1

RUN_ROOT="${RUN_ROOT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
rollout = root / "rollouts" / "summary.json"
writeback = root / "writeback" / "acquire_all" / "summary.json"
for path in (rollout, writeback):
    if not path.is_file():
        raise FileNotFoundError(path)
record = {
    "schema_version": "muno21-acquire-all-baseline-v1",
    "status": "complete",
    "split": "val",
    "method": "acquire_all",
    "rollout_summary": str(rollout),
    "rollout_sha256": hashlib.sha256(rollout.read_bytes()).hexdigest(),
    "writeback_summary": str(writeback),
    "writeback_sha256": hashlib.sha256(writeback.read_bytes()).hexdigest(),
    "test_assets_read": False,
}
(root / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

trap - EXIT
date -Is >>"${LOG}"
