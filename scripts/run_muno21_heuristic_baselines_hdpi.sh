#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/agent/muno21_heuristic_baselines_hdpi_v1}"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent"
SELECTOR_ROOT="${STORAGE_ROOT}/runs/selector"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
MODEL="/home/wh/hf_models/Qwen3-4B"
EPISODES="${DATA_ROOT}/episodes_train_val_v1.jsonl"
STATES="${DATA_ROOT}/selector_states_v1.jsonl"
ROLLOUTS="${RUN_ROOT}/rollouts"
LOG="${RUN_ROOT}/run.log"
METHODS=(random cheapest quality_first uncertainty mapex greedy_utility)

checkpoints=(
  "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt"
  "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt"
  "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt"
)

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0
rm -f "${RUN_ROOT}/FAILED.json"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    printf '{"status":"failed","exit_code":%d,"split":"val","test_assets_read":false}\n' \
      "${rc}" >"${RUN_ROOT}/FAILED.json"
  fi
  exit "${rc}"
}
trap record_failure EXIT

required=("${MODEL}" "${STATES}" "${EPISODES}" "${UPDATER}" "${checkpoints[@]}")
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 3; }
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
printf '%s run_root=%s gpu=%s\n' "$(date -Is)" "${RUN_ROOT}" "${GPU}" >>"${LOG}"

if [[ ! -s "${ROLLOUTS}/summary.json" ]]; then
  checkpoint_args=()
  for path in "${checkpoints[@]}"; do checkpoint_args+=(--checkpoint "${path}"); done
  method_csv="$(IFS=,; echo "${METHODS[*]}")"
  "${PYTHON}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${ROLLOUTS}" "${checkpoint_args[@]}" \
    --split val --budgets 1.5,3.0,4.5 --selector-device cpu \
    --methods "${method_csv}" --seed 20260821 >>"${LOG}" 2>&1
fi

# The rollout stage is CPU-only. Enter the GPU stage only when this card is
# free and adding it keeps the project at or below five physical GPUs.
idle_polls=0
while ((idle_polls < 3)); do
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

for method in "${METHODS[@]}"; do
  output="${RUN_ROOT}/writeback/${method}"
  [[ -s "${output}/summary.json" ]] && continue
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${ROLLOUTS}/${method}.jsonl" "${output}" \
    --device cuda:0 --split val --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${LOG}" 2>&1
done

RUN_ROOT="${RUN_ROOT}" METHODS="$(IFS=,; echo "${METHODS[*]}")" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
outputs = {}
for method in os.environ["METHODS"].split(","):
    path = root / "writeback" / method / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    outputs[method] = {
        "summary": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "schema_version": "muno21-heuristic-baselines-hdpi-v1",
            "split": "val",
            "methods": outputs,
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

trap - EXIT
date -Is >>"${LOG}"
