#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2}"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent"
SELECTOR_ROOT="${STORAGE_ROOT}/runs/selector"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
MODEL="/home/wh/hf_models/Qwen3-4B"
EPISODES="${DATA_ROOT}/episodes_train_val_v1.jsonl"
STATES="${DATA_ROOT}/selector_states_v1.jsonl"
ROLLOUTS="${RUN_ROOT}/rollouts"
LOG="${RUN_ROOT}/run.log"
LOCK="${RUN_ROOT}/.lock"

edit_checkpoints=(
  "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt"
  "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt"
  "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt"
)
generic_checkpoints=(
  "${SELECTOR_ROOT}/muno21_evidence_generic_v5_seed20260811/best.pt"
  "${SELECTOR_ROOT}/muno21_evidence_generic_v5_seed20260812/best.pt"
  "${SELECTOR_ROOT}/muno21_evidence_generic_v5_seed20260813/best.pt"
)

mkdir -p "${RUN_ROOT}"
exec 9>"${LOCK}"
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

required=("${MODEL}" "${STATES}" "${EPISODES}" "${UPDATER}")
required+=("${edit_checkpoints[@]}" "${generic_checkpoints[@]}")
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 3; }
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
{
  date -Is
  echo "run_root=${RUN_ROOT} gpu_a=${GPU_A} gpu_b=${GPU_B}"
} >>"${LOG}"

if [[ ! -s "${ROLLOUTS}/summary.json" ]]; then
  checkpoint_args=()
  generic_args=()
  for path in "${edit_checkpoints[@]}"; do checkpoint_args+=(--checkpoint "${path}"); done
  for path in "${generic_checkpoints[@]}"; do
    generic_args+=(--generic-checkpoint "${path}")
  done
  "${PYTHON}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${ROLLOUTS}" \
    "${checkpoint_args[@]}" "${generic_args[@]}" \
    --split val --budgets 1.5,3.0,4.5 --selector-device cpu \
    --methods oracle,generic_selector,edit_conditioned_selector \
    --seed 20260821 >>"${LOG}" 2>&1
fi

run_writeback() {
  local method="$1"
  local gpu="$2"
  local output="${RUN_ROOT}/writeback/${method}"
  [[ -s "${output}/summary.json" ]] && return 0
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${ROLLOUTS}/${method}.jsonl" "${output}" \
    --device cuda:0 --split val --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${LOG}" 2>&1
}

run_writeback generic_selector "${GPU_A}" &
pid_generic=$!
run_writeback edit_conditioned_selector "${GPU_B}" &
pid_edit=$!
wait "${pid_generic}"
wait "${pid_edit}"
run_writeback oracle "${GPU_A}"

RUN_ROOT="${RUN_ROOT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
outputs = {}
for method in ("generic_selector", "edit_conditioned_selector", "oracle"):
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
            "schema_version": "muno21-policy-baselines-hdpi-v2",
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
