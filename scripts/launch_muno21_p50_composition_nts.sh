#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORE="${STORE:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${PYTHON:-/home/wh/venvs/activemap/bin/python}"
SOURCE="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
ROOT="${STORE}/runs/selector/muno21_p50_composition_v1"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_SEED="${TRAIN_SEED:-20260811}"

mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/COMPLETE.json" ]] || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
for path in "${PYTHON}" "${SOURCE}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 2; }
done
IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
[[ "${#GPUS[@]}" -eq 4 ]] || { echo "GPU_IDS must contain four devices" >&2; exit 2; }

subset_seeds=(20260812 20260813 20260814 20260815)

run_one() {
  local index="$1"
  local subset_seed="${subset_seeds[$index]}"
  local gpu="${GPUS[$index]}"
  local label="subset${subset_seed}"
  local data_dir="${ROOT}/data/${label}"
  local samples="${data_dir}/selector_states_p0500.jsonl"
  local run="${ROOT}/${label}"
  local config="${ROOT}/${label}.json"
  local log="${ROOT}/logs/${label}.log"
  [[ ! -e "${run}" ]] || { echo "refusing partial run: ${run}" >&2; return 3; }
  {
    echo "$(date -Is) START ${label} train_seed=${TRAIN_SEED} gpu=${gpu}"
    "${PYTHON}" scripts/build_grouped_selector_data_scale.py \
      "${SOURCE}" "${data_dir}" --fractions 0.5 --seed "${subset_seed}"
    "${PYTHON}" - "${config}" "${samples}" "${run}" "${TRAIN_SEED}" "${subset_seed}" <<'PY'
import json
import sys
from pathlib import Path

config = {
    "seed": int(sys.argv[4]),
    "data": {"samples": sys.argv[2]},
    "model": {"hidden_dim": 128, "dropout": 0.10},
    "training": {
        "device": "cuda:0", "epochs": 80, "patience": 12,
        "batch_size": 128, "num_workers": 2, "learning_rate": 0.0003,
        "min_learning_rate": 0.000001, "scheduler_factor": 0.5,
        "scheduler_patience": 3, "target_sampling_power": 1.0,
        "weight_decay": 0.0001, "imitation_loss_weight": 1.0,
        "regret_loss_weight": 0.25, "listwise_loss_weight": 0.50,
        "utility_regression_weight": 0.0, "utility_scale": 1.0,
        "utility_temperature": 0.25, "stop_loss_weight": 0.0,
        "acquire_loss_weight": 1.0, "calibrate_stop_margin": True,
        "checkpoint_metric": "mean_chosen_utility",
        "checkpoint_min_acquire_recall": 0.0,
        "checkpoint_min_mean_utility": 0.000001,
        "grad_clip": 1.0, "resume": False,
    },
    "monitoring": {"tensorboard": True, "pause_poll_seconds": 10},
    "ablation": {
        "name": f"conservative_v5_p50_subset_{sys.argv[5]}",
        "condition_on_hypothesis": True, "drop_hypothesis_groups": [],
        "drop_evidence_groups": [], "drop_state_groups": [],
        "false_edit_penalty": True, "allow_stop": True,
    },
    "output_dir": sys.argv[3],
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + "\n")
PY
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap.cli train-selector \
      "${config}" --output "${run}"
    PROJECT_ROOT="${PROJECT_ROOT}" MUNO21_SELECTOR_STATES="${samples}" \
      ACTIVEMAP_PYTHON="${PYTHON}" bash scripts/evaluate_muno21_evidence_selector.sh "${run}"
    date -Is >"${ROOT}/status/${label}.done"
    echo "$(date -Is) DONE ${label}"
  } >"${log}" 2>&1
}

pids=()
for index in 0 1 2 3; do run_one "${index}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if ((status == 0)); then
  printf '{"status":"complete","schema_version":"muno21-p50-composition-v1","train_seed":20260811,"subset_seeds":[20260812,20260813,20260814,20260815],"split":"validation-only","test_assets_read":false}\n' >"${ROOT}/COMPLETE.json"
else
  printf '{"status":"failed","split":"validation-only","test_assets_read":false}\n' >"${ROOT}/FAILED.json"
  exit 1
fi
