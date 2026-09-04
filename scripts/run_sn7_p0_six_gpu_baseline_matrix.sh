#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/p0_static_selector_baseline_matrix_v2_20260801}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${STORAGE_ROOT}/artifacts/paper_evidence/sn7_p0_static_selector_baseline_matrix_v2_20260801}"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1"
STATES="${DATA}/states_val_step0.jsonl"
EPISODES="${DATA}/episodes_val.jsonl"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt"
SELECTOR_ROOT="${STORAGE_ROOT}/runs/selector"
GPUS=(1 2 3 4 5 7)

LEARNED_LABELS=(
  edit_s20260730 edit_s20260731 edit_s20260801
  generic_s20260730 generic_s20260731 generic_s20260801
)
LEARNED_CHECKPOINTS=(
  "${SELECTOR_ROOT}/sn7_step0_two_stage_v2_seed20260730/edit_utility_seed20260730/best.pt"
  "${SELECTOR_ROOT}/sn7_step0_two_stage_v2_seed20260731/edit_utility_seed20260731/best.pt"
  "${SELECTOR_ROOT}/sn7_step0_two_stage_v2_seed20260801/edit_utility_seed20260801/best.pt"
  "${SELECTOR_ROOT}/sn7_step0_two_stage_v2_seed20260730/generic_utility_seed20260730/best.pt"
  "${SELECTOR_ROOT}/sn7_step0_two_stage_v2_seed20260731/generic_utility_seed20260731/best.pt"
  "${SELECTOR_ROOT}/sn7_step0_two_stage_v2_seed20260801/generic_utility_seed20260801/best.pt"
)
FIXED_LABELS=(
  always_stop random cheapest clear_per_cost uncertainty_gate shortlist_oracle_upper_bound
)

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

required=("${STATES}" "${EPISODES}" "${UPDATER}" "${LEARNED_CHECKPOINTS[@]}")
for path in "${required[@]}"; do
  [[ -s "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 3; }
done

for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" != 0 && "${gpu}" != 6 ]] || { echo "GPU0/6 are forbidden" >&2; exit 4; }
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || { echo "GPU${gpu} occupied by ${pids}" >&2; exit 5; }
done

"${PYTHON}" - "${RUN_ROOT}" "${STATES}" "${EPISODES}" "${UPDATER}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, states, episodes, updater = map(Path, sys.argv[1:])
payload = {
    "schema_version": "sn7-p0-six-gpu-baseline-launch-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "split": "val",
    "test_assets_read": False,
    "gpus": [1, 2, 3, 4, 5, 7],
    "waves": [
        ["edit_s20260730", "edit_s20260731", "edit_s20260801", "generic_s20260730", "generic_s20260731", "generic_s20260801"],
        ["always_stop", "random", "cheapest", "clear_per_cost", "uncertainty_gate", "shortlist_oracle_upper_bound"],
    ],
    "inputs": {
        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in {"states": states, "episodes": episodes, "updater": updater}.items()
    },
    "protocol": {
        "identical_support": True,
        "max_candidates": 16,
        "max_acquisitions": 2,
        "writeback_threshold": 0.5,
        "bootstrap_unit": "AOI",
        "final_bootstrap_repetitions": 10000,
    },
}
(root / "launch_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

write_status() {
  local label="$1" status="$2" gpu="$3" stage="$4" rc="${5:-0}"
  printf '{"label":"%s","status":"%s","gpu":%s,"stage":"%s","exit_code":%s,"split":"val","test_assets_read":false}\n' \
    "${label}" "${status}" "${gpu}" "${stage}" "${rc}" >"${RUN_ROOT}/status/${label}.json"
}

run_job() {
  local label="$1" gpu="$2" checkpoint="${3:-}"
  local job="${RUN_ROOT}/jobs/${label}"
  local closed="${job}/closed_loop"
  local input="${job}/writeback_input.jsonl"
  local writeback="${job}/writeback"
  local log="${RUN_ROOT}/logs/${label}.log"
  if [[ -s "${job}/COMPLETE.json" ]]; then
    return 0
  fi
  [[ ! -e "${job}" ]] || { echo "refusing partial job ${job}" >&2; return 7; }
  mkdir -p "${job}"
  write_status "${label}" running "${gpu}" closed_loop
  (
    set -e
    date -Is
    echo "label=${label} gpu=${gpu} checkpoint=${checkpoint:-none}"
    command=(
      "${PYTHON}" scripts/evaluate_active_catalog_closed_loop_baselines.py
      "${STATES}" "${closed}" --episodes "${EPISODES}"
      --policy "${label}" --device cuda:0 --split val
      --max-candidates 16 --max-acquisitions 2
      --bootstrap-repetitions 500 --seed 20260801
    )
    if [[ -n "${checkpoint}" ]]; then
      command+=(--learned-selector "${label}=${checkpoint}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}"

    write_status "${label}" running "${gpu}" convert
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${closed}/${label}.jsonl" "${input}" --split val

    write_status "${label}" running "${gpu}" writeback
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER}" "${EPISODES}" "${input}" "${writeback}" \
      --gpu "${gpu}" --python "${PYTHON}" --image-size 512 --threshold 0.5 \
      --protocol-name sn7-p0-static-selector-baseline-v1 --split val \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"

    "${PYTHON}" - "${job}" "${label}" "${gpu}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, label, gpu = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
paths = {
    "closed_loop_summary": root / "closed_loop/summary.json",
    "closed_loop_trace": root / f"closed_loop/{label}.jsonl",
    "writeback_summary": root / "writeback/evaluation/summary.json",
    "writeback_trace": root / "writeback/evaluation/writeback.jsonl",
}
for path in paths.values():
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
payload = {
    "status": "complete",
    "label": label,
    "gpu": gpu,
    "split": "val",
    "test_assets_read": False,
    "artifacts": {
        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in paths.items()
    },
}
(root / "COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    write_status "${label}" complete "${gpu}" done
    date -Is
  ) >>"${log}" 2>&1 || {
    rc=$?
    write_status "${label}" failed "${gpu}" failed "${rc}"
    return "${rc}"
  }
}

run_wave_learned() {
  pids=()
  for index in "${!LEARNED_LABELS[@]}"; do
    run_job "${LEARNED_LABELS[index]}" "${GPUS[index]}" "${LEARNED_CHECKPOINTS[index]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

run_wave_fixed() {
  pids=()
  for index in "${!FIXED_LABELS[@]}"; do
    run_job "${FIXED_LABELS[index]}" "${GPUS[index]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

run_wave_learned || {
  printf '{"status":"failed","stage":"learned_wave","split":"val","test_assets_read":false}\n' >"${RUN_ROOT}/FAILED.json"
  exit 11
}
run_wave_fixed || {
  printf '{"status":"failed","stage":"fixed_wave","split":"val","test_assets_read":false}\n' >"${RUN_ROOT}/FAILED.json"
  exit 12
}

method_args=()
for label in "${LEARNED_LABELS[@]}" "${FIXED_LABELS[@]}"; do
  method_args+=(--method "${label}=${RUN_ROOT}/jobs/${label}/writeback/evaluation/writeback.jsonl")
done
"${PYTHON}" scripts/build_sn7_executable_controller_table.py \
  "${ARTIFACT_ROOT}" "${method_args[@]}" \
  --reference always_stop --candidate edit_s20260730 \
  --repetitions 10000 --seed 20260801

"${PYTHON}" - "${RUN_ROOT}" "${ARTIFACT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, artifact = map(Path, sys.argv[1:])
table = artifact / "controller_table.json"
payload = {
    "status": "complete",
    "schema_version": "sn7-p0-six-gpu-baseline-matrix-v1",
    "split": "val",
    "test_assets_read": False,
    "job_count": 12,
    "table": str(table),
    "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
}
(root / "COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
