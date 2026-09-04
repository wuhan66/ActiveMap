#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
WAIT_FOR="${STORAGE_ROOT}/runs/agent/muno21_overnight_capacity_matrix_20260802/COMPLETE.json"
WAIT_FAILED="${STORAGE_ROOT}/runs/agent/muno21_overnight_capacity_matrix_20260802/FAILED.json"
STATIC_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/p0_static_selector_baseline_matrix_v2_20260801"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/residual_safe_executable_matrix_20260802"
ARTIFACT_ROOT="${STORAGE_ROOT}/artifacts/paper_evidence/sn7_unified_executable_controller_matrix_20260802"
GPUS=(1 2 3 4 5 7)
LABELS=(residual_s20260821 residual_s20260822 residual_s20260823 safe_s20260821 safe_s20260822 safe_s20260823)
TRACES=(
  "${STORAGE_ROOT}/runs/sn7_active_catalog/hybrid_residual_8k_no_tools_fullval_matrix_20260801/seed20260821/evaluation/traces.jsonl"
  "${STORAGE_ROOT}/runs/sn7_active_catalog/hybrid_residual_8k_no_tools_fullval_matrix_20260801/seed20260822/evaluation/traces.jsonl"
  "${STORAGE_ROOT}/runs/sn7_active_catalog/hybrid_residual_8k_no_tools_fullval_matrix_20260801/seed20260823/evaluation/traces.jsonl"
  "${STORAGE_ROOT}/artifacts/paper_evidence/sn7_terminal_only_safe_commit_no_tools_stump_20260801/seed20260821/oof_traces.jsonl"
  "${STORAGE_ROOT}/artifacts/paper_evidence/sn7_terminal_only_safe_commit_no_tools_stump_20260801/seed20260822/oof_traces.jsonl"
  "${STORAGE_ROOT}/artifacts/paper_evidence/sn7_terminal_only_safe_commit_no_tools_stump_20260801/seed20260823/oof_traces.jsonl"
)

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/status" "${RUN_ROOT}/logs"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

for path in "${PYTHON}" "${UPDATER}" "${EPISODES}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
for trace in "${TRACES[@]}"; do
  [[ -s "${trace}" ]] || { echo "missing trace: ${trace}" >&2; exit 4; }
done

write_status() {
  printf '{"label":"%s","gpu":%s,"status":"%s","stage":"%s","split":"val","test_assets_read":false}\n' \
    "$1" "$2" "$3" "$4" >"${RUN_ROOT}/status/$1.json"
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

# Conversion is CPU-only and runs while the preceding six-GPU matrix finishes.
for index in "${!LABELS[@]}"; do
  label="${LABELS[index]}"
  job="${RUN_ROOT}/jobs/${label}"
  mkdir -p "${job}"
  input="${job}/writeback_input.jsonl"
  if [[ ! -s "${input}" ]]; then
    write_status "${label}" "${GPUS[index]}" queued convert
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${TRACES[index]}" "${input}" --split val \
      >"${RUN_ROOT}/logs/${label}_convert.log" 2>&1
  fi
  [[ "$(wc -l <"${input}")" -eq 6369 ]] || { echo "bad support: ${label}" >&2; exit 5; }
done

while [[ ! -s "${WAIT_FOR}" ]]; do
  [[ ! -s "${WAIT_FAILED}" ]] || { echo "upstream failed" >&2; exit 6; }
  sleep 30
done
for gpu in "${GPUS[@]}"; do
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 30
  done
done

run_job() {
  local index="$1" label="${LABELS[$1]}" gpu="${GPUS[$1]}"
  local job="${RUN_ROOT}/jobs/${label}" output="${RUN_ROOT}/jobs/${label}/writeback"
  [[ ! -e "${job}/COMPLETE.json" ]] || return 0
  write_status "${label}" "${gpu}" running writeback
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${job}/writeback_input.jsonl" "${output}" \
    --gpu "${gpu}" --python "${PYTHON}" --image-size 512 --threshold 0.5 \
    --protocol-name sn7-residual-safe-identical-support-v1 --split val \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    >"${RUN_ROOT}/logs/${label}.log" 2>&1
  "${PYTHON}" - "${job}" "${label}" "${gpu}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, label, gpu = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
paths = [root / "writeback_input.jsonl", root / "writeback/evaluation/summary.json", root / "writeback/evaluation/writeback.jsonl"]
for path in paths:
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
payload = {"status":"complete", "label":label, "gpu":gpu, "split":"val", "test_assets_read":False,
           "artifacts":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(root / "COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  write_status "${label}" "${gpu}" complete done
}

pids=()
for index in "${!LABELS[@]}"; do
  (set -e; run_job "${index}") & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then status=1; fi; done
if [[ "${status}" -ne 0 ]]; then
  printf '{"status":"failed","split":"val","test_assets_read":false}\n' >"${RUN_ROOT}/FAILED.json"
  exit 7
fi

method_args=()
STATIC_LABELS=(edit_s20260730 edit_s20260731 edit_s20260801 generic_s20260730 generic_s20260731 generic_s20260801 always_stop random cheapest clear_per_cost uncertainty_gate shortlist_oracle_upper_bound)
for label in "${STATIC_LABELS[@]}"; do
  method_args+=(--method "${label}=${STATIC_ROOT}/jobs/${label}/writeback/evaluation/writeback.jsonl")
done
for label in "${LABELS[@]}"; do
  method_args+=(--method "${label}=${RUN_ROOT}/jobs/${label}/writeback/evaluation/writeback.jsonl")
done
"${PYTHON}" scripts/build_sn7_executable_controller_table.py \
  "${ARTIFACT_ROOT}" "${method_args[@]}" \
  --reference always_stop --candidate safe_s20260821 \
  --repetitions 10000 --seed 20260802

printf '{"status":"complete","jobs":6,"records_per_job":6369,"split":"val","test_assets_read":false}\n' >"${RUN_ROOT}/COMPLETE.json"
