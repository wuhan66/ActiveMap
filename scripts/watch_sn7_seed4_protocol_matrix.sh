#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
SEED="${SN7_PROTOCOL_SEED:-20260720}"
GPU_LIST="${SN7_PROTOCOL_GPUS:-1,2,3,4,5}"
ADAPTER_ROOT="${RUN}/qwen3vl4b_weighted_replication_seed4/seed${SEED}"
ADAPTER="${ADAPTER_ROOT}/final"
STATE="${ADAPTER_ROOT}/run_state.json"
OUTPUT="${RUN}/modern_agent_protocol_controls_seed${SEED}_n512_v3"
LOG="${STORAGE_ROOT}/logs/sn7_protocol_seed${SEED}_posttrain.log"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
exec >>"${LOG}" 2>&1
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 5 ]]; then
  echo "SN7_PROTOCOL_GPUS must contain five physical GPU ids" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  case "${gpu}" in
    0|6) echo "GPU${gpu} is reserved and cannot be used" >&2; exit 2 ;;
    1|2|3|4|5|7) ;;
    *) echo "unsupported physical GPU id: ${gpu}" >&2; exit 2 ;;
  esac
done

while true; do
  if [[ -s "${STATE}" ]]; then
    status="$("${PYTHON}" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status","unknown"))' \
      "${STATE}")"
    case "${status}" in
      completed) break ;;
      failed)
        printf '{"status":"failed","reason":"sft_failed","test_assets_read":false}\n' \
          >"${OUTPUT}/FAILED.json"
        exit 4
        ;;
    esac
  fi
  sleep "${POLL_SECONDS}"
done
[[ -s "${ADAPTER}/adapter_config.json" ]] || exit 5

export SEED ADAPTER
export DIRECT_PROTOCOL_VERSION=v2
export REACT_PROTOCOL_VERSION=v2
export PLAN_EXECUTE_PROTOCOL_VERSION=v2
export MODERN_AGENT_PROTOCOL_VERSION=v3

run_if_missing() {
  local summary="$1"
  local gpu="$2"
  shift 2
  if [[ ! -s "${summary}" ]]; then
    GPU="${gpu}" "$@"
  fi
}

direct="${RUN}/closed_loop_sft_seed${SEED}_n512_v2"
react="${RUN}/react_style_qwen_seed${SEED}_n512_v2"
plan="${RUN}/plan_execute_qwen_seed${SEED}_n512_v2"
geo="${RUN}/geommagent_style_qwen_seed${SEED}_full_v3"
sense="${RUN}/sensesearch_style_qwen_seed${SEED}_full_v3"

declare -a pids=()
declare -a labels=()
run_if_missing "${direct}/evaluation/summary.json" "${GPUS[0]}" \
  bash scripts/run_sn7_direct_closed_loop_replication.sh &
pids+=("$!"); labels+=("direct")
run_if_missing "${react}/evaluation/summary.json" "${GPUS[1]}" \
  bash scripts/run_sn7_react_baseline.sh full &
pids+=("$!"); labels+=("react")
run_if_missing "${plan}/evaluation/summary.json" "${GPUS[2]}" \
  bash scripts/run_sn7_plan_execute_replication.sh &
pids+=("$!"); labels+=("plan")
run_if_missing "${geo}/evaluation/summary.json" "${GPUS[3]}" \
  bash scripts/run_sn7_modern_agent_protocol.sh geommagent_style full &
pids+=("$!"); labels+=("geommagent")
run_if_missing "${sense}/evaluation/summary.json" "${GPUS[4]}" \
  bash scripts/run_sn7_modern_agent_protocol.sh sensesearch_style full &
pids+=("$!"); labels+=("sensesearch")

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${labels[$index]} evaluation failed" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf '{"status":"failed","reason":"protocol_child_failed","test_assets_read":false}\n' \
    >"${OUTPUT}/FAILED.json"
  exit 6
fi

assessment_failed=0
for specification in \
  "geommagent_style=${geo}" \
  "sensesearch_style=${sense}"; do
  protocol="${specification%%=*}"
  root="${specification#*=}"
  if ! "${PYTHON}" scripts/assess_sn7_agent_protocol_run.py \
    "${root}" --protocol "${protocol}" --expected-count 512 \
    --min-valid-action-rate 0.95 --replace-existing; then
    assessment_failed=1
  fi
done

for specification in \
  "react=${react}" \
  "plan=${plan}" \
  "geommagent=${geo}" \
  "sensesearch=${sense}"; do
  method="${specification%%=*}"
  root="${specification#*=}"
  paired="${OUTPUT}/${method}_minus_direct.json"
  if [[ ! -s "${paired}" ]]; then
    "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
      "${paired}" --candidate "${method}" --reference direct \
      --records "direct=${direct}/evaluation/traces.jsonl" \
      --records "${method}=${root}/evaluation/traces.jsonl" \
      --repetitions 5000 --seed "${SEED}"
  fi
done

SEED="${SEED}" GPU_LIST="${GPU_LIST}" OUTPUT="${OUTPUT}" \
ASSESSMENT_FAILED="${assessment_failed}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT"])
record = {
    "schema_version": "sn7-protocol-seed-posttrain-v1",
    "status": "complete",
    "seed": int(os.environ["SEED"]),
    "physical_gpus": [int(value) for value in os.environ["GPU_LIST"].split(",")],
    "strict_protocol_gate_passed": not bool(int(os.environ["ASSESSMENT_FAILED"])),
    "comparisons": sorted(str(path) for path in output.glob("*_minus_direct.json")),
    "test_assets_read": False,
}
(output / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY
