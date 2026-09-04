#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
POLL_SECONDS="${POLL_SECONDS:-60}"
OUTPUT="${RUN}/controller_validation_diagnostic_n512_seed20260718"
EXPECTED_RECORDS=512

sft="${RUN}/closed_loop_sft_seed20260718_n512/evaluation/traces.jsonl"
react="${RUN}/react_style_qwen_seed20260718_n512_v2/evaluation/traces.jsonl"
rl_lr2p5_kl005="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl005_bal50_n4096_seed20260718_n512/evaluation/traces.jsonl"
rl_lr2p5_kl010="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl010_bal50_n4096_seed20260718_n512/evaluation/traces.jsonl"
rl_lr5_kl010="${RUN}/closed_loop_rl_conservative_lr5e6_kl010_bal50_n4096_seed20260718_n512/evaluation/traces.jsonl"
rl_lr5_bal25="${RUN}/closed_loop_rl_conservative_lr5e6_kl005_bal25_n4096_seed20260718_n512/evaluation/traces.jsonl"
rl_lr2p5_bal25="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl010_bal25_n4096_seed20260718_n512/evaluation/traces.jsonl"
react_paired="${RUN}/react_style_qwen_seed20260718_n512_v2_paired_vs_sft.json"
react_smoke_assessment="${RUN}/react_style_qwen_seed20260718_smoke_v2/smoke_assessment.json"

trace_is_complete() {
  local path="$1"
  local summary="${path%/traces.jsonl}/summary.json"
  [[ -s "${path}" && -s "${summary}" ]] || return 1
  [[ "$(wc -l < "${path}")" -eq "${EXPECTED_RECORDS}" ]]
}

until trace_is_complete "${react}" && [[ -s "${react_paired}" ]]; do
  if [[ -s "${react_smoke_assessment}" ]]; then
    passed="$("${PYTHON}" -c \
      "import json; print(str(json.load(open('${react_smoke_assessment}'))['passed']).lower())")"
    if [[ "${passed}" != "true" ]]; then
      echo "ReAct smoke failed; full controller table is intentionally blocked." >&2
      echo "Assessment: ${react_smoke_assessment}" >&2
      exit 20
    fi
  fi
  sleep "${POLL_SECONDS}"
done

inputs=(
  "${sft}"
  "${rl_lr2p5_kl005}"
  "${rl_lr2p5_kl010}"
  "${rl_lr5_kl010}"
  "${rl_lr5_bal25}"
  "${rl_lr2p5_bal25}"
)
for path in "${inputs[@]}"; do
  until trace_is_complete "${path}"; do
    sleep "${POLL_SECONDS}"
  done
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
"${PYTHON}" scripts/build_sn7_controller_validation_table.py \
  "${OUTPUT}" --reference sft --candidate react \
  --expected-records "${EXPECTED_RECORDS}" --repetitions 2000 --seed 20260718 \
  --method "sft=${sft}" \
  --method "react=${react}" \
  --method "rl_lr2p5_kl005_bal50=${rl_lr2p5_kl005}" \
  --method "rl_lr2p5_kl010_bal50=${rl_lr2p5_kl010}" \
  --method "rl_lr5_kl010_bal50=${rl_lr5_kl010}" \
  --method "rl_lr5_kl005_bal25=${rl_lr5_bal25}" \
  --method "rl_lr2p5_kl010_bal25=${rl_lr2p5_bal25}"
