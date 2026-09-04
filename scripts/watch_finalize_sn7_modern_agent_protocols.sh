#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
POLL_SECONDS="${POLL_SECONDS:-30}"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
VERSION="${MODERN_AGENT_PROTOCOL_VERSION:-v3}"
LOG="${STORAGE_ROOT}/logs/sn7_modern_agent_protocols_finalize_${VERSION}.log"
PROMOTION="${RUN}/modern_agent_protocol_controls_n512_${VERSION}/PROTOCOL_PROMOTION.json"

declare -A roots=(
  [geommagent_style]="${RUN}/geommagent_style_qwen_seed20260718_full_${VERSION}"
  [sensesearch_style]="${RUN}/sensesearch_style_qwen_seed20260718_full_${VERSION}"
)

run_completed() {
  local state="$1/run_state.json"
  [[ -s "${state}" ]] || return 1
  "${PYTHON}" -c '
import json
import sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if state.get("status") == "completed" and state.get("returncode") == 0 else 1)
' "${state}"
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
printf '%s finalizer_started version=%s\n' "$(date -Is)" "${VERSION}" >>"${LOG}"

assessment_failed=0
for protocol in geommagent_style sensesearch_style; do
  while ! run_completed "${roots[${protocol}]}"; do
    state="${roots[${protocol}]}/run_state.json"
    if [[ -s "${state}" ]] && grep -q '"status": "failed"' "${state}"; then
      printf '%s failed_run protocol=%s state=%s\n' \
        "$(date -Is)" "${protocol}" "${state}" >>"${LOG}"
      exit 5
    fi
    sleep "${POLL_SECONDS}"
  done
  if ! "${PYTHON}" scripts/assess_sn7_agent_protocol_run.py \
    "${roots[${protocol}]}" --protocol "${protocol}" \
    --expected-count 512 --min-valid-action-rate 0.95 \
    --replace-existing >>"${LOG}" 2>&1; then
    assessment_failed=1
  fi
done

sft="${RUN}/closed_loop_sft_seed20260718_n512/evaluation/traces.jsonl"
react="${RUN}/react_style_qwen_seed20260718_n512_v2/evaluation/traces.jsonl"
geommagent="${roots[geommagent_style]}/evaluation/traces.jsonl"
sensesearch="${roots[sensesearch_style]}/evaluation/traces.jsonl"
comparison_root="${RUN}/modern_agent_protocol_controls_n512_${VERSION}"
mkdir -p "${comparison_root}"
for path in "${sft}" "${react}" "${geommagent}" "${sensesearch}"; do
  [[ -s "${path}" ]] || {
    printf 'missing paired trace: %s\n' "${path}" >&2
    exit 6
  }
done

if [[ ! -s "${comparison_root}/geommagent_paired.json" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${comparison_root}/geommagent_paired.json" \
    --candidate geommagent --reference sft \
    --records "sft=${sft}" --records "react=${react}" \
    --records "geommagent=${geommagent}" --records "sensesearch=${sensesearch}" \
    --repetitions 5000 --seed 20260730 >>"${LOG}" 2>&1
fi
if [[ ! -s "${comparison_root}/sensesearch_paired.json" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${comparison_root}/sensesearch_paired.json" \
    --candidate sensesearch --reference sft \
    --records "sft=${sft}" --records "react=${react}" \
    --records "geommagent=${geommagent}" --records "sensesearch=${sensesearch}" \
    --repetitions 5000 --seed 20260730 >>"${LOG}" 2>&1
fi

PROMOTION="${PROMOTION}" ASSESSMENT_FAILED="${assessment_failed}" \
  "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

promotion = Path(os.environ["PROMOTION"])
failed = bool(int(os.environ["ASSESSMENT_FAILED"]))
promotion.write_text(
    json.dumps(
        {
            "schema_version": "sn7-modern-agent-protocol-promotion-v1",
            "comparison_complete": True,
            "all_protocols_promoted": not failed,
            "minimum_valid_action_rate": 0.95,
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
printf '%s finalizer_complete output=%s\n' \
  "$(date -Is)" "${comparison_root}" >>"${LOG}"
