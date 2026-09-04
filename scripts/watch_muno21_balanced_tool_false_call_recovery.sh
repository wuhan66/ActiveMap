#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: watch_muno21_balanced_tool_false_call_recovery.sh GPU SEED SELECTOR_SEED}"
SEED="${2:?usage: watch_muno21_balanced_tool_false_call_recovery.sh GPU SEED SELECTOR_SEED}"
SELECTOR_SEED="${3:?usage: watch_muno21_balanced_tool_false_call_recovery.sh GPU SEED SELECTOR_SEED}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${SEED}"
DECISION="${RUN}/evaluation/selection/static_checkpoint_decision.json"
PROMOTION="${RUN}/evaluation/selection/promoted_adapter.json"
CONTROL="${RUN}/control/false_call_auto_recovery_watcher"
RECOVERY="${RUN}/control/false_call_recovery_v1/protocol.json"
LOCK="${CONTROL}/.lock"
LOG="${CONTROL}/watcher.log"

mkdir -p "${CONTROL}"
exec 9>"${LOCK}"
flock -n 9 || exit 0
exec >>"${LOG}" 2>&1

echo "[$(date -Is)] waiting for seed=${SEED} initial promotion decision"
if [[ -s "${RUN}/train.pid" ]]; then
  train_pid="$(cat "${RUN}/train.pid")"
  while kill -0 "${train_pid}" 2>/dev/null; do
    sleep 60
  done
fi

while [[ ! -s "${DECISION}" && ! -s "${PROMOTION}" ]]; do
  sleep 60
done
if [[ -s "${PROMOTION}" ]]; then
  printf '{"status":"not_required","reason":"initial_promotion_passed","seed":%s,"test_assets_read":false}\n' \
    "${SEED}" >"${CONTROL}/COMPLETE.json"
  exit 0
fi
if [[ -s "${RECOVERY}" ]]; then
  printf '{"status":"not_required","reason":"recovery_already_registered","seed":%s,"test_assets_read":false}\n' \
    "${SEED}" >"${CONTROL}/COMPLETE.json"
  exit 0
fi

DECISION="${DECISION}" CONTROL="${CONTROL}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

decision = json.load(open(os.environ["DECISION"], encoding="utf-8"))
failed = {
    gate
    for checkpoint in decision["checkpoints"]
    for gate in checkpoint.get("failed_gates", [])
}
record = {
    "eligible": (
        decision.get("selection_passed") is False
        and failed == {"false_call_rate"}
    ),
    "selection_passed": decision.get("selection_passed"),
    "failed_gates": sorted(failed),
    "split": "val",
    "test_assets_read": False,
}
(Path(os.environ["CONTROL"]) / "eligibility.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY
eligible="$("${PYTHON}" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["eligible"]).lower())' "${CONTROL}/eligibility.json")"
if [[ "${eligible}" != "true" ]]; then
  printf '{"status":"not_required","reason":"ineligible_failure_gates","seed":%s,"test_assets_read":false}\n' \
    "${SEED}" >"${CONTROL}/COMPLETE.json"
  exit 0
fi

while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
  sleep 15
done

cd "${PROJECT_ROOT}"
ACTIVEMAP_GPU_IDS="${ACTIVEMAP_GPU_IDS:-0,1,2,3,4}" \
  bash scripts/resume_muno21_balanced_tool_false_call_recovery.sh \
  "${GPU}" "${SEED}" "${SELECTOR_SEED}"

printf '{"status":"launched","reason":"false_call_only_failure","seed":%s,"gpu":%s,"test_assets_read":false}\n' \
  "${SEED}" "${GPU}" >"${CONTROL}/COMPLETE.json"
echo "[$(date -Is)] launched full-budget recovery for seed=${SEED} gpu=${GPU}"
