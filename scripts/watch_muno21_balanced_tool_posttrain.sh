#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: watch_muno21_balanced_tool_posttrain.sh GPU AGENT_SEED SELECTOR_SEED}"
SEED="${2:?usage: watch_muno21_balanced_tool_posttrain.sh GPU AGENT_SEED SELECTOR_SEED}"
SELECTOR_SEED="${3:?usage: watch_muno21_balanced_tool_posttrain.sh GPU AGENT_SEED SELECTOR_SEED}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_FAMILY="muno21_qwen3_4b_balanced_tool_sft"
RUN_DIR="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${SEED}"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
ROLLOUT_ROOT="${MUNO21_AGENT_ROLLOUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v10_balanced_tool_seed${SEED}_val}"
WRITEBACK_ROOT="${MUNO21_WRITEBACK_ROOT:-${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${SEED}/evaluation/writeback}"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
EPISODES="${STORAGE_ROOT}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
CONTROL="${MUNO21_POSTTRAIN_CONTROL:-${RUN_DIR}/evaluation/posttrain_watcher}"
LOG="${CONTROL}/watcher.log"
LOCK="${CONTROL}/.lock"
WAIT_FOR_GPU_FREE="${MUNO21_WAIT_FOR_GPU_FREE:-0}"

mkdir -p "${CONTROL}"
exec 9>"${LOCK}"
flock -n 9 || exit 0
exec >>"${LOG}" 2>&1

echo "[$(date -Is)] waiting for seed=${SEED} gpu=${GPU}"
if [[ -s "${RUN_DIR}/train.pid" ]]; then
  train_pid="$(cat "${RUN_DIR}/train.pid")"
  while kill -0 "${train_pid}" 2>/dev/null; do
    sleep 60
  done
fi

if [[ "${WAIT_FOR_GPU_FREE}" == "1" ]]; then
  while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
    --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    echo "[$(date -Is)] waiting for physical GPU ${GPU}"
    sleep 60
  done
fi

for path in \
  "${RUN_DIR}/final/adapter_config.json" \
  "${RUN_DIR}/eval_metrics.json" \
  "${DATA_ROOT}/val/sft_composed.jsonl" \
  "${UPDATER}" \
  "${EPISODES}"; do
  [[ -s "${path}" ]] || {
    printf '{"status":"failed","reason":"missing_input","path":"%s","test_assets_read":false}\n' \
      "${path}" >"${CONTROL}/FAILED.json"
    exit 3
  }
done

cd "${PROJECT_ROOT}"
export ACTIVEMAP_TRAINING_SLOT_HELD=1
export ACTIVEMAP_SERVER_ENV="${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
export ACTIVEMAP_AGENT_GPU="${GPU}"
export MUNO21_AGENT_GPU="${GPU}"
export MUNO21_AGENT_SEED="${SEED}"
export MUNO21_SELECTOR_SEED="${SELECTOR_SEED}"
export MUNO21_AGENT_RUN_FAMILY="${RUN_FAMILY}"
export MUNO21_AGENT_RUN_DIR="${RUN_DIR}"
export MUNO21_AGENT_DATA_ROOT="${DATA_ROOT}"
export MUNO21_TOOL_DATA="${DATA_ROOT}"
export MUNO21_AGENT_SEEDS="${SEED}"
export MUNO21_SELECTOR_SEEDS="${SELECTOR_SEED}"
export MUNO21_AGENT_ROLLOUT_ROOT="${ROLLOUT_ROOT}"
export MUNO21_MAX_TOOL_CALLS=2
export MUNO21_INCLUDE_HEURISTICS_FIRST_SEED=0
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! bash scripts/evaluate_promote_sparse_tool_sft.sh; then
  SEED="${SEED}" CONTROL="${CONTROL}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

control = Path(os.environ["CONTROL"])
record = {
    "schema_version": "muno21-balanced-tool-posttrain-failure-v1",
    "status": "failed",
    "stage": "checkpoint_promotion",
    "seed": int(os.environ["SEED"]),
    "split": "val",
    "test_assets_read": False,
}
(control / "FAILED.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY
  exit 5
fi
bash scripts/evaluate_agent_three_seeds.sh

ROLLOUT_SUMMARY="${ROLLOUT_ROOT}/seed${SEED}/summary.json" \
ASSET_ROOT_MAP="/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["ROLLOUT_SUMMARY"])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
source, target = os.environ["ASSET_ROOT_MAP"].split("=", 1)
expected_map = {"source": source, "target": target}
if expected_map not in summary["protocol"].get("asset_root_maps", []):
    raise SystemExit(f"missing asset root map in {summary_path}")
forced = [
    row
    for row in summary["results"]
    if row["method"] == "forced_tools" and float(row["budget"]) >= 3.0
]
if not forced or max(float(row["tool_success_rate"]) for row in forced) <= 0.0:
    raise SystemExit(f"forced tool execution never succeeded in {summary_path}")
PY

for method in \
  qwen3_4b_sft \
  qwen3_4b_sft_tools_no_belief \
  qwen3_4b_sft_tool_to_belief; do
  output="${WRITEBACK_ROOT}/${method}"
  if [[ ! -s "${output}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
      "${UPDATER}" "${EPISODES}" "${ROLLOUT_ROOT}/seed${SEED}/${method}.jsonl" \
      "${output}" --device cuda:0 --split val --image-size 512 --threshold 0.5 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  fi
done

safe_method="qwen3_4b_sft_tool_to_belief_safe_delta"
safe_output="${WRITEBACK_ROOT}/${safe_method}"
if [[ ! -s "${safe_output}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" \
    "${ROLLOUT_ROOT}/seed${SEED}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${safe_output}" --device cuda:0 --split val --image-size 512 --threshold 0.5 \
    --add-min-delta-component-pixels 1536 \
    --delete-min-delta-component-pixels 1024 \
    --reshape-min-delta-component-pixels 0 \
    --preserve-largest-delta-component \
    --protocol-name "muno21-operation-conditioned-safe-delta-v1" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
fi

SEED="${SEED}" GPU="${GPU}" ROLLOUT_ROOT="${ROLLOUT_ROOT}" \
WRITEBACK_ROOT="${WRITEBACK_ROOT}" CONTROL="${CONTROL}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

rollout_root = Path(os.environ["ROLLOUT_ROOT"])
writeback_root = Path(os.environ["WRITEBACK_ROOT"])
methods = (
    "qwen3_4b_sft",
    "qwen3_4b_sft_tools_no_belief",
    "qwen3_4b_sft_tool_to_belief",
    "qwen3_4b_sft_tool_to_belief_safe_delta",
)
files = {}
for method in methods:
    path = writeback_root / method / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    files[method] = {
        "summary": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
rollout = rollout_root / f"seed{os.environ['SEED']}" / "summary.json"
if not rollout.is_file():
    raise FileNotFoundError(rollout)
record = {
    "schema_version": "muno21-balanced-tool-posttrain-v1",
    "status": "complete",
    "seed": int(os.environ["SEED"]),
    "gpu": int(os.environ["GPU"]),
    "rollout_summary": str(rollout),
    "rollout_sha256": hashlib.sha256(rollout.read_bytes()).hexdigest(),
    "writebacks": files,
    "test_assets_read": False,
}
(Path(os.environ["CONTROL"]) / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

echo "[$(date -Is)] post-train evaluation complete for seed=${SEED}"
