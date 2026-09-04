#!/usr/bin/env bash
set -euo pipefail

ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ROOT}/envs/activemap-agent/bin/python"
RUN="${ROOT}/runs/sn7_active_catalog"
DATA="${ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
SFT="${RUN}/closed_loop_sft_seed20260718_n512"
OUTPUT="${RUN}/plan_execute_qwen_seed20260718_n512"
COMPARISON="${RUN}/plan_execute_qwen_seed20260718_n512_paired_vs_sft.json"
INPUT="${RUN}/plan_execute_qwen_seed20260718_n512_writeback_input.jsonl"
SFT_INPUT="${RUN}/closed_loop_sft_seed20260718_n512_writeback_input.jsonl"
WRITEBACK="${RUN}/plan_execute_qwen_seed20260718_n512_writeback"
SFT_WRITEBACK="${RUN}/closed_loop_sft_seed20260718_n512_writeback"
WRITEBACK_COMPARISON="${RUN}/plan_execute_qwen_seed20260718_n512_writeback_vs_sft.json"
VISUALS="${ROOT}/runs/paper_visuals/sft_vs_plan_execute_seed20260718_n512"
MANIFEST="${RUN}/plan_execute_qwen_seed20260718_n512_manifest.json"
UPDATER="${ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
EPISODES="${DATA}/closed_loop_v1/episodes_val.jsonl"
GPU="${GPU:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DEFER_WRITEBACK_UNTIL_SESSION="${DEFER_WRITEBACK_UNTIL_SESSION:-}"

for required in \
  "${SFT}/evaluation/traces.jsonl"; do
  [[ -s "${required}" ]] || {
    echo "SFT prerequisite failed before plan-execute: ${required}" >&2
    exit 5
  }
done

gpu_is_idle() {
  local pids
  pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]]
}
until gpu_is_idle; do sleep "${POLL_SECONDS}"; done

cd "${REPO}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
if [[ ! -s "${OUTPUT}/evaluation/traces.jsonl" ]]; then
  [[ ! -e "${OUTPUT}" ]] || { echo "refusing partial plan-execute output" >&2; exit 3; }
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${ADAPTER}" \
    "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
    "${DATA}/closed_loop_v1/episodes_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${OUTPUT}" --gpu "${GPU}" --seed 20260718 \
    --policy-mode plan_execute --tool-mode none \
    --max-candidates 16 --max-acquisitions 2 --max-new-tokens 96 \
    --bootstrap-repetitions 500 --limit 512 --monitor-interval 5
fi

if [[ ! -s "${COMPARISON}" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${COMPARISON}" --reference sft --repetitions 5000 --seed 20260729 \
    --records "sft=${SFT}/evaluation/traces.jsonl" \
    --records "plan_execute=${OUTPUT}/evaluation/traces.jsonl"
fi
if [[ ! -s "${INPUT}" ]]; then
  "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
    "${OUTPUT}/evaluation/traces.jsonl" "${INPUT}" --split val
fi
if [[ ! -s "${SFT_INPUT}" ]]; then
  "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
    "${SFT}/evaluation/traces.jsonl" "${SFT_INPUT}" --split val
fi
if [[ -n "${DEFER_WRITEBACK_UNTIL_SESSION}" ]]; then
  while tmux has-session -t "${DEFER_WRITEBACK_UNTIL_SESSION}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
fi
until gpu_is_idle; do sleep "${POLL_SECONDS}"; done
if [[ ! -s "${SFT_WRITEBACK}/evaluation/writeback.jsonl" ]]; then
  [[ ! -e "${SFT_WRITEBACK}" ]] || {
    echo "refusing partial SFT writeback" >&2
    exit 6
  }
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${SFT_INPUT}" "${SFT_WRITEBACK}" \
    --gpu "${GPU}" --image-size 512 --threshold 0.5 \
    --protocol-name sn7-sft-n512-vector-writeback-v1 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${ROOT}" \
    --split val --limit 512 --monitor-interval 5
fi
until gpu_is_idle; do sleep "${POLL_SECONDS}"; done
if [[ ! -s "${WRITEBACK}/evaluation/writeback.jsonl" ]]; then
  [[ ! -e "${WRITEBACK}" ]] || { echo "refusing partial plan writeback" >&2; exit 4; }
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${INPUT}" "${WRITEBACK}" \
    --gpu "${GPU}" --image-size 512 --threshold 0.5 \
    --protocol-name sn7-open-loop-plan-execute-writeback-v1 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${ROOT}" \
    --split val --limit 512 --monitor-interval 5
fi
if [[ ! -s "${WRITEBACK_COMPARISON}" ]]; then
  "${PYTHON}" scripts/compare_agent_writebacks.py \
    "${SFT_WRITEBACK}/evaluation/writeback.jsonl" \
    "${WRITEBACK}/evaluation/writeback.jsonl" \
    "${WRITEBACK_COMPARISON}" --bootstrap 5000 --seed 20260729 \
    --group-key aoi_id
fi
if [[ ! -s "${VISUALS}/summary.json" ]]; then
  "${PYTHON}" scripts/render_active_catalog_paired_examples.py \
    "${EPISODES}" "${VISUALS}" --candidate plan_execute \
    --method "sft=${SFT}/evaluation/traces.jsonl,${SFT_WRITEBACK}/evaluation/writeback.jsonl" \
    --method "plan_execute=${OUTPUT}/evaluation/traces.jsonl,${WRITEBACK}/evaluation/writeback.jsonl" \
    --per-edit-success 2 --per-edit-failure 1 --per-edit-disagreement 1 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${ROOT}" \
    --min-visual-fraction 0.01
fi

"${PYTHON}" - "${MANIFEST}" "${COMPARISON}" "${WRITEBACK_COMPARISON}" "${VISUALS}/summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
sources = [pathlib.Path(value) for value in sys.argv[2:]]
payload = {
    "schema_version": "sn7-plan-execute-baseline-manifest-v1",
    "protocol": "single initial visual plan followed by open-loop execution",
    "same_model_adapter_as_sft": True,
    "intermediate_belief_visible_to_planner": False,
    "split": "val",
    "test_assets_read": False,
    "artifacts": {
        path.name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sources
    },
}
output.write_text(json.dumps(payload, indent=2) + "\n")
PY
