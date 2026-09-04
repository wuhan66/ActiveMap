#!/usr/bin/env bash
set -euo pipefail

# Validation-only reviewer closure. New outputs are never mixed with frozen test artifacts.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/reviewer_closure_20260814}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl}"
CHECKPOINT="${CHECKPOINT:-${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt}"

test -f "${EPISODES}"
test -f "${CHECKPOINT}"
test ! -e "${RUN_ROOT}/online_smoke_gpu0"
test ! -e "${RUN_ROOT}/updater_latency_gpu1.json"
mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 nohup "${PYTHON}" scripts/evaluate_online_persistent_map_maintenance.py \
  "${EPISODES}" "${CHECKPOINT}" "${RUN_ROOT}/online_smoke_gpu0" \
  --split val --device cuda:0 --image-size 512 --minimum-chain-length 2 \
  --safe-confidence-threshold 0.7 --safe-replay-iou-threshold 0.99 --max-chains 10 \
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
  >"${RUN_ROOT}/logs/online_smoke_gpu0.log" 2>&1 &
echo $! >"${RUN_ROOT}/online_smoke_gpu0.pid"

CUDA_VISIBLE_DEVICES=1 nohup "${PYTHON}" scripts/benchmark_updater_latency.py \
  "${EPISODES}" "${CHECKPOINT}" "${RUN_ROOT}/updater_latency_gpu1.json" \
  --device cuda:0 --image-size 512 --batch-size 8 --warmup 20 --repetitions 100 \
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
  >"${RUN_ROOT}/logs/updater_latency_gpu1.log" 2>&1 &
echo $! >"${RUN_ROOT}/updater_latency_gpu1.pid"

"${PYTHON}" scripts/audit_selection_safe_commit_2x2.py \
  "${RUN_ROOT}/selection_safe_commit_2x2" \
  --no-extra-policy notool --learned-policy benefit \
  --confidence-threshold 0.7 --replay-iou-threshold 0.99 --bootstrap-repetitions 5000 \
  --record "20260730:notool=${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/seed20260730/notool/writeback/evaluation/writeback.jsonl" \
  --record "20260730:benefit=${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/seed20260730/benefit/writeback/evaluation/writeback.jsonl" \
  --record "20260731:notool=${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/seed20260731/notool/writeback/evaluation/writeback.jsonl" \
  --record "20260731:benefit=${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/seed20260731/benefit/writeback/evaluation/writeback.jsonl" \
  --record "20260801:notool=${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/seed20260801/notool/writeback/evaluation/writeback.jsonl" \
  --record "20260801:benefit=${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/seed20260801/benefit/writeback/evaluation/writeback.jsonl" \
  >"${RUN_ROOT}/logs/selection_safe_commit_2x2.log" 2>&1

echo "reviewer-closure smoke launched under ${RUN_ROOT}"
