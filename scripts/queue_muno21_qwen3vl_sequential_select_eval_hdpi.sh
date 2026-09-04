#!/usr/bin/env bash
set -euo pipefail

# Generated-action evaluation for the mixed STOP/ACQUIRE controller. It waits
# for the dedicated SELECT-stage SFT and never accesses frozen test assets.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
RUN="${RUN:-${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_v1_seed20260862}"
ADAPTER="${ADAPTER:-${RUN}/final}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/val/selector_sft.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_v1_seed20260862_eval}"
LOG="${LOG:-${STORE}/logs/muno21_qwen3vl4b_sequential_select_v1_seed20260862.eval.log}"
GPU="${GPU:-7}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT}"
mkdir -p "$(dirname "${LOG}")"
[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing existing evaluation output: ${OUTPUT}" >&2
  exit 2
}
for path in "${PY}" "${MODEL}/config.json" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done
until [[ -s "${ADAPTER}/adapter_model.safetensors" ]]; do
  echo "Waiting for SELECT adapter: ${ADAPTER}"
  sleep "${POLL_SECONDS}"
done

export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/evaluate_sequential_selector.py \
  "${MODEL}" "${ADAPTER}" "${VAL}" "${OUTPUT}" --device cuda:0 \
  --expected-records 414 --bootstrap-repetitions 2000 --seed 20260862 \
  >"${LOG}" 2>&1

echo "Sequential SELECT evaluation completed: ${OUTPUT}"
