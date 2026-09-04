#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
GPU="${GPU:-2}"
POLL_SECONDS="${POLL_SECONDS:-30}"
READY_MARKER="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1/dilate4/COMPLETE.json"
POLICY_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1/dilate4"
LOG="${STORAGE_ROOT}/logs/sn7_modern_agent_protocols_gpu${GPU}.log"

cd "${PROJECT_ROOT}"
printf '%s watcher_started gpu=%s policy_root=%s\n' \
  "$(date -Is)" "${GPU}" "${POLICY_ROOT}" >>"${LOG}"
while [[ ! -s "${READY_MARKER}" ]]; do
  sleep "${POLL_SECONDS}"
done

# GPU4 owns the nine dilate4 policy endpoints after state generation. Wait for
# those endpoints to finish before reusing GPU2, otherwise GPU4 can materialize
# its CUDA context after a memory poll and briefly create a sixth active card.
while true; do
  completed="$(
    find "${POLICY_ROOT}" -mindepth 3 -maxdepth 3 -type f \
      -name COMPLETE.json | wc -l
  )"
  failures="$(
    find "${POLICY_ROOT}" -mindepth 3 -maxdepth 3 -type f \
      -name FAILED.json | wc -l
  )"
  ((failures == 0)) || {
    echo "dilate4 policy endpoint failed" >&2
    exit 5
  }
  ((completed >= 9)) && break
  sleep "${POLL_SECONDS}"
done

# Require three consecutive polls where the target is free and at most four
# physical GPUs are active. This prevents the morphology policy lanes on GPU4
# and the modern baseline from briefly exceeding the five-GPU project cap.
idle_polls=0
while ((idle_polls < 3)); do
  mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  used="${gpu_memory[GPU]}"
  active=0
  for memory in "${gpu_memory[@]}"; do
    ((memory >= 512)) && active=$((active + 1))
  done
  if ((used < 512 && active <= 4)); then
    idle_polls=$((idle_polls + 1))
  else
    idle_polls=0
  fi
  sleep "${POLL_SECONDS}"
done

{
  date -Is
  echo "gpu=${GPU} marker=${READY_MARKER}"
  for protocol in geommagent_style sensesearch_style; do
    GPU="${GPU}" bash scripts/run_sn7_modern_agent_protocol.sh "${protocol}" smoke
    "${STORAGE_ROOT}/envs/activemap-agent/bin/python" \
      scripts/assess_sn7_agent_protocol_run.py \
      "${STORAGE_ROOT}/runs/sn7_active_catalog/${protocol}_qwen_seed20260718_smoke_v3" \
      --protocol "${protocol}" --expected-count 2 --min-valid-action-rate 0.95
  done
  for protocol in geommagent_style sensesearch_style; do
    GPU="${GPU}" bash scripts/run_sn7_modern_agent_protocol.sh "${protocol}" full
    "${STORAGE_ROOT}/envs/activemap-agent/bin/python" \
      scripts/assess_sn7_agent_protocol_run.py \
      "${STORAGE_ROOT}/runs/sn7_active_catalog/${protocol}_qwen_seed20260718_full_v3" \
      --protocol "${protocol}" --expected-count 512 --min-valid-action-rate 0.95
  done

  sft="${STORAGE_ROOT}/runs/sn7_active_catalog/closed_loop_sft_seed20260718_n512/evaluation/traces.jsonl"
  react="${STORAGE_ROOT}/runs/sn7_active_catalog/react_style_qwen_seed20260718_n512_v2/evaluation/traces.jsonl"
  geommagent="${STORAGE_ROOT}/runs/sn7_active_catalog/geommagent_style_qwen_seed20260718_full_v3/evaluation/traces.jsonl"
  sensesearch="${STORAGE_ROOT}/runs/sn7_active_catalog/sensesearch_style_qwen_seed20260718_full_v3/evaluation/traces.jsonl"
  comparison_root="${STORAGE_ROOT}/runs/sn7_active_catalog/modern_agent_protocol_controls_n512_v3"
  mkdir -p "${comparison_root}"
  for path in "${sft}" "${react}" "${geommagent}" "${sensesearch}"; do
    [[ -s "${path}" ]] || { echo "missing paired trace: ${path}" >&2; exit 6; }
  done
  "${STORAGE_ROOT}/envs/activemap-agent/bin/python" \
    scripts/compare_active_catalog_closed_loop.py \
    "${comparison_root}/geommagent_paired.json" \
    --candidate geommagent --reference sft \
    --records "sft=${sft}" --records "react=${react}" \
    --records "geommagent=${geommagent}" --records "sensesearch=${sensesearch}" \
    --repetitions 5000 --seed 20260730
  "${STORAGE_ROOT}/envs/activemap-agent/bin/python" \
    scripts/compare_active_catalog_closed_loop.py \
    "${comparison_root}/sensesearch_paired.json" \
    --candidate sensesearch --reference sft \
    --records "sft=${sft}" --records "react=${react}" \
    --records "geommagent=${geommagent}" --records "sensesearch=${sensesearch}" \
    --repetitions 5000 --seed 20260730
  date -Is
} >>"${LOG}" 2>&1
