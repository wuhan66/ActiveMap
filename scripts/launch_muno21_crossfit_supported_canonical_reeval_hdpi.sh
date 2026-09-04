#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate completed v4 reliability pilots under the explicit SELECT
# evidence-ID canonicalization protocol. This is validation-only and never
# mutates the original pilot evaluation receipts.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/val/selector_sft.jsonl}"
LOG_ROOT="${LOG_ROOT:-${STORE}/logs}"
if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPUS <<<"${GPU_LIST}"
else
  GPUS=(1 2 3 4)
fi

RUNS=(
  "muno21_qwen3vl_crossfit_supported_select_unweighted_pilot_seed20260875"
  "muno21_qwen3vl_crossfit_supported_select_weighted_retry_pilot_seed20260875"
  "muno21_qwen3vl_crossfit_supported_weight075_pilot_seed20260875"
  "muno21_qwen3vl_crossfit_supported_weight025_pilot_seed20260875"
)

[[ ${#GPUS[@]} -ge ${#RUNS[@]} ]] || {
  echo "Need at least ${#RUNS[@]} GPUs, got ${#GPUS[@]}" >&2
  exit 2
}

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done
mkdir -p "${LOG_ROOT}"

pids=()
for index in "${!RUNS[@]}"; do
  name="${RUNS[$index]}"
  gpu="${GPUS[$index]}"
  run="${STORE}/runs/agent/${name}"
  output="${STORE}/runs/agent/${name}_eval_canonical_v2"
  log="${LOG_ROOT}/${name}.eval_canonical_v2.log"
  [[ -d "${run}/final" ]] || { echo "Missing checkpoint: ${run}/final" >&2; exit 4; }
  [[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; exit 5; }
  (
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "${PY}" scripts/evaluate_sequential_selector.py \
        "${MODEL}" "${run}/final" "${VAL}" "${output}" --device cuda:0 \
        --expected-records 414 --bootstrap-repetitions 10000 --seed 20260875 \
        >"${log}" 2>&1
  ) &
  pid="$!"
  pids+=("${pid}")
  echo "started ${name} on physical GPU ${gpu}: pid=${pid}"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
