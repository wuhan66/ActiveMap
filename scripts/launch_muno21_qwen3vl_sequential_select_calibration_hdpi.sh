#!/usr/bin/env bash
set -euo pipefail

# Overnight calibration matrix for the only unresolved controller decision:
# whether evidence is worth acquiring. The primary 25% target run is launched
# separately. This script adds the natural and prevalence-matched controls, then
# hands each completed adapter to the same generated-action validation gate.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train/selector_balanced_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/val/selector_sft.jsonl}"
SEED="${SEED:-20260862}"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

launch_variant() {
  local gpu="$1" eval_gpu="$2" name="$3" target="$4"
  local run="${STORE}/runs/agent/${name}"
  local train_log="${STORE}/logs/${name}.log"
  local eval_output="${STORE}/runs/agent/${name}_eval"
  local eval_log="${STORE}/logs/${name}.eval.queue.log"
  [[ ! -e "${run}" ]] || { echo "Refusing existing run: ${run}" >&2; return 3; }
  [[ ! -e "${eval_output}" ]] || { echo "Refusing existing eval: ${eval_output}" >&2; return 4; }

  args=()
  if [[ "${target}" != "natural" ]]; then
    args+=(--acquire-sampling-target "${target}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    nohup "${PY}" scripts/train_semantic_vlm_sft.py \
      "${MODEL}" "${TRAIN}" "${run}" --eval-jsonl "${VAL}" \
      --epochs 2 --learning-rate 0.0001 --batch-size 1 --gradient-accumulation 16 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 --seed "${SEED}" \
      --logging-steps 5 --eval-steps 50 --save-steps 50 --save-total-limit 4 \
      --early-stopping-patience 3 --dataloader-num-workers 2 \
      --dataloader-persistent-workers "${args[@]}" \
      >"${train_log}" 2>&1 < /dev/null &
  echo "$! GPU${gpu} training ${name}"

  GPU="${eval_gpu}" RUN="${run}" ADAPTER="${run}/final" OUTPUT="${eval_output}" \
    LOG="${STORE}/logs/${name}.eval.log" \
    nohup bash scripts/queue_muno21_qwen3vl_sequential_select_eval_hdpi.sh \
      >"${eval_log}" 2>&1 < /dev/null &
  echo "$! GPU${eval_gpu} evaluation queue ${name}"
}

# The 15% target matches the 58/414 natural validation acquisition prevalence;
# the natural run exposes whether any resampling is actually necessary.
launch_variant 1 3 muno21_qwen3vl4b_sequential_select_cal15_v1_seed20260862 0.15
launch_variant 2 4 muno21_qwen3vl4b_sequential_select_natural_v1_seed20260862 natural

echo "Sequential SELECT calibration jobs launched. The primary 25% run remains on GPU5 with its GPU7 evaluator."
