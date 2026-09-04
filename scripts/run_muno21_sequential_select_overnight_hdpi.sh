#!/usr/bin/env bash
set -euo pipefail

# Self-contained, validation-only overnight controller pipeline.  It is safe to
# detach with nohup: every stage writes durable logs and a later stage only runs
# after its declared prerequisite is complete.  GPU 0 and GPU 6 are untouched.
#
# Existing jobs expected before this controller is started:
#   - GPU 5: primary SELECT SFT (25% ACQUIRE sampling)
#   - GPU 1: prevalence-matched SELECT SFT (15% ACQUIRE sampling)
#   - GPU 7: primary evaluation watcher
# This controller adds the natural-sampling control on GPU 2, evaluates the two
# newly completed controls on their freed training GPUs, aggregates the three
# predeclared gates, and optionally runs exactly one joint-SFT continuation.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
SELECT_TRAIN="${SELECT_TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train/selector_balanced_sft.jsonl}"
SELECT_VAL="${SELECT_VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/val/selector_sft.jsonl}"
JOINT_TRAIN="${JOINT_TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train/joint_curriculum_sft.jsonl}"
SEED="${SEED:-20260862}"
POLL_SECONDS="${POLL_SECONDS:-90}"
ALLOW_JOINT="${ALLOW_JOINT:-1}"

PRIMARY="muno21_qwen3vl4b_sequential_select_v1_seed${SEED}"
CAL15="muno21_qwen3vl4b_sequential_select_cal15_v1_seed${SEED}"
NATURAL="muno21_qwen3vl4b_sequential_select_natural_v1_seed${SEED}"
AGGREGATE="${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_calibration_v1.json"
PIPELINE_LOG="${STORE}/logs/muno21_qwen3vl4b_sequential_overnight_v1.log"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${SELECT_TRAIN}" "${SELECT_VAL}" "${JOINT_TRAIN}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

timestamp() { date '+%F %T'; }
note() { echo "[$(timestamp)] $*"; }

wait_for_adapter() {
  local run="$1"
  local adapter="${STORE}/runs/agent/${run}/final/adapter_model.safetensors"
  until [[ -s "${adapter}" ]]; do
    note "waiting for adapter: ${run}"
    sleep "${POLL_SECONDS}"
  done
}

wait_for_trainer_exit() {
  local run="$1"
  # Saving the final adapter can precede CUDA teardown.  Avoid sharing the GPU
  # with that trainer before its evaluator is started.
  while pgrep -f "train_semantic_vlm_sft.py.*${run}" >/dev/null; do
    note "waiting for trainer exit: ${run}"
    sleep "${POLL_SECONDS}"
  done
}

run_evaluation() {
  local run="$1" gpu="$2"
  local output="${STORE}/runs/agent/${run}_eval"
  local log="${STORE}/logs/${run}.eval.log"
  [[ -s "${output}/summary.json" ]] && { note "evaluation already complete: ${run}"; return; }
  [[ ! -e "${output}" ]] || { note "existing incomplete evaluation blocks overwrite: ${output}"; return 3; }
  wait_for_adapter "${run}"
  wait_for_trainer_exit "${run}"
  note "evaluating ${run} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "${PY}" scripts/evaluate_sequential_selector.py \
      "${MODEL}" "${STORE}/runs/agent/${run}/final" "${SELECT_VAL}" "${output}" \
      --device cuda:0 --expected-records 414 --bootstrap-repetitions 2000 --seed "${SEED}" \
      >"${log}" 2>&1
  note "evaluation complete: ${run}"
}

launch_natural_train() {
  local run_dir="${STORE}/runs/agent/${NATURAL}"
  local log="${STORE}/logs/${NATURAL}.log"
  [[ -e "${run_dir}" ]] && { note "natural run already exists: ${run_dir}"; return; }
  note "starting natural-sampling SELECT control on GPU2"
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    nohup "${PY}" scripts/train_semantic_vlm_sft.py \
      "${MODEL}" "${SELECT_TRAIN}" "${run_dir}" --eval-jsonl "${SELECT_VAL}" \
      --epochs 2 --learning-rate 0.0001 --batch-size 1 --gradient-accumulation 16 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 --seed "${SEED}" \
      --logging-steps 5 --eval-steps 50 --save-steps 50 --save-total-limit 4 \
      --early-stopping-patience 3 --dataloader-num-workers 2 --dataloader-persistent-workers \
      >"${log}" 2>&1 < /dev/null &
  note "natural training pid=$!"
}

aggregate_select() {
  [[ -s "${AGGREGATE}" ]] && { note "SELECT aggregate already exists"; return; }
  note "aggregating SELECT calibration gates"
  STORE="${STORE}" PY="${PY}" OUTPUT="${AGGREGATE}" POLL_SECONDS="${POLL_SECONDS}" \
    bash scripts/watch_muno21_qwen3vl_sequential_select_calibration_hdpi.sh
}

launch_joint_if_promoted() {
  [[ "${ALLOW_JOINT}" == "1" ]] || { note "joint continuation disabled"; return; }
  local selected
  selected="$(${PY} - "${AGGREGATE}" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if not record["recommended_run_passed"]:
    raise SystemExit(4)
print(record["recommended_run"].removesuffix("_eval"))
PY
)" || { note "no SELECT variant passed: stopping before joint SFT"; return; }
  local joint="muno21_qwen3vl4b_sequential_joint_v1_from_${selected}"
  local run_dir="${STORE}/runs/agent/${joint}"
  local eval_dir="${STORE}/runs/agent/${joint}_select_eval"
  [[ -s "${eval_dir}/summary.json" ]] && { note "joint continuation already evaluated"; return; }
  [[ ! -e "${run_dir}" ]] || { note "existing joint output blocks overwrite: ${run_dir}"; return 3; }
  note "SELECT gate passed (${selected}); starting one joint-SFT continuation on GPU4"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    "${PY}" scripts/train_semantic_vlm_sft.py \
      "${MODEL}" "${JOINT_TRAIN}" "${run_dir}" --eval-jsonl "${SELECT_VAL}" \
      --init-adapter "${STORE}/runs/agent/${selected}/final" \
      --epochs 1 --learning-rate 0.00005 --batch-size 1 --gradient-accumulation 16 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 --seed "${SEED}" \
      --logging-steps 10 --eval-steps 100 --save-steps 100 --save-total-limit 3 \
      --early-stopping-patience 2 --dataloader-num-workers 2 --dataloader-persistent-workers \
      >"${STORE}/logs/${joint}.log" 2>&1
  note "evaluating joint continuation on GPU4"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "${PY}" scripts/evaluate_sequential_selector.py \
      "${MODEL}" "${run_dir}/final" "${SELECT_VAL}" "${eval_dir}" \
      --device cuda:0 --expected-records 414 --bootstrap-repetitions 2000 --seed "${SEED}" \
      >"${STORE}/logs/${joint}.select_eval.log" 2>&1
  note "joint continuation evaluation complete; no GRPO is scheduled by this pipeline"
}

note "overnight sequential controller pipeline started"
launch_natural_train

# GPU7 already owns the primary evaluator.  The other evaluators reuse their
# released training GPUs, so the controller never exceeds the configured cap.
run_evaluation "${CAL15}" 1 &
cal_eval_pid=$!
run_evaluation "${NATURAL}" 2 &
natural_eval_pid=$!

wait_for_adapter "${PRIMARY}"
primary_summary="${STORE}/runs/agent/${PRIMARY}_eval/summary.json"
until [[ -s "${primary_summary}" ]]; do
  note "waiting for existing primary evaluation"
  sleep "${POLL_SECONDS}"
done
wait "${cal_eval_pid}"
wait "${natural_eval_pid}"
aggregate_select
launch_joint_if_promoted
note "overnight sequential controller pipeline finished"
