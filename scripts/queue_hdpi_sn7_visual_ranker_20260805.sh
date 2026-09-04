#!/usr/bin/env bash
set -euo pipefail

# Validation-only visual-policy screen for the SN7 VLA-inspired claim.
# GPUs 0 and 6 are intentionally excluded; every stage is resumable.
PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-${STORE}/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed3/seed20260719/final}"
RL="${RL:-${STORE}/runs/sn7_active_catalog/active_catalog_rl_states}"
DATA="${DATA:-${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full}"
RUN="${RUN:-${STORE}/runs/sn7_active_catalog/visual_ranker_screen_20260805}"
ROOT="${STORE}/runs/queues/sn7_visual_ranker_20260805"
HEAD="${HEAD:-${STORE}/runs/sn7_active_catalog/vla_utility_head_last_pilot}"
LOG="${STORE}/logs/sn7_visual_ranker_20260805"
GPUS=(1 2 3)

mkdir -p "${ROOT}/logs" "${ROOT}/status" "${RUN}" "${LOG}"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for path in \
  "${PYTHON}" "${MODEL}/config.json" "${ADAPTER}/adapter_config.json" \
  "${RL}/train.jsonl" "${RL}/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/train.jsonl" \
  "${DATA}/active_catalog_sft_v4/train_evaluation_index.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${HEAD}/gate.joblib" "${HEAD}/summary.json"; do
  [[ -e "${path}" ]] || { echo "missing input: ${path}" >&2; touch "${ROOT}/QUEUE_FAILED"; exit 3; }
done

extract_shard() {
  local gpu="$1" shard="$2"
  for split in train val; do
    local output="${RUN}/vla_features_${split}_s${shard}of3"
    local input="${RL}/${split}.jsonl"
    if [[ ! -s "${output}/summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/extract_active_catalog_vla_features.py \
        "${MODEL}" "${ADAPTER}" "${input}" "${output}" \
        --pooling last --batch-size 2 --num-shards 3 --shard-index "${shard}" \
        >"${ROOT}/logs/features_${split}_${shard}.log" 2>&1
    fi
  done
  touch "${ROOT}/status/features_${shard}.done"
}

for shard in 0 1 2; do
  extract_shard "${GPUS[$shard]}" "${shard}" &
done
wait

if [[ ! -s "${RUN}/vla_features_train/summary.json" ]]; then
  "${PYTHON}" scripts/merge_active_catalog_vla_features.py \
    "${RUN}/vla_features_train" \
    "${RUN}/vla_features_train_s0of3" \
    "${RUN}/vla_features_train_s1of3" \
    "${RUN}/vla_features_train_s2of3" \
    >"${ROOT}/logs/merge_train.log" 2>&1
fi
if [[ ! -s "${RUN}/vla_features_val/summary.json" ]]; then
  "${PYTHON}" scripts/merge_active_catalog_vla_features.py \
    "${RUN}/vla_features_val" \
    "${RUN}/vla_features_val_s0of3" \
    "${RUN}/vla_features_val_s1of3" \
    "${RUN}/vla_features_val_s2of3" \
    >"${ROOT}/logs/merge_val.log" 2>&1
fi
touch "${ROOT}/status/features_merged.done"

train_variant() {
  local gpu="$1" name="$2" hidden="$3" dropout="$4" pairwise="$5"
  local output="${RUN}/${name}"
  if [[ ! -s "${output}/best.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_active_catalog_candidate_ranker.py \
      "${DATA}/active_catalog_sft_v4/train.jsonl" \
      "${DATA}/active_catalog_sft_v4/train_evaluation_index.jsonl" \
      "${DATA}/active_catalog_sft_v4/val.jsonl" \
      "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
      "${output}" --device cuda --seed 20260805 --epochs 20 --patience 5 \
      --batch-size 256 --learning-rate 1e-4 --hidden-dim "${hidden}" \
      --dropout "${dropout}" --regression-weight 1 --listwise-weight 1 \
      --pairwise-weight "${pairwise}" --gate-weight 0.25 --positive-weight 4 \
      --positive-sampling-fraction 0.25 --maximum-false-call-rate 0.05 \
      --train-state-features "${RUN}/vla_features_train" \
      --val-state-features "${RUN}/vla_features_val" \
      >"${ROOT}/logs/${name}.log" 2>&1
  fi
}

train_variant 1 ranker_h32_p0 32 0.5 0.0 & p1=$!
train_variant 2 ranker_h32_p05 32 0.5 0.5 & p2=$!
train_variant 3 ranker_h64_p1 64 0.3 1.0 & p3=$!
wait "${p1}" "${p2}" "${p3}"
touch "${ROOT}/status/rankers_trained.done"

evaluate_variant() {
  local gpu="$1" name="$2"
  local output="${RUN}/closed_loop_${name}"
  if [[ ! -s "${output}/evaluation/summary.json" ]]; then
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${ADAPTER}" \
      "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
      "${DATA}/closed_loop_v1/episodes_val.jsonl" \
      "${DATA}/active_catalog_sft_v4/val.jsonl" \
      "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
      "${output}" --gpu "${gpu}" --seed 20260805 \
      --policy-mode utility_head_ranker \
      --ranker-checkpoint "${RUN}/${name}/best.pt" \
      --utility-head "${HEAD}/gate.joblib" \
      --utility-head-summary "${HEAD}/summary.json" \
      --utility-threshold-override 0.02 --max-candidates 16 \
      --max-acquisitions 2 --bootstrap-repetitions 2000 --monitor-interval 5 \
      >"${ROOT}/logs/closed_loop_${name}.log" 2>&1
  fi
}

evaluate_variant 1 ranker_h32_p0 & e1=$!
evaluate_variant 2 ranker_h32_p05 & e2=$!
evaluate_variant 3 ranker_h64_p1 & e3=$!
status=0
for pid in "${e1}" "${e2}" "${e3}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  touch "${ROOT}/QUEUE_FAILED"
  exit 7
fi

printf '%s\n' '{"schema_version":"sn7-visual-ranker-screen-v1","split":"val","test_assets_read":false,"seed":20260805,"gpus":[1,2,3],"variants":["ranker_h32_p0","ranker_h32_p05","ranker_h64_p1"],"threshold":0.02}' >"${RUN}/protocol.json"
date -Is >"${ROOT}/QUEUE_COMPLETED"
