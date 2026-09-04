#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808}"
SFT_REPLAY="${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
ROLLOUT_ROOT="${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n2048_v1"
MATRIX="${STORE}/runs/agent/muno21_sft_anchored_grpo_n2048_v1"
LOGS="${STORE}/logs/muno21_sft_anchored_grpo_n2048_v1"

[[ ! -e "${MATRIX}" ]] || exit 0
for path in "${PY}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${SFT_REPLAY}" "${STATES}" "${EPISODES}" "${BELIEF}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
for index in $(seq 0 7); do
  seed=$((20261060 + index))
  [[ -s "${ROLLOUT_ROOT}/rollout${index}_seed${seed}/qwen3_4b_sft_tool_to_belief.jsonl" ]] || exit 3
  [[ -s "${ROLLOUT_ROOT}/rollout${index}_seed${seed}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" ]] || exit 3
done
mkdir -p "${MATRIX}" "${LOGS}"
cat >"${MATRIX}/protocol.json" <<EOF
{"schema_version":"muno21-sft-anchored-grpo-n2048-v1","support":"g8-n2048","sft_replay_weight":0.05,"sft_replay_probability":0.25,"sft_tool_replay_fraction":0.50,"sequence_kl_coef":0.01,"false_edit_lagrange":0.25,"policy_seeds":[20261311,20261312,20261313],"physical_gpus":[1,2],"test_assets_read":false}
EOF

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
rollout_args=()
for index in $(seq 0 7); do
  seed=$((20261060 + index)); run="${ROLLOUT_ROOT}/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

run_one() {
  local gpu="$1" seed="$2" run="${MATRIX}/seed${2}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT}" "${run}/policy" "${rollout_args[@]}" \
    --objective sequence-clip --dynamic-sampling variable-only \
    --false-edit-lagrange 0.25 --false-edit-target 0.0966 --false-edit-dual-lr 1.0 \
    --minimum-keep-trajectories 100 --minimum-false-edit-trajectories 1 \
    --learning-rate 2.5e-7 --entropy-coef 0.001 --sequence-kl-coef 0.01 \
    --sft-replay "${SFT_REPLAY}" --sft-replay-weight 0.05 \
    --sft-replay-probability 0.25 --sft-tool-replay-fraction 0.50 \
    --gradient-accumulation 4 --epochs 1 --seed "${seed}" >"${LOGS}/seed${seed}.log" 2>&1
  local output="${run}/validation_full_gpu_selector"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${run}/policy/final" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 512 --max-length 2048 --max-tool-calls 2 --seed "${seed}" >>"${LOGS}/seed${seed}.log" 2>&1
  printf '{"status":"complete","seed":%s,"test_assets_read":false}\n' "${seed}" >"${run}/COMPLETED.json"
}

(run_one 1 20261311; run_one 1 20261313) & p1=$!
(run_one 2 20261312) & p2=$!
wait "${p1}" "${p2}"
date -Is >"${MATRIX}/MATRIX_COMPLETED"
