#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
ADAPTER="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR_ROOT="${STORE}/runs/selector"
ROOT="${STORE}/runs/agent/muno21_grpo_exploration_sweep_v1"
LOGS="${STORE}/logs/muno21_grpo_exploration_sweep_v1"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_actor() {
  local gpu="$1" label="$2" temperature="$3" seed="$4"
  local output="${ROOT}/${label}_seed${seed}"
  [[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; return 3; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" \
    --adapter "$ADAPTER" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief \
    --split train --budgets 1.5,3.0,4.5 --limit 16 \
    --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    --do-sample --temperature "$temperature" --top-p 1.0 \
    --record-training-payload >"${LOGS}/${label}_seed${seed}.log" 2>&1
}

# Two 4B actors fit on each 24 GB card. This sweep finds the lowest sampling
# temperature that restores action and terminal-reward variance after SFT.
pids=()
(run_actor 1 t1p5 1.5 20260841) & pids+=("$!")
(run_actor 1 t2p0 2.0 20260842) & pids+=("$!")
(run_actor 2 t2p5 2.5 20260843) & pids+=("$!")
(run_actor 2 t3p0 3.0 20260844) & pids+=("$!")
(run_actor 3 t3p5 3.5 20260845) & pids+=("$!")
(run_actor 3 t4p0 4.0 20260846) & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
