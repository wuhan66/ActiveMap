#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${TOOL_BELIEF_RUN_ROOT:-/mnt/mydisk/wh/ActiveMap/runs/agent}"
STAGE_CHECKPOINT="${STAGE_CHECKPOINT:-${RUN_ROOT}/tool_belief_stage_head_v2_seed20260824/best_safety.pt}"
TRAIN_DETAILS="${TRAIN_DETAILS:-${RUN_ROOT}/tool_belief_anchored_v4_seed20260821_sequence_train_features/details.jsonl}"
VAL_DETAILS="${VAL_DETAILS:-${RUN_ROOT}/tool_belief_anchored_v4_seed20260821_sequence_eval_detailed/details.jsonl}"
SELECTOR_STATES="${SELECTOR_STATES:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/selector_states_v1.jsonl}"
SEED="${SEED:-20260825}"
GPU="${GPU:-3}"
RUN_NAME="${RUN_NAME:-tool_belief_stopping_policy_v2_seed${SEED}}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
EVAL_DIR="${RUN_ROOT}/${RUN_NAME}_eval"
LOG_ROOT="${LOG_ROOT:-/mnt/mydisk/wh/ActiveMap/logs}"
TELEMETRY="${LOG_ROOT}/${RUN_NAME}_gpu.csv"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for path in "$PYTHON" "$STAGE_CHECKPOINT" "$TRAIN_DETAILS" "$VAL_DETAILS" "$SELECTOR_STATES"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$RUN_DIR" || -e "$EVAL_DIR" ]]; then
  echo "Refusing to overwrite an existing stopping-policy run: $RUN_DIR" >&2
  exit 3
fi
if pgrep -u "$USER" -f '[t]rain_tool_belief_stopping_policy.py' >/dev/null; then
  echo "Another Tool-Belief stopping-policy trainer is already running" >&2
  exit 4
fi
gpu_uuid="$({ nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits || true; } \
  | awk -F, -v gpu_index="$GPU" \
    '$1 + 0 == gpu_index {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
if [[ -z "$gpu_uuid" ]]; then
  echo "Physical GPU $GPU was not found" >&2
  exit 5
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | grep -Fxq "$gpu_uuid"; then
  echo "Physical GPU $GPU is not free; refusing to start" >&2
  exit 6
fi

mkdir -p "$LOG_ROOT"
printf 'timestamp,index,utilization_gpu_percent,memory_used_mib,power_draw_w\n' > "$TELEMETRY"
echo "Starting causal stopping policy on physical GPU $GPU, seed=$SEED"
echo "run_dir=$RUN_DIR"
echo "telemetry=$TELEMETRY"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/train_tool_belief_stopping_policy.py \
  "$STAGE_CHECKPOINT" "$TRAIN_DETAILS" "$VAL_DETAILS" "$SELECTOR_STATES" "$RUN_DIR" \
  --device cuda:0 --seed "$SEED" --epochs 200 --patience 24 --batch-size 128 &
train_pid=$!
(
  while kill -0 "$train_pid" 2>/dev/null; do
    nvidia-smi --id="$GPU" \
      --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader,nounits >> "$TELEMETRY"
    sleep 0.2
  done
) &
telemetry_pid=$!
status=0
wait "$train_pid" || status=$?
wait "$telemetry_pid" || true
if [[ "$status" -ne 0 ]]; then
  echo "Stopping-policy training failed with exit_code=$status" >&2
  exit "$status"
fi

checkpoint="$RUN_DIR/best_safe.pt"
if [[ ! -f "$checkpoint" ]]; then
  checkpoint="$RUN_DIR/best_utility.pt"
fi
"$PYTHON" scripts/evaluate_tool_belief_stopping_policy.py \
  "$checkpoint" "$STAGE_CHECKPOINT" "$VAL_DETAILS" "$SELECTOR_STATES" "$EVAL_DIR" \
  --device cpu
echo "Stopping-policy training and independent evaluation complete"
