#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DATA_ROOT="${TOOL_BELIEF_RUN_ROOT:-/mnt/mydisk/wh/ActiveMap/runs/agent}"
TRAIN_DETAILS="${TRAIN_DETAILS:-${DATA_ROOT}/tool_belief_anchored_v4_seed20260821_sequence_train_features/details.jsonl}"
VAL_DETAILS="${VAL_DETAILS:-${DATA_ROOT}/tool_belief_anchored_v4_seed20260821_sequence_eval_detailed/details.jsonl}"
SEED="${SEED:-20260824}"
GPU="${GPU:-3}"
RUN_NAME="${RUN_NAME:-tool_belief_stage_head_v2_seed${SEED}}"
RUN_DIR="${DATA_ROOT}/${RUN_NAME}"
EVAL_DIR="${DATA_ROOT}/${RUN_NAME}_eval"
LOG_ROOT="${LOG_ROOT:-/mnt/mydisk/wh/ActiveMap/logs}"
TELEMETRY="${LOG_ROOT}/${RUN_NAME}_gpu.csv"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python is not executable: $PYTHON" >&2
  exit 2
fi
for path in "$TRAIN_DETAILS" "$VAL_DETAILS"; do
  if [[ ! -f "$path" ]]; then
    echo "Required trajectory details are missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$RUN_DIR" || -e "$EVAL_DIR" ]]; then
  echo "Refusing to overwrite an existing stage-head run: $RUN_DIR" >&2
  exit 3
fi
if pgrep -u "$USER" -f '[t]rain_tool_belief_decision_head.py' >/dev/null; then
  echo "Another Tool-Belief decision-head trainer is already running" >&2
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
echo "Starting causal stage head on physical GPU $GPU, seed=$SEED"
echo "run_dir=$RUN_DIR"
echo "telemetry=$TELEMETRY"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/train_tool_belief_decision_head.py \
  "$TRAIN_DETAILS" "$VAL_DETAILS" "$RUN_DIR" \
  --device cuda:0 --prefix-stages 0,1,2,3 --seed "$SEED" \
  --epochs 160 --patience 24 --batch-size 128 &
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
  echo "Stage-head training failed with exit_code=$status" >&2
  exit "$status"
fi

"$PYTHON" scripts/evaluate_tool_belief_stage_head.py \
  "$RUN_DIR/best_safety.pt" "$VAL_DETAILS" "$EVAL_DIR/summary.json" --device cpu
echo "Stage-head training and independent evaluation complete"
