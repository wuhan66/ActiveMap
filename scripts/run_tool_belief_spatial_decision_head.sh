#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${TOOL_BELIEF_RUN_ROOT:-/mnt/mydisk/wh/ActiveMap/runs/agent}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent}"
TRAIN_DETAILS="${TRAIN_DETAILS:-${RUN_ROOT}/tool_belief_anchored_v4_seed20260821_sequence_train_features/details.jsonl}"
VAL_DETAILS="${VAL_DETAILS:-${RUN_ROOT}/tool_belief_anchored_v4_seed20260821_sequence_eval_detailed/details.jsonl}"
TRAIN_RECORDS="${TRAIN_RECORDS:-${PROCESSED_ROOT}/tool_belief_v1/train.jsonl}"
VAL_RECORDS="${VAL_RECORDS:-${PROCESSED_ROOT}/tool_belief_v1/val.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROCESSED_ROOT}/tool_belief_v1/artifacts/temporal_change}"
SEED="${SEED:-20260829}"
GPU="${GPU:-3}"
ABLATION="${ABLATION:-spatial}"
EXPERIMENT_VERSION="${EXPERIMENT_VERSION:-v1}"
FUSION_MODE="${FUSION_MODE:-concat}"
GATE_BIAS="${GATE_BIAS:--2.0}"
RUN_NAME="${RUN_NAME:-tool_belief_spatial_decision_${EXPERIMENT_VERSION}_${ABLATION}_seed${SEED}}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-/mnt/mydisk/wh/ActiveMap/logs}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"
TELEMETRY="${LOG_ROOT}/${RUN_NAME}_gpu.csv"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for path in \
  "$PYTHON" "$TRAIN_DETAILS" "$VAL_DETAILS" "$TRAIN_RECORDS" "$VAL_RECORDS" "$ARTIFACT_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  exit 3
fi
if pgrep -u "$USER" -f '[a]ctivemap.cli train-updater|[t]rain_agent_(sft|dpo).py' >/dev/null; then
  echo "Another ActiveMap training experiment is already running" >&2
  exit 4
fi
if pgrep -u "$USER" -f '[t]rain_tool_belief_spatial_decision_head.py' >/dev/null; then
  echo "Another spatial Tool-Belief trainer is already running" >&2
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
if [[ "$ABLATION" != "spatial" && "$ABLATION" != "no_spatial" ]]; then
  echo "ABLATION must be spatial or no_spatial" >&2
  exit 7
fi
if [[ "$EXPERIMENT_VERSION" == "v1" && "$FUSION_MODE" != "concat" ]]; then
  echo "v1 requires concat fusion" >&2
  exit 8
fi
if [[ "$EXPERIMENT_VERSION" == "v2" && "$FUSION_MODE" != "gated_residual" ]]; then
  echo "v2 requires gated_residual fusion" >&2
  exit 8
fi

mkdir -p "$LOG_ROOT"
printf 'timestamp,index,utilization_gpu_percent,memory_used_mib,power_draw_w\n' > "$TELEMETRY"
extra_args=()
if [[ "$ABLATION" == "no_spatial" ]]; then
  extra_args+=(--no-spatial)
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/train_tool_belief_spatial_decision_head.py \
  "$TRAIN_DETAILS" "$VAL_DETAILS" "$TRAIN_RECORDS" "$VAL_RECORDS" \
  "$ARTIFACT_DIR" "$RUN_DIR" --device cuda:0 --seed "$SEED" \
  --epochs 120 --patience 20 --batch-size 32 \
  --fusion-mode "$FUSION_MODE" --gate-bias "$GATE_BIAS" "${extra_args[@]}" \
  > "$LOG_PATH" 2>&1 &
train_pid=$!
echo "$train_pid" > "${LOG_PATH}.pid"
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
echo "$status" > "${LOG_PATH}.exit_code"
exit "$status"
