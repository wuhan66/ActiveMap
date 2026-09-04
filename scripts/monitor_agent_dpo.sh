#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: monitor_agent_dpo.sh RUN_ROOT [GPU]}"
GPU="${2:-3}"
PID_FILE="${RUN_ROOT}/train.pid"
TELEMETRY="${RUN_ROOT}/gpu_telemetry.csv"
STATUS="${RUN_ROOT}/completion_status.json"

[[ -s "$PID_FILE" ]] || { echo "Missing DPO PID file: $PID_FILE" >&2; exit 2; }
train_pid="$(cat "$PID_FILE")"
[[ "$train_pid" =~ ^[0-9]+$ ]] || { echo "Invalid DPO PID: $train_pid" >&2; exit 2; }

printf 'timestamp,index,utilization_gpu_percent,memory_used_mib,power_draw_w\n' > "$TELEMETRY"
while kill -0 "$train_pid" 2>/dev/null; do
  nvidia-smi --id="$GPU" \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits >> "$TELEMETRY"
  sleep 1
done

complete=true
for path in \
  "${RUN_ROOT}/final/adapter_model.safetensors" \
  "${RUN_ROOT}/train_metrics.json" \
  "${RUN_ROOT}/eval_metrics.json"; do
  if [[ ! -s "$path" ]]; then
    complete=false
  fi
done

printf '{\n  "train_pid": %s,\n  "artifacts_complete": %s,\n  "test_assets_read": false\n}\n' \
  "$train_pid" "$complete" > "$STATUS"
if [[ "$complete" != "true" ]]; then
  exit 1
fi
