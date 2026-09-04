#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 DATA_ROOT BASE_EVAL_ROOT RUNS_ROOT PHYSICAL_GPU_INDEX [POLL_SECONDS]" >&2
  exit 2
fi

DATA_ROOT="$1"
BASE_EVAL_ROOT="$2"
RUNS_ROOT="$3"
GPU_INDEX="$4"
POLL_SECONDS="${5:-30}"
if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU index must be non-negative and poll seconds must be positive" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EVAL="$BASE_EVAL_ROOT/summary.json"
AUDIT="$DATA_ROOT/audit.json"
TRAIN="$DATA_ROOT/train.jsonl"
VAL="$DATA_ROOT/val.jsonl"
SEED=20260821

echo "waiting_for=$BASE_EVAL poll_seconds=$POLL_SECONDS"
while [[ ! -s "$BASE_EVAL" ]]; do
  sleep "$POLL_SECONDS"
done
/home/wh/venvs/activemap/bin/python - "$BASE_EVAL" <<'PY'
import json
import sys

path = sys.argv[1]
report = json.load(open(path, encoding="utf-8"))
if report.get("gates", {}).get("passed") is not True:
    raise SystemExit(f"base intervention gates did not pass: {path}")
PY

gpu_is_free() {
  local gpu_uuid
  gpu_uuid="$({ nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits || true; } \
    | awk -F', *' -v gpu_index="$GPU_INDEX" '$1 == gpu_index {print $2}')"
  if [[ -z "$gpu_uuid" ]]; then
    echo "GPU index $GPU_INDEX does not exist" >&2
    return 1
  fi
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"
}

run_ablation() {
  local name="$1"
  shift
  local run_dir="$RUNS_ROOT/tool_belief_v1_${name}_seed${SEED}"
  local eval_dir="${run_dir}_intervention_eval"
  if [[ -e "$run_dir/summary.json" || -e "$eval_dir/summary.json" ]]; then
    echo "refusing_start=existing artifacts for $name" >&2
    return 1
  fi
  if pgrep -u "$USER" -f '[p]ython .*scripts/train_' >/dev/null; then
    echo "refusing_start=another ActiveMap trainer is active" >&2
    return 1
  fi
  if ! gpu_is_free; then
    echo "refusing_start=GPU $GPU_INDEX is occupied before $name" >&2
    return 1
  fi

  echo "ablation_start=$(date --iso-8601=seconds) name=$name gpu=$GPU_INDEX"
  cd "$PROJECT_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH=src \
    /home/wh/venvs/activemap/bin/python scripts/train_tool_belief.py \
    "$TRAIN" "$VAL" "$run_dir" \
    --audit-report "$AUDIT" --device cuda:0 --seed "$SEED" "$@"
  PYTHONPATH=src /home/wh/venvs/activemap/bin/python \
    scripts/evaluate_tool_belief_interventions.py \
    "$VAL" "$run_dir/best.pt" "$eval_dir" --device cpu
  echo "ablation_complete=$(date --iso-8601=seconds) name=$name"
}

run_ablation no_operation --operation-weight 0
run_ablation no_teacher --teacher-kl-weight 0
