#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DATA_ROOT="${POST_ACQUISITION_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/post_acquisition_tool_pair_v1}"
RUN_ROOT="${POST_ACQUISITION_RUN_ROOT:-${STORAGE_ROOT}/runs/agent}"
GPU="${POST_ACQUISITION_GPU:-0}"
SEEDS=(20260821 20260822 20260823)

for path in "${DATA_ROOT}/train.jsonl" "${DATA_ROOT}/val.jsonl"; do
  [[ -s "$path" ]] || { echo "Required input is missing: $path" >&2; exit 2; }
done

run_one() {
  local variant="$1"
  local seed="$2"
  local output="${RUN_ROOT}/post_acquisition_belief_${variant}_seed${seed}_v2"
  local log="${output}/train.log"
  local gate_args=()
  [[ "$variant" == "gated" ]] && gate_args=(--reliability-gate --gate-bias -1.5)

  if [[ -s "${output}/summary.json" ]]; then
    echo "skip_completed=${variant}_seed${seed}"
    return
  fi
  mkdir -p "$output"
  cat >"${output}/launch.txt" <<EOF
protocol=post-acquisition-reliability-gate-matched-v2
variant=${variant}
physical_gpu=${GPU}
seed=${seed}
train=${DATA_ROOT}/train.jsonl
val=${DATA_ROOT}/val.jsonl
test_assets_read=false
EOF
  echo "start=$(date --iso-8601=seconds) variant=${variant} seed=${seed} gpu=${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON" "${PROJECT_ROOT}/scripts/train_post_acquisition_tool_belief.py" \
      "${DATA_ROOT}/train.jsonl" "${DATA_ROOT}/val.jsonl" "$output" \
      --device cuda:0 --epochs 200 --batch-size 128 --patience 20 \
      --seed "$seed" "${gate_args[@]}" 2>&1 | tee "$log"
}

cd "$PROJECT_ROOT"
for variant in gated ungated; do
  for seed in "${SEEDS[@]}"; do
    run_one "$variant" "$seed"
  done
done

echo "complete=$(date --iso-8601=seconds)"
