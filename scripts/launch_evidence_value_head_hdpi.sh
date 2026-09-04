#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/wh/projects/activemap-v1}"
DATA="${DATA:-/home/wh/ActiveMap/processed/sn7_v1/agent/executable_selector_v3_512_sharded/states_train_val_executable_balanced_m15_512.jsonl}"
ROOT="${ROOT:-/home/wh/ActiveMap/runs/evidence_value_head_v1_20260725}"
PYTHON="${PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"

launch() {
  local gpu="$1"
  local seed="$2"
  local name="$3"
  shift 3
  local output="${ROOT}/${name}"
  if [[ -e "${output}/run" ]]; then
    echo "Refusing to overwrite ${output}/run" >&2
    return 1
  fi
  mkdir -p "${output}"
  (
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. nohup "${PYTHON}" \
      scripts/train_evidence_value_head.py \
      "${DATA}" "${output}/run" \
      --device cuda:0 \
      --seed "${seed}" \
      --epochs 40 \
      --batch-size 128 \
      "$@" \
      >"${output}/train.log" 2>&1 < /dev/null &
    echo "$!" >"${output}/pid"
  )
  echo "${name}: pid=$(cat "${output}/pid") gpu=${gpu}"
}

mkdir -p "${ROOT}"
launch 0 20260725 seed1
launch 2 20260726 seed2
launch 3 20260727 seed3
launch 6 20260725 utility_only_seed1 \
  --quality-weight 0 \
  --beneficial-weight 0 \
  --unsafe-loss-weight 0 \
  --missed-loss-weight 0 \
  --unsafe-penalty 0 \
  --missed-penalty 0
