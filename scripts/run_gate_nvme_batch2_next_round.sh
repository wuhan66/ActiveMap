#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 stage|smoke|train [arguments...]" >&2
  exit 2
fi

MODE="$1"
shift
ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${ACTIVEMAP_REPOSITORY:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-$ROOT/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
SOURCE="${GATE_SOURCE_DATA:-$ROOT/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_gate_sft_v1}"
STAGED="${GATE_NVME_DATA:-/data1/wh/ActiveMap/cache/active_catalog_gate_sft_v1}"
SMOKE_ROOT="${GATE_SMOKE_ROOT:-$ROOT/runs/sn7_active_catalog/nvme_batch2_smoke}"
INIT_ADAPTER="${INIT_ADAPTER:-$ROOT/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed2/seed20260718/final}"

cd "$REPO"
export PYTHONPATH=src

case "$MODE" in
  stage)
    "$PYTHON" scripts/stage_vlm_sft_nvme.py "$SOURCE" "$STAGED" --workers 8
    ;;
  smoke)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 smoke GPU" >&2
      exit 2
    fi
    [[ -s "$STAGED/stage_manifest.json" ]] || { echo "run stage first" >&2; exit 3; }
    "$PYTHON" scripts/launch_semantic_vlm_sft_seeds.py \
      "$MODEL" "$STAGED/train.jsonl" "$STAGED/val.jsonl" "$SMOKE_ROOT" \
      --gpu "$1" --seed 20260720 --epochs 0.01 --batch-size 2 \
      --gradient-accumulation 8 --max-train-samples 8 --max-eval-samples 8 \
      --dataloader-num-workers 4 --dataloader-prefetch-factor 2 \
      --dataloader-persistent-workers --attn-implementation sdpa \
      --init-adapter "$INIT_ADAPTER" --acquire-sampling-target 0.10
    "$PYTHON" scripts/validate_vlm_sft_smoke.py "$SMOKE_ROOT" --max-peak-memory-mib 23500
    ;;
  train)
    if [[ $# -ne 4 ]]; then
      echo "usage: $0 train RUN_NAME SEED ACQUIRE_TARGET GPU" >&2
      exit 2
    fi
    [[ -s "$STAGED/stage_manifest.json" ]] || { echo "run stage first" >&2; exit 3; }
    [[ -s "$SMOKE_ROOT/smoke_approved.json" ]] || { echo "approved smoke missing" >&2; exit 3; }
    export GATE_DATA="$STAGED"
    export GATE_BATCH_SIZE=2
    export GATE_GRADIENT_ACCUMULATION=8
    export GATE_DATALOADER_WORKERS=4
    export GATE_DATALOADER_PREFETCH=2
    export GATE_ATTN_IMPLEMENTATION=sdpa
    exec bash scripts/run_active_catalog_gate_ranker_experiment.sh "$@"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
