#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-selector}"
MODE="${2:-full}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/server_hdpi_env.sh"
# shellcheck source=/dev/null
source "$ROOT_DIR/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV}/bin/python}"
DATA_ROOT="${ACTIVEMAP_STORAGE_ROOT:-${ACTIVEMAP_DATA_ROOT:-/home/wh/ActiveMap}}"
GPU="${ACTIVEMAP_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
if [[ ! "$GPU" =~ ^(0|1)$ ]]; then
  echo "Refusing GPU $GPU: ActiveMap is restricted to physical GPUs 0 and 1" >&2
  exit 2
fi
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

case "$TASK:$MODE" in
  selector:full) MATRIX="configs/experiments/ablation_matrix.yaml" ;;
  updater:full) MATRIX="configs/experiments/updater_ablation_matrix.yaml" ;;
  selector:smoke) MATRIX="configs/experiments/selector_smoke_matrix.yaml" ;;
  updater:smoke) MATRIX="configs/experiments/updater_smoke_matrix.yaml" ;;
  selector:paper) MATRIX="configs/experiments/muno21_selector_paper_ablations.yaml" ;;
  updater:paper) MATRIX="configs/experiments/sn7_updater_paper_ablations.yaml" ;;
  *) echo "Usage: bash scripts/run_ablations.sh {selector|updater} {full|smoke|paper}"; exit 1 ;;
esac

if [[ "$MODE" == "paper" ]]; then
  "$PYTHON" scripts/assert_training_ready.py \
    configs/experiments/paper_registry.yaml "$DATA_ROOT"
fi

"$PYTHON" -m activemap run-ablations "$MATRIX"
