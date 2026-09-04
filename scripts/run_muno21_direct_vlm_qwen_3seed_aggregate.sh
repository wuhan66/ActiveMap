#!/usr/bin/env bash
set -euo pipefail

export SEEDS="20260831 20260901 20260902"
export OUTPUT="${OUTPUT:-/home/wh/ActiveMap/artifacts/paper_evidence/muno21_direct_vlm_supervised_3seed_v1}"
exec bash "${PROJECT_ROOT:-/home/wh/projects/activemap-v1}/scripts/watch_aggregate_muno21_direct_vlm_sft_3seed.sh"
