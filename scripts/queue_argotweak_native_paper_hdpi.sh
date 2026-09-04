#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
ROOT="${ROOT:-${STORE}/runs/argotweak/paper_queue_v2}"
ACTIVE_PYTHON="${STORE}/envs/activemap-agent/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
WEIGHTS="${HOME}/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
GPU="${GPU:-0}"
NATIVE="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2"
mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/queue.lock"
flock -n 9 || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"

stage() {
  local name="$1"; shift
  if [[ -e "${ROOT}/status/${name}.done" ]]; then return; fi
  printf '%s\tSTART\t%s\n' "$(date -Is)" "${name}" >> "${ROOT}/queue.tsv"
  if "$@" > >(tee "${ROOT}/logs/${name}.log") 2>&1; then
    touch "${ROOT}/status/${name}.done"
    printf '%s\tDONE\t%s\n' "$(date -Is)" "${name}" >> "${ROOT}/queue.tsv"
  else
    touch "${ROOT}/status/${name}.failed"
    printf '%s\tFAILED\t%s\n' "$(date -Is)" "${name}" >> "${ROOT}/queue.tsv"
    exit 1
  fi
}

stage 00_native_adapter env ROOT="${NATIVE}" MATCH_DISTANCE=1.5 \
  bash scripts/launch_argotweak_native_adapter_hdpi.sh

stage 01_train_features "${ACTIVE_PYTHON}" scripts/extract_argotweak_visual_features.py \
  --episodes "${NATIVE}/episodes/train.jsonl" --weights "${WEIGHTS}" \
  --output-dir "${ROOT}/features/train" --device cuda --batch-size 128 --workers 12

stage 02_val_features "${ACTIVE_PYTHON}" scripts/extract_argotweak_visual_features.py \
  --episodes "${NATIVE}/episodes/val.jsonl" --weights "${WEIGHTS}" \
  --output-dir "${ROOT}/features/val" --device cuda --batch-size 128 --workers 12

stage 03_selector_seed1 "${ACTIVE_PYTHON}" scripts/train_argotweak_visual_selector.py \
  --features "${ROOT}/features/train" --output-dir "${ROOT}/selector/seed1" \
  --seed 20260841 --device cuda

stage 04_selector_val_seed1 "${ACTIVE_PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
  --features "${ROOT}/features/val" --checkpoint "${ROOT}/selector/seed1/best.pt" \
  --output-dir "${ROOT}/validation/seed1" --device cuda

# Replication is deliberately promotion-gated: failed ideas consume one seed, not three.
stage 05_promotion "${GIS_PYTHON}" -c \
  "import json,pathlib; p=pathlib.Path('${ROOT}/validation/seed1/summary.json'); m=json.loads(p.read_text()); ok=m['utility_gain_over_direct'] > 0 and m['exact_top1'] >= 0.10; pathlib.Path('${ROOT}/status/PROMOTED' if ok else '${ROOT}/status/NOT_PROMOTED').touch(); print(json.dumps({'promoted':ok,'criteria':{'utility_gain_over_direct_gt':0,'exact_top1_ge':0.10},'metrics':m},indent=2))"

if [[ -e "${ROOT}/status/PROMOTED" ]]; then
  stage 06_selector_seed2 "${ACTIVE_PYTHON}" scripts/train_argotweak_visual_selector.py \
    --features "${ROOT}/features/train" --output-dir "${ROOT}/selector/seed2" --seed 20260842 --device cuda
  stage 07_selector_seed3 "${ACTIVE_PYTHON}" scripts/train_argotweak_visual_selector.py \
    --features "${ROOT}/features/train" --output-dir "${ROOT}/selector/seed3" --seed 20260843 --device cuda
  stage 08_selector_val_seed2 "${ACTIVE_PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${ROOT}/features/val" --checkpoint "${ROOT}/selector/seed2/best.pt" \
    --output-dir "${ROOT}/validation/seed2" --device cuda
  stage 09_selector_val_seed3 "${ACTIVE_PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${ROOT}/features/val" --checkpoint "${ROOT}/selector/seed3/best.pt" \
    --output-dir "${ROOT}/validation/seed3" --device cuda
fi

cat > "${ROOT}/NEXT_EXPERIMENTS.md" <<'EOF'
# ArgoTweak queue handoff

Completed here: native pose-aware adapter, frozen seven-camera ResNet50 features,
train-only AOI-disjoint selector screen, and promotion-gated replication.

Next required experiments after reviewing the selector screen:

1. Freeze Safe Commit confidence on train/calibration only (0.30/0.40/0.50/0.60/0.70).
2. Evaluate Prior-only, center-frame Direct, Acquire-all, Greedy top-k, visual ActiveMap top-k, and Oracle on validation.
3. Report edit F1, false-edit rate, missed-edit rate, geometry Chamfer, topology F1, evidence cost, and quality-cost AUC.
4. Run three seeds only for a non-dominated ActiveMap configuration.
5. Keep detector fine-tuning as a negative adaptation ablation; frozen perception is canonical.
EOF
touch "${ROOT}/QUEUE_COMPLETED"
printf '%s\tDONE\tQUEUE\n' "$(date -Is)" >> "${ROOT}/queue.tsv"
