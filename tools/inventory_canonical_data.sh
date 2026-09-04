#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${1:?usage: inventory_canonical_data.sh STORAGE_ROOT OUTPUT_TSV}"
OUTPUT="${2:?usage: inventory_canonical_data.sh STORAGE_ROOT OUTPUT_TSV}"

assets=(
  "processed/muno21_v2/rsprompter_road_v2_rle/annotations/train.json"
  "processed/muno21_v2/rsprompter_road_v2_rle/annotations/val.json"
  "processed/muno21_v2/updater/updater_samples.jsonl"
  "processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools/train/sft_composed.jsonl"
  "processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools/train/sft_composed.summary.json"
  "processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools/val/sft_composed.jsonl"
  "processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools/val/sft_composed.summary.json"
  "processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl"
  "processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.summary.json"
  "processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/val/sft_composed.jsonl"
  "processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/val/sft_composed.summary.json"
  "processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/balanced_prior_audit.json"
  "models/sam_road/sam_vit_b_01ec64.pth"
  "models/sam_road/spacenet_vitb_256_e10.ckpt"
  "runs/semantic/muno21_sam_road_head_seed20260716_v1/checkpoints/best_full.ckpt"
)

mkdir -p "$(dirname "$OUTPUT")"
{
  printf 'asset\tstatus\tbytes\tsha256\n'
  for asset in "${assets[@]}"; do
    path="${STORAGE_ROOT}/${asset}"
    if [[ -f "$path" ]]; then
      bytes=$(stat -c '%s' "$path")
      checksum=$(sha256sum "$path" | awk '{print $1}')
      printf '%s\tpresent\t%s\t%s\n' "$asset" "$bytes" "$checksum"
    else
      printf '%s\tmissing\t0\t-\n' "$asset"
    fi
  done
} > "$OUTPUT"

cat "$OUTPUT"
