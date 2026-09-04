#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
CHANGE_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
BAN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
VISUAL_ROOT="${STORAGE_ROOT}/runs/paper_visuals/sn7_updater_20260726"
HISTORY_ROOT="${STORAGE_ROOT}/history/20260726_pre_logfix"
FINAL_ROOT="${STORAGE_ROOT}/runs/finalization"
POLL_SECONDS="${POLL_SECONDS:-180}"

required=(
  "${CHANGE_ROOT}/full_weight5_modality_comparison_20260726.json"
  "${CHANGE_ROOT}/full_weight5_modality_aoi_bootstrap_20260726.json"
  "${CHANGE_ROOT}/candidate_localization_audit_20260726.json"
  "${CHANGE_ROOT}/proposal_jitter16_pilot_20260726.json"
  "${BAN_ROOT}/full_weight5_three_seed_20260726.json"
  "${BAN_ROOT}/ban_vs_changemamba_aoi_bootstrap_20260726.json"
  "${VISUAL_ROOT}/bundle_summary.json"
)
while true; do
  missing=0
  for path in "${required[@]}"; do
    if [[ ! -s "${path}" ]]; then
      echo "Waiting for ${path}"
      missing=1
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

mkdir -p \
  "${HISTORY_ROOT}/runs" \
  "${HISTORY_ROOT}/logs" \
  "${HISTORY_ROOT}/visual_smokes" \
  "${FINAL_ROOT}"
for seed in 20260725 20260726 20260727; do
  source="${BAN_ROOT}/full_weight5_seed${seed}_v1_pre_logfix"
  if [[ -e "${source}" ]]; then
    mv "${source}" "${HISTORY_ROOT}/runs/"
  fi
done
if [[ -e "${STORAGE_ROOT}/logs/opencd_ban_formal_20260726_pre_logfix" ]]; then
  mv "${STORAGE_ROOT}/logs/opencd_ban_formal_20260726_pre_logfix" \
    "${HISTORY_ROOT}/logs/"
fi
for source in \
  "${STORAGE_ROOT}/runs/paper_visuals/sn7_updater_20260726_smoke_changemamba" \
  "${STORAGE_ROOT}/runs/paper_visuals/sn7_updater_20260726_smoke_changemamba_v2"; do
  if [[ -e "${source}" ]]; then
    mv "${source}" "${HISTORY_ROOT}/visual_smokes/"
  fi
done

cd "${PROJECT_ROOT}"
"${PYTHON}" - "${FINAL_ROOT}" "${HISTORY_ROOT}" "${required[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_root = Path(sys.argv[1])
history_root = Path(sys.argv[2])
paths = [Path(value) for value in sys.argv[3:]]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": "activemap-current-round-finalization-v1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "required_artifacts": {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    },
    "archived_superseded_outputs": str(history_root),
    "archive_policy": "move-only; no experiment data deleted",
    "ready_for_manual_ledger_freeze": True,
    "test_assets_read": False,
}
path = output_root / "current_round_20260726.json"
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
