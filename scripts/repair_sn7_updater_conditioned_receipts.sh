#!/usr/bin/env bash
set -euo pipefail

# Repair only receipt files for C5 updater-conditioned state shards.
# It never rewrites states.jsonl and only acts when structured progress says complete.
RUN_ROOT="${RUN_ROOT:?set updater-conditioned run root}"

repair_one() {
  local state="$1" split="$2"
  local root="${RUN_ROOT}/states/${state}/${split}"
  local progress="${root}/states.progress.json"
  local summary="${root}/states.summary.json"
  local output="${root}/states.jsonl"
  local failed="${root}/FAILED.json"
  local superseded="${root}/FAILED_SUPERSEDED_BY_RECEIPT_REPAIR.json"
  local complete="${root}/COMPLETE.json"

  [[ -s "${progress}" && -s "${summary}" && -s "${output}" ]] || return 0
  grep -qE '"status": "(complete|inference_complete)"' "${progress}" || return 0

  if [[ -e "${failed}" && ! -e "${superseded}" ]]; then
    mv "${failed}" "${superseded}"
  fi

  local checkpoint episodes
  checkpoint="$(grep -m1 '"checkpoint"' "${summary}" | sed 's/.*: //; s/[",]//g; s/^ *//')"
  episodes="$(grep -m1 '"episodes"' "${summary}" | sed 's/.*: //; s/,//g; s/^ *//')"
  printf '{"status":"complete","state":"%s","split":"%s","checkpoint":"%s","episodes":%s,"receipt_repair":true,"test_assets_read":false}\n' \
    "${state}" "${split}" "${checkpoint}" "${episodes}" > "${complete}"
}

repair_one f0 train
repair_one f0 val
repair_one f1 train
repair_one f1 val

find "${RUN_ROOT}/states" -mindepth 2 -maxdepth 2 \( -name COMPLETE.json -o -name FAILED.json -o -name FAILED_SUPERSEDED_BY_RECEIPT_REPAIR.json \) -print
