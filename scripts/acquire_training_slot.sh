#!/usr/bin/env bash
# Source from every GPU training launcher. The lock descriptor is inherited by
# the launched process, so the slot remains held after a nohup launcher exits.

if [[ "${ACTIVEMAP_TRAINING_SLOT_HELD:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

ACTIVEMAP_STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
slot_dir="${ACTIVEMAP_STORAGE_ROOT}/run_control"
slot_lock="${slot_dir}/training.lock"
slot_owner="${slot_dir}/training_owner.tsv"
mkdir -p "${slot_dir}"

exec 7>"${slot_lock}"
if ! flock -n 7; then
  echo "Another ActiveMap training experiment holds ${slot_lock}" >&2
  if [[ -s "${slot_owner}" ]]; then
    cat "${slot_owner}" >&2
  fi
  return 6 2>/dev/null || exit 6
fi

printf 'claimed_at\t%s\npid\t%s\nlauncher\t%s\n' \
  "$(date --iso-8601=seconds)" "$$" "${ACTIVEMAP_LAUNCHER_NAME:-${0##*/}}" \
  >"${slot_owner}"
export ACTIVEMAP_TRAINING_SLOT_HELD=1
