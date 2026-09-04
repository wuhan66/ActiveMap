#!/usr/bin/env bash

activemap_assert_allowed_gpu() {
  local gpu="${1:?GPU id is required}"
  local allowed_ids="${ACTIVEMAP_GPU_IDS:-0,1}"
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "Refusing invalid GPU id: ${gpu}" >&2
    return 2
  }
  [[ ",${allowed_ids}," == *",${gpu},"* ]] || {
    echo "Refusing GPU ${gpu}: outside ACTIVEMAP_GPU_IDS=${allowed_ids}" >&2
    return 2
  }
}
