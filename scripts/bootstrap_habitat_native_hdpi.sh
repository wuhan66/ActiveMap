#!/usr/bin/env bash
set -euo pipefail

# Install Habitat-Sim into the existing ActiveMap agent environment.  The
# currently published Conda package targets Python 3.9, while this project uses
# Python 3.10, so the supported source-build route keeps one shared environment.
PYTHON="${HABITAT_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
SOURCE_ROOT="${HABITAT_SOURCE_ROOT:-/home/wh/ActiveMap/external/habitat-sim-source}"
DATA_ROOT="${HABITAT_DATA_ROOT:-/home/wh/ActiveMap/datasets/habitat}"
SCENE_UIDS="${HABITAT_SCENE_UIDS:-habitat_test_scenes replica_cad_dataset}"
SOURCE_COMMIT="${HABITAT_SOURCE_COMMIT:-unknown}"

[[ -x "${PYTHON}" ]] || {
  echo "ActiveMap Python executable not found: ${PYTHON}" >&2
  exit 1
}

if ! "${PYTHON}" -c 'import habitat_sim' 2>/dev/null; then
  if [[ ! -f "${SOURCE_ROOT}/pyproject.toml" ]]; then
    git clone --depth 1 --recurse-submodules \
      https://github.com/facebookresearch/habitat-sim.git "${SOURCE_ROOT}"
  fi
  "${PYTHON}" -m pip install --upgrade \
    'scikit-build-core>=0.10' 'pybind11>=2.10' 'numpy-quaternion>=2024.0.0'
  (
    cd "${SOURCE_ROOT}"
    HABITAT_BUILD_GUI_VIEWERS=OFF \
    HABITAT_WITH_AUDIO=OFF \
    HABITAT_WITH_CUDA=OFF \
      "${PYTHON}" -m pip install . --no-build-isolation
  )
fi

"${PYTHON}" - <<'PY'
import habitat_sim

print(f"habitat-sim import passed: {habitat_sim.__version__}")
PY

mkdir -p "${DATA_ROOT}"
"${PYTHON}" -m habitat_sim.utils.datasets_download \
  --uids ${SCENE_UIDS} \
  --data-path "${DATA_ROOT}"

if git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  SOURCE_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
fi
cat >"${DATA_ROOT}/ACTIVEMAP_BOOTSTRAP_COMPLETE.json" <<EOF
{
  "schema_version": "activemap-habitat-bootstrap-v1",
  "python": "${PYTHON}",
  "habitat_sim_commit": "${SOURCE_COMMIT}",
  "scene_uids": "${SCENE_UIDS}",
  "role": "indoor_rgbd_navigation_development",
  "test_assets_read": false
}
EOF

echo "Native Habitat bootstrap complete: ${SOURCE_ROOT}"
