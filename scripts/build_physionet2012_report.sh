#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${ROOT}/.venv/bin/python"
ARTIFACTS_DIR="${1:-${ROOT}/artifacts/physionet2012_last_run}"

exec "${PYTHON}" -m conceptfan_realdata.cli report \
  --prepared "${ARTIFACTS_DIR}/prepared/prepared_physionet2012.npz" \
  --artifacts-root "${ARTIFACTS_DIR}" \
  --output-dir "${ROOT}/reports/physionet2012"
