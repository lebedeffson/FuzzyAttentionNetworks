#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${ROOT}/.venv/bin/python"

exec "${PYTHON}" -m conceptfan_realdata.cli full \
  --config "${ROOT}/configs/physionet2012/data.yaml" \
  --report-dir "${ROOT}/reports/physionet2012" \
  "$@"
