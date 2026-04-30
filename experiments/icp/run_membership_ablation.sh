#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-swat}"
DATA_DIR="${2:-/home/lebedeffson/Code/андрей/archive_preview/ICP}"
OUT_ROOT="${3:-results/icp_membership_ablation}"
PYTHON_BIN="${PYTHON_BIN:-/home/lebedeffson/Code/venv/bin/python}"

for membership in gaussian bell sigmoid mixed; do
  "${PYTHON_BIN}" experiments/icp/run_review_experiments.py \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --models fan \
    --membership "${membership}" \
    --seeds 42 43 44 \
    --epochs 15 \
    --batch-size 512 \
    --out-dir "${OUT_ROOT}/${DATASET}_${membership}"
done
