#!/usr/bin/env bash
set -euo pipefail

# Multi-seed launcher for paper_exact on 4 GPUs (sequential runs).
#
# Usage:
#   bash scripts/run_paper_exact_multi_seed_4gpu.sh [RUNTIME_BS] [SEEDS]
#
# Examples:
#   bash scripts/run_paper_exact_multi_seed_4gpu.sh
#   bash scripts/run_paper_exact_multi_seed_4gpu.sh 8 "42 52 62"
#   bash scripts/run_paper_exact_multi_seed_4gpu.sh 6 "42 52 62 72 82"

RUNTIME_BS="${1:-8}"
SEEDS="${2:-42 52 62}"
OUT_ROOT="${OUT_ROOT:-outputs}"

export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[MULTI_SEED] runtime_batch_size=${RUNTIME_BS}"
echo "[MULTI_SEED] seeds=${SEEDS}"
echo "[MULTI_SEED] out_root=${OUT_ROOT}"

for SEED in ${SEEDS}; do
  OUT_DIR="${OUT_ROOT}/paper_exact_seed${SEED}_4gpu"
  echo "============================================================"
  echo "[MULTI_SEED] start seed=${SEED} -> ${OUT_DIR}"
  echo "============================================================"

  python -u scripts/train_enhanced_vis.py \
    --config configs/paper_exact.yaml \
    --output "${OUT_DIR}" \
    --multi_gpu \
    --num_gpus 4 \
    --runtime_batch_size "${RUNTIME_BS}" \
    --seed "${SEED}"

  echo "[MULTI_SEED] done seed=${SEED}"
done

echo "[MULTI_SEED] all runs finished."

