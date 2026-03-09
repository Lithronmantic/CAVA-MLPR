#!/usr/bin/env bash
set -euo pipefail

# Multi-seed launcher using Multi-GPU safe entrypoint.
# Usage:
#   bash scripts/run_paper_exact_multi_seed_4gpu_safe.sh [RUNTIME_BS] [SEEDS] [CONFIG]

RUNTIME_BS="${1:-32}"
SEEDS="${2:-42 52 62}"
CONFIG_PATH="${3:-configs/paper_exact_linux.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs}"

export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[MULTI_SEED_SAFE] runtime_batch_size=${RUNTIME_BS}"
echo "[MULTI_SEED_SAFE] seeds=${SEEDS}"
echo "[MULTI_SEED_SAFE] config=${CONFIG_PATH}"
echo "[MULTI_SEED_SAFE] out_root=${OUT_ROOT}"

for SEED in ${SEEDS}; do
  OUT_DIR="${OUT_ROOT}/paper_exact_seed${SEED}_4gpu_safe"
  echo "============================================================"
  echo "[MULTI_SEED_SAFE] start seed=${SEED} -> ${OUT_DIR}"
  echo "============================================================"

  python -u scripts/train_enhanced_vis_multi_gpu_safe.py \
    --config "${CONFIG_PATH}" \
    --output "${OUT_DIR}" \
    --multi_gpu \
    --num_gpus 4 \
    --runtime_batch_size "${RUNTIME_BS}" \
    --seed "${SEED}"
done

echo "[MULTI_SEED_SAFE] all runs finished."

