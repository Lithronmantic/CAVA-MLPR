#!/usr/bin/env bash
set -euo pipefail

RUNTIME_BS="${1:-32}"
SEEDS="${2:-42 52 62}"
CONFIG_PATH="${3:-configs/paper_exact_linux.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs}"

export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[MULTI_SEED_SAFE_V2] runtime_batch_size=${RUNTIME_BS}"
echo "[MULTI_SEED_SAFE_V2] seeds=${SEEDS}"
echo "[MULTI_SEED_SAFE_V2] config=${CONFIG_PATH}"
echo "[MULTI_SEED_SAFE_V2] out_root=${OUT_ROOT}"

for SEED in ${SEEDS}; do
  OUT_DIR="${OUT_ROOT}/paper_exact_seed${SEED}_4gpu_safe_v2"
  echo "============================================================"
  echo "[MULTI_SEED_SAFE_V2] start seed=${SEED} -> ${OUT_DIR}"
  echo "============================================================"
  python -u scripts/train_enhanced_vis_multi_gpu_safe_v2.py \
    --config "${CONFIG_PATH}" \
    --output "${OUT_DIR}" \
    --multi_gpu \
    --num_gpus 4 \
    --runtime_batch_size "${RUNTIME_BS}" \
    --seed "${SEED}"
done

echo "[MULTI_SEED_SAFE_V2] all runs finished."

