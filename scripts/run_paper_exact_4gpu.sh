#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_paper_exact_4gpu.sh [OUTPUT_DIR] [RUNTIME_BS]
#
# Example:
#   bash scripts/run_paper_exact_4gpu.sh outputs/paper_exact_seed42_4gpu 8

OUTPUT_DIR="${1:-outputs/paper_exact_seed42_4gpu}"
RUNTIME_BS="${2:-8}"

export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u scripts/train_enhanced_vis.py \
  --config configs/paper_exact.yaml \
  --output "${OUTPUT_DIR}" \
  --multi_gpu \
  --num_gpus 4 \
  --runtime_batch_size "${RUNTIME_BS}"

