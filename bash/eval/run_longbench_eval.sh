#!/usr/bin/env bash
set -euo pipefail

ARGS=(--config recipe/eval_recipe/longbench_jpegcache.yaml)
if [[ -n "${CACHEJPEG_ENTROPY_BACKEND:-}" ]]; then
  ARGS+=(--cachejpeg-entropy-backend "${CACHEJPEG_ENTROPY_BACKEND}")
fi
if [[ -n "${CACHEJPEG_BANDWIDTH_MBPS:-}" ]]; then
  ARGS+=(--cachejpeg-bandwidth-mbps "${CACHEJPEG_BANDWIDTH_MBPS}")
fi

# Default to physical GPU1. Override with CACHEJPEG_CUDA_VISIBLE_DEVICES if needed.
CUDA_VISIBLE_DEVICES="${CACHEJPEG_CUDA_VISIBLE_DEVICES:-1}" \
  python script/evaluation/unified_evaluator.py "${ARGS[@]}"
