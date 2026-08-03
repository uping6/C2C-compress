#!/usr/bin/env bash
set -euo pipefail

ARGS=(
  --config recipe/eval_recipe/longbench_jpegcache_stream_50mbps.yaml
  --cachejpeg-bandwidth-mbps "${CACHEJPEG_BANDWIDTH_MBPS:-50}"
)
if [[ -n "${CACHEJPEG_ENTROPY_BACKEND:-}" ]]; then
  ARGS+=(--cachejpeg-entropy-backend "${CACHEJPEG_ENTROPY_BACKEND}")
fi

python script/evaluation/unified_evaluator.py "${ARGS[@]}"
