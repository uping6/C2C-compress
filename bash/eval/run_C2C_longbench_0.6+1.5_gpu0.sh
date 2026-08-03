#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVALUATOR="${REPO_ROOT}/script/evaluation/unified_evaluator.py"

CONFIGS=(
  "${REPO_ROOT}/recipe/eval_recipe/C2C_longbench_0.6+1.5_rosetta_gpu0.yaml"
  "${REPO_ROOT}/recipe/eval_recipe/C2C_longbench_0.6+1.5_jpegcache_rosetta_gpu0.yaml"
  "${REPO_ROOT}/recipe/eval_recipe/C2C_longbench_0.6+1.5_receiver_only_gpu0.yaml"
)

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Config not found: ${config}" >&2
    exit 1
  fi
done

echo "Running 3 LongBench-E experiments sequentially on GPU 0"
for config in "${CONFIGS[@]}"; do
  echo
  echo "===== $(basename "${config}") ====="
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${EVALUATOR}" --config "${config}"
done

echo
echo "All three experiments completed."
