#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# This script is pinned to the current local Rosetta environment and the
# model/checkpoint layout that already exists under this repository.
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/rosetta/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/recipe/eval_recipe/cachejpeg_rosetta_eval.yaml}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
GPU_IDS="${GPU_IDS:-0}"
ANSWER_METHOD="${ANSWER_METHOD:-generate}"
CHECKPOINT_SUBFOLDER="${CHECKPOINT_SUBFOLDER:-final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/local/final_results/cachejpeg_rosetta_matrix_local}"

LOCAL_MODELS_BASE_DIR="${REPO_ROOT}/local/models"
CHECKPOINTS_BASE_DIR="${REPO_ROOT}/C2C_Fuser"
HOMO_SRC="/data/smy/HomoC2C-KV/src"

DATASETS=(
  "mmlu-redux"
  "ceval"
)

TEACHER_KEYS=(
  "qwen25_math_1p5b"
  "qwen3_4b"
  "qwen3_4b_base"
)

declare -A LOCAL_TEACHER_MODELS=(
  ["qwen25_math_1p5b"]="${LOCAL_MODELS_BASE_DIR}/Qwen2.5-Math-1.5B"
  ["qwen3_4b"]="${LOCAL_MODELS_BASE_DIR}/Qwen3-4B"
  ["qwen3_4b_base"]="${LOCAL_MODELS_BASE_DIR}/Qwen3-4B-Base"
)

declare -A CHECKPOINT_ROOTS=(
  ["qwen25_math_1p5b"]="${CHECKPOINTS_BASE_DIR}/qwen3_0.6b+qwen2.5_1.5b_math_Fuser"
  ["qwen3_4b"]="${CHECKPOINTS_BASE_DIR}/qwen3_0.6b+qwen3_4b_Fuser"
  ["qwen3_4b_base"]="${CHECKPOINTS_BASE_DIR}/qwen3_0.6b+qwen3_4b_base_Fuser"
)

TMP_DIR="${REPO_ROOT}/local/tmp_eval_configs_cachejpeg_rosetta_local"
mkdir -p "${TMP_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found or not executable: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base config not found: ${BASE_CONFIG}"
  exit 1
fi

if [[ ! -d "${HOMO_SRC}" ]]; then
  echo "HomoC2C-KV source path not found: ${HOMO_SRC}"
  exit 1
fi

for teacher_key in "${TEACHER_KEYS[@]}"; do
  teacher_model="${LOCAL_TEACHER_MODELS[$teacher_key]}"
  checkpoint_root="${CHECKPOINT_ROOTS[$teacher_key]}"
  checkpoint_dir="${checkpoint_root}/${CHECKPOINT_SUBFOLDER}"

  if [[ ! -d "${teacher_model}" ]]; then
    echo "Local teacher model directory does not exist for ${teacher_key}: ${teacher_model}"
    exit 1
  fi
  if [[ ! -d "${checkpoint_root}" ]]; then
    echo "Checkpoint root does not exist for ${teacher_key}: ${checkpoint_root}"
    exit 1
  fi
  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "Checkpoint subfolder does not exist for ${teacher_key}: ${checkpoint_dir}"
    exit 1
  fi
done

for teacher_key in "${TEACHER_KEYS[@]}"; do
  teacher_model="${LOCAL_TEACHER_MODELS[$teacher_key]}"
  checkpoint_root="${CHECKPOINT_ROOTS[$teacher_key]}"

  for dataset in "${DATASETS[@]}"; do
    tmp_config="${TMP_DIR}/qwen3-0.6B__${teacher_key}__${dataset}.yaml"
    run_output_dir="${OUTPUT_ROOT}/${teacher_key}/${dataset}"

    "${PYTHON_BIN}" - "${BASE_CONFIG}" "${tmp_config}" "${BASE_MODEL}" "${teacher_model}" "${checkpoint_root}" "${run_output_dir}" "${dataset}" "${GPU_IDS}" "${ANSWER_METHOD}" "${CHECKPOINT_SUBFOLDER}" "${HOMO_SRC}" <<'PY'
import copy
import sys
from pathlib import Path

import yaml

(
    base_config_path,
    output_config_path,
    base_model,
    teacher_model,
    checkpoint_root,
    output_dir,
    dataset,
    gpu_ids_raw,
    answer_method,
    checkpoint_subfolder,
    homo_src,
) = sys.argv[1:]

with open(base_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg = copy.deepcopy(cfg)
cfg.setdefault("model", {})
cfg.setdefault("output", {})
cfg.setdefault("eval", {})
cfg["model"]["model_name"] = "cachejpeg_rosetta"
cfg["model"].setdefault("rosetta_config", {})
cfg["model"]["rosetta_config"]["base_model"] = base_model
cfg["model"]["rosetta_config"]["teacher_model"] = teacher_model
cfg["model"]["rosetta_config"]["checkpoints_dir"] = checkpoint_root
cfg["model"].setdefault("cachejpeg_rosetta_config", {})
cfg["model"]["cachejpeg_rosetta_config"]["homo_c2c_kv_src"] = homo_src
cfg["model"]["cachejpeg_rosetta_config"].setdefault("codec", {})
cfg["model"]["cachejpeg_rosetta_config"]["codec"]["homo_c2c_kv_src"] = homo_src
cfg["output"]["output_dir"] = output_dir
cfg["eval"]["dataset"] = dataset
cfg["eval"]["answer_method"] = answer_method
cfg["eval"]["rosetta_checkpoint_subfolder"] = checkpoint_subfolder
cfg["eval"]["gpu_ids"] = [int(x.strip()) for x in gpu_ids_raw.split(",") if x.strip()]

output_path = Path(output_config_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

    echo "============================================================"
    echo "CacheJPEG-Rosetta evaluation"
    echo "Dataset          : ${dataset}"
    echo "Teacher key      : ${teacher_key}"
    echo "Teacher model    : ${teacher_model}"
    echo "Checkpoint root  : ${checkpoint_root}"
    echo "Checkpoint stage : ${CHECKPOINT_SUBFOLDER}"
    echo "GPU_IDS          : ${GPU_IDS}"
    echo "Output dir       : ${run_output_dir}"
    echo "Config           : ${tmp_config}"
    echo "============================================================"

    (
      cd "${REPO_ROOT}"
      "${PYTHON_BIN}" script/evaluation/unified_evaluator.py --config "${tmp_config}"
    )
  done
done

echo "All local CacheJPEG-Rosetta matrix evaluations finished."
