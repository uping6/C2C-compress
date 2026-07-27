#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/recipe/eval_recipe/unified_eval.yaml}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
GPU_IDS="${GPU_IDS:-1}"
ANSWER_METHOD="${ANSWER_METHOD:-generate}"
CHECKPOINT_SUBFOLDER="${CHECKPOINT_SUBFOLDER:-final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-local/final_results/qwen3_0.6b_eval_matrix}"
CHECKPOINTS_BASE_DIR="${CHECKPOINTS_BASE_DIR:-${REPO_ROOT}/C2C_Fuser}"

# Only teacher models use local directories.
USE_LOCAL_TEACHERS="${USE_LOCAL_TEACHERS:-1}"
LOCAL_MODELS_BASE_DIR="${LOCAL_MODELS_BASE_DIR:-${REPO_ROOT}/local/models}"

DATASETS=(
  # "mmlu-redux"
  "openbookqa"
  # "ai2-arc"
  "ceval"
)

TEACHER_KEYS=(
  # "llama32_1b_instruct"
  "qwen25_math_1p5b"
  "qwen3_4b"
  "qwen3_4b_base"
)

declare -A TEACHER_MODELS=(
#   ["llama32_1b_instruct"]="meta-llama/Llama-3.2-1B-Instruct"
  ["qwen25_math_1p5b"]="Qwen/Qwen2.5-Math-1.5B"
  ["qwen3_4b"]="Qwen/Qwen3-4B"
  ["qwen3_4b_base"]="Qwen/Qwen3-4B-Base"
)

declare -A LOCAL_TEACHER_MODELS=(
  #   ["llama32_1b_instruct"]="${LOCAL_MODELS_BASE_DIR}/Llama-3.2-1B-Instruct"
  ["qwen25_math_1p5b"]="${LOCAL_MODELS_BASE_DIR}/Qwen2.5-Math-1.5B"
  ["qwen3_4b"]="${LOCAL_MODELS_BASE_DIR}/Qwen3-4B"
  ["qwen3_4b_base"]="${LOCAL_MODELS_BASE_DIR}/Qwen3-4B-Base"
)

# Rosetta projector checkpoint roots.
# Each root is expected to contain the subfolder selected by CHECKPOINT_SUBFOLDER
# such as `${CHECKPOINT_ROOTS[key]}/final`.
declare -A CHECKPOINT_ROOTS=(
#   ["llama32_1b_instruct"]="${CHECKPOINTS_BASE_DIR}/qwen3_0.6b+llam3.2_1b_Fuser"
  ["qwen25_math_1p5b"]="${CHECKPOINTS_BASE_DIR}/qwen3_0.6b+qwen2.5_1.5b_math_Fuser"
  ["qwen3_4b"]="${CHECKPOINTS_BASE_DIR}/qwen3_0.6b+qwen3_4b_Fuser"
  ["qwen3_4b_base"]="${CHECKPOINTS_BASE_DIR}/qwen3_0.6b+qwen3_4b_base_Fuser"
)

TMP_DIR="${REPO_ROOT}/local/tmp_eval_configs"
mkdir -p "${TMP_DIR}"

# if [[ "${USE_LOCAL_MODELS}" == "1" ]]; then
#   BASE_MODEL="${LOCAL_MODELS_BASE_DIR}/Qwen3-0.6B"
#   if [[ ! -d "${BASE_MODEL}" ]]; then
#     echo "Local base model directory does not exist: ${BASE_MODEL}"
#     echo "Please check LOCAL_MODELS_BASE_DIR or disable USE_LOCAL_MODELS."
#     exit 1
#   fi
# fi
# Base model always uses Hugging Face model id by default.
# It will be downloaded/cached automatically by transformers if not available locally.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"

# Teacher models can use local paths when USE_LOCAL_TEACHERS=1.
USE_LOCAL_TEACHERS="${USE_LOCAL_TEACHERS:-1}"
LOCAL_MODELS_BASE_DIR="${LOCAL_MODELS_BASE_DIR:-${REPO_ROOT}/local/models}"
for teacher_key in "${TEACHER_KEYS[@]}"; do
  if [[ -z "${TEACHER_MODELS[$teacher_key]+x}" ]]; then
    echo "Missing TEACHER_MODELS entry for teacher_key: ${teacher_key}"
    exit 1
  fi

  if [[ -z "${CHECKPOINT_ROOTS[$teacher_key]+x}" ]]; then
    echo "Missing CHECKPOINT_ROOTS entry for teacher_key: ${teacher_key}"
    exit 1
  fi

  teacher_model="${TEACHER_MODELS[$teacher_key]}"

  if [[ "${USE_LOCAL_TEACHERS}" == "1" ]]; then
    if [[ -z "${LOCAL_TEACHER_MODELS[$teacher_key]+x}" ]]; then
      echo "Missing LOCAL_TEACHER_MODELS entry for teacher_key: ${teacher_key}"
      exit 1
    fi

    teacher_model="${LOCAL_TEACHER_MODELS[$teacher_key]}"

    if [[ ! -d "${teacher_model}" ]]; then
      echo "Local teacher model directory does not exist for ${teacher_key}: ${teacher_model}"
      echo "Please check LOCAL_MODELS_BASE_DIR or disable USE_LOCAL_TEACHERS."
      exit 1
    fi
  fi

  checkpoint_root="${CHECKPOINT_ROOTS[$teacher_key]}"

  if [[ ! -d "${checkpoint_root}" ]]; then
    echo "Checkpoint root for ${teacher_key} does not exist: ${checkpoint_root}"
    echo "Please check CHECKPOINTS_BASE_DIR or the checkpoint folder name."
    exit 1
  fi

  for dataset in "${DATASETS[@]}"; do
    tmp_config="${TMP_DIR}/$(basename "${BASE_MODEL}")__${teacher_key}__${dataset}.yaml"
    run_output_dir="${OUTPUT_ROOT}/${teacher_key}/${dataset}"

    python - "${BASE_CONFIG}" "${tmp_config}" "${BASE_MODEL}" "${teacher_model}" "${checkpoint_root}" "${run_output_dir}" "${dataset}" "${GPU_IDS}" "${ANSWER_METHOD}" "${CHECKPOINT_SUBFOLDER}" <<'PY'
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
) = sys.argv[1:]

with open(base_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg = copy.deepcopy(cfg)
cfg.setdefault("model", {})
cfg.setdefault("output", {})
cfg.setdefault("eval", {})
cfg["model"]["model_name"] = "Rosetta"
cfg["model"].setdefault("rosetta_config", {})
cfg["model"]["rosetta_config"]["base_model"] = base_model
cfg["model"]["rosetta_config"]["teacher_model"] = teacher_model
cfg["model"]["rosetta_config"]["checkpoints_dir"] = checkpoint_root
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
    echo "Running dataset=${dataset}"
    echo "Teacher=${teacher_model}"
    echo "Checkpoint root=${checkpoint_root}"
    echo "Output dir=${run_output_dir}"
    echo "Config=${tmp_config}"
    echo "============================================================"

    (
      cd "${REPO_ROOT}"
      python script/evaluation/unified_evaluator.py --config "${tmp_config}"
    )
  done
done

echo "All matrix evaluations finished."
