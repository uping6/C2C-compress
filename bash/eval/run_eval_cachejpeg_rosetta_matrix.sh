#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/recipe/eval_recipe/cachejpeg_rosetta_eval.yaml}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
GPU_IDS="${GPU_IDS:-0}"
ANSWER_METHOD="${ANSWER_METHOD:-generate}"
CHECKPOINT_SUBFOLDER="${CHECKPOINT_SUBFOLDER:-final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-local/final_results/cachejpeg_rosetta_matrix}"
CHECKPOINTS_BASE_DIR="${CHECKPOINTS_BASE_DIR:-${REPO_ROOT}/C2C_Fuser}"
USE_LOCAL_TEACHERS="${USE_LOCAL_TEACHERS:-1}"
LOCAL_MODELS_BASE_DIR="${LOCAL_MODELS_BASE_DIR:-${REPO_ROOT}/local/models}"

DATASETS=(
  "mmlu-redux"
  "ceval"
)

TEACHER_KEYS=(
  "qwen25_math_1p5b"
  "qwen3_4b"
  "qwen3_4b_base"
)

declare -A TEACHER_MODELS=(
  ["qwen25_math_1p5b"]="Qwen/Qwen2.5-Math-1.5B"
  ["qwen3_4b"]="Qwen/Qwen3-4B"
  ["qwen3_4b_base"]="Qwen/Qwen3-4B-Base"
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

TMP_DIR="${REPO_ROOT}/local/tmp_eval_configs_cachejpeg_rosetta"
mkdir -p "${TMP_DIR}"

for teacher_key in "${TEACHER_KEYS[@]}"; do
  teacher_model="${TEACHER_MODELS[$teacher_key]}"
  if [[ "${USE_LOCAL_TEACHERS}" == "1" ]]; then
    teacher_model="${LOCAL_TEACHER_MODELS[$teacher_key]}"
  fi
  checkpoint_root="${CHECKPOINT_ROOTS[$teacher_key]}"

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
cfg["model"]["model_name"] = "cachejpeg_rosetta"
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

    (
      cd "${REPO_ROOT}"
      python script/evaluation/unified_evaluator.py --config "${tmp_config}"
    )
  done
done
