#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Fixed one-pair CacheJPEG-Rosetta evaluation.
# Receiver/base receives the fused cache; sharer/teacher produces the cache to transmit.
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/rosetta/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/recipe/eval_recipe/cachejpeg_rosetta_eval.yaml}"

RECEIVER_BASE_MODEL="Qwen/Qwen3-0.6B"
SHARER_TEACHER_MODEL="${REPO_ROOT}/local/models/Qwen3-4B"
CHECKPOINT_ROOT="${REPO_ROOT}/C2C_Fuser/qwen3_0.6b+qwen3_4b_Fuser"
CHECKPOINT_SUBFOLDER="final"
DATASET="ceval"

GPU_IDS="${GPU_IDS:-0}"
ANSWER_METHOD="generate"
HOMO_C2C_KV_SRC="/data/smy/HomoC2C-KV/src"
OUTPUT_DIR="${REPO_ROOT}/local/final_results/cachejpeg_rosetta_single/qwen3_0.6b__qwen3_4b/${DATASET}"
TMP_DIR="${REPO_ROOT}/local/tmp_eval_configs_cachejpeg_rosetta_single"
TMP_CONFIG="${TMP_DIR}/qwen3_0.6b__qwen3_4b__${DATASET}.yaml"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found or not executable: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base config not found: ${BASE_CONFIG}"
  exit 1
fi

if [[ ! -d "${SHARER_TEACHER_MODEL}" ]]; then
  echo "Sharer/teacher model directory not found: ${SHARER_TEACHER_MODEL}"
  exit 1
fi

if [[ ! -d "${CHECKPOINT_ROOT}/${CHECKPOINT_SUBFOLDER}" ]]; then
  echo "Fuser checkpoint directory not found: ${CHECKPOINT_ROOT}/${CHECKPOINT_SUBFOLDER}"
  exit 1
fi

if [[ ! -d "${HOMO_C2C_KV_SRC}" ]]; then
  echo "HomoC2C-KV source path not found: ${HOMO_C2C_KV_SRC}"
  exit 1
fi

mkdir -p "${TMP_DIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${TMP_CONFIG}" "${RECEIVER_BASE_MODEL}" "${SHARER_TEACHER_MODEL}" "${CHECKPOINT_ROOT}" "${CHECKPOINT_SUBFOLDER}" "${DATASET}" "${GPU_IDS}" "${ANSWER_METHOD}" "${HOMO_C2C_KV_SRC}" "${OUTPUT_DIR}" <<'PY'
import copy
import sys
from pathlib import Path

import yaml

(
    base_config_path,
    output_config_path,
    receiver_base_model,
    sharer_teacher_model,
    checkpoint_root,
    checkpoint_subfolder,
    dataset,
    gpu_ids_raw,
    answer_method,
    homo_src,
    output_dir,
) = sys.argv[1:]

with open(base_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg = copy.deepcopy(cfg)
cfg.setdefault("model", {})
cfg.setdefault("output", {})
cfg.setdefault("eval", {})

cfg["model"]["model_name"] = "cachejpeg_rosetta"
cfg["model"].setdefault("rosetta_config", {})
cfg["model"]["rosetta_config"]["base_model"] = receiver_base_model
cfg["model"]["rosetta_config"]["teacher_model"] = sharer_teacher_model
cfg["model"]["rosetta_config"]["checkpoints_dir"] = checkpoint_root

cfg["model"].setdefault("cachejpeg_rosetta_config", {})
cfg["model"]["cachejpeg_rosetta_config"]["sharer_model_role"] = "teacher"
cfg["model"]["cachejpeg_rosetta_config"]["receiver_model_role"] = "base"
cfg["model"]["cachejpeg_rosetta_config"]["homo_c2c_kv_src"] = homo_src
cfg["model"]["cachejpeg_rosetta_config"].setdefault("codec", {})
cfg["model"]["cachejpeg_rosetta_config"]["codec"]["method"] = "cachejpeg"
cfg["model"]["cachejpeg_rosetta_config"]["codec"]["homo_c2c_kv_src"] = homo_src

cfg.setdefault("output", {})
cfg["output"]["output_dir"] = output_dir

cfg["eval"]["dataset"] = dataset
cfg["eval"]["gpu_ids"] = [int(x.strip()) for x in gpu_ids_raw.split(",") if x.strip()]
cfg["eval"]["answer_method"] = answer_method
cfg["eval"]["rosetta_checkpoint_subfolder"] = checkpoint_subfolder

output_path = Path(output_config_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

echo "============================================================"
echo "Fixed CacheJPEG-Rosetta one-pair evaluation"
echo "Receiver/base       : ${RECEIVER_BASE_MODEL}"
echo "Sharer/teacher      : ${SHARER_TEACHER_MODEL}"
echo "Dataset             : ${DATASET}"
echo "Fuser checkpoint    : ${CHECKPOINT_ROOT}/${CHECKPOINT_SUBFOLDER}"
echo "GPU_IDS             : ${GPU_IDS}"
echo "Generated config    : ${TMP_CONFIG}"
echo "Output dir          : ${OUTPUT_DIR}"
echo "============================================================"

(
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" script/evaluation/unified_evaluator.py --config "${TMP_CONFIG}"
)
