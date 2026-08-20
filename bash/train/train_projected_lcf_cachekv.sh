#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1 python script/train/SFT_train.py   --config recipe/train_recipe/C2C_openhermes_50k_concat_lcf_projected_kv_raw.json
CUDA_VISIBLE_DEVICES=1 python script/train/SFT_train.py   --config recipe/train_recipe/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant.json