#!/usr/bin/env bash
set -euo pipefail

cd /home/limingyan/C2C-compress
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# conda run --no-capture-output -n rosetta torchrun \
#   --nproc_per_node=2 \
#   --master_port=29511 \
#   script/train/SFT_train.py \
#   --config recipe/train_recipe/C2C_longbench_latent_kv_split_raw_stage1.json

# test -f local/checkpoints/0.6+1.5B_latent_kv_split_raw_longbench_e/final/projector_0.pt

conda run --no-capture-output -n rosetta torchrun \
  --nproc_per_node=2 \
  --master_port=29512 \
  script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench_latent_kv_split_adaptive_quant.json

# CUDA_VISIBLE_DEVICES=4 python script/train/SFT_train.py \
#   --config recipe/train_recipe/C2C_longbench_latent_kv_split_adaptive_quant.json