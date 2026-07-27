#!/usr/bin/env bash
set -euo pipefail

cd /data/smy_data

conda run -n rosetta python script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench_latent_kv_joint_raw_stage1.json

test -f local/checkpoints/0.6+1.5B_latent_kv_joint_raw_longbench_e/final/projector_0.pt

conda run -n rosetta python script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench_latent_kv_joint_adaptive_quant.json
