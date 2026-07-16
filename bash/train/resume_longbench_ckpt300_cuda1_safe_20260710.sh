#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data/smy/lmy/C2C-compress-master"
cd "${ROOT_DIR}"

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
  conda deactivate || break
done
conda activate rosetta

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_longbench_resume300_cuda1_20260710
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TORCH_DISABLE_DYNAMO=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONPYCACHEPREFIX

mkdir -p "${HF_DATASETS_CACHE}" ./longbench/logs

exec python -B script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench_resume_ckpt300_cuda1_safe.json \
  --log_dir ./longbench/logs \
  --log_name train_0.6_0.5_longbench_resume_from300_cuda1_safe_20260710.log \
  --resume_from_checkpoint /data/smy/lmy/C2C-compress-master/local/checkpoints/0.6+0.5B_C2C_longbench_e/checkpoint-300
