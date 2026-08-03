#!/usr/bin/env bash
set -euo pipefail

# 固定到项目根目录，避免后台启动时 cwd 不一致。
ROOT_DIR="/data/smy/lmy/C2C-compress-master"
cd "${ROOT_DIR}"

# 尽量从“干净”的 conda 状态切到 rosetta，避免从别的 env 嵌套进来时残留 Python 状态。
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
  conda deactivate || break
done
conda activate rosetta

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_20260709
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TORCH_DISABLE_DYNAMO=1

# 避免后台任务继承到异常代理配置，当前训练所需模型和数据都已在本地。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONPYCACHEPREFIX

mkdir -p "${HF_DATASETS_CACHE}"
MASTER_PORT="${MASTER_PORT:-29515}"

exec python -B -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_longbench.json \
  --log_dir ./longbench/logs \
  --log_name train_0.6_0.5_longbench_resume_from300_dualgpu_20260709_script.log \
  --resume_from_checkpoint /data/smy/lmy/C2C-compress-master/local/checkpoints/0.6+0.5B_C2C_longbench_e/checkpoint-300
