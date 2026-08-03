#!/usr/bin/env bash
set -euo pipefail

cd /data/smy_data
exec conda run --no-capture-output -n rosetta \
  env PYTHONPATH=/data/smy_data:/data/smy/HomoC2C-KV/src CUDA_VISIBLE_DEVICES=0,1 \
  python -u /data/smy_data/tmp/eval_hotpotqa_e_zero_sharer_kv.py --shard-size 10 \
  > /data/smy_data/tmp/hotpotqa_e_zero_sharer_kv.log 2>&1
