#!/usr/bin/env bash
set -euo pipefail

cd /data/smy_data
exec conda run --no-capture-output -n rosetta \
  env PYTHONPATH=/data/smy_data:/data/smy/HomoC2C-KV/src \
  python -u /data/smy_data/tmp/run_longbench_e_receiver_only.py \
  2>&1 | tee /data/smy_data/tmp/longbench_e_receiver_only.log
