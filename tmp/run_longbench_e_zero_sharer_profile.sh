#!/usr/bin/env bash
set -euo pipefail

cd /data/smy_data
exec conda run --no-capture-output -n rosetta \
  env PYTHONPATH=/data/smy_data:/data/smy/HomoC2C-KV/src \
  python -u /data/smy_data/tmp/run_longbench_e_zero_sharer_profile.py \
  2>&1 | tee /data/smy_data/tmp/longbench_e_cachejpeg_rosetta_zero_sharer_profile.log
