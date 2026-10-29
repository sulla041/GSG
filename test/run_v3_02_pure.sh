#!/bin/bash
# V3.02 独立版本：8 层、1000 步、纯净动力学（sum 聚合、无 eta）训练脚本
set -e
cd /home/yiyh/dac2027
PYTHONPATH=. /data/yiyh/dac2027/envs/v2ex_cuda/bin/python test/run_v3_02_hop_training.py \
    --output /tmp/v3_02_hops_t8_pure \
    --dynamics-steps 8 \
    --train-steps 1000
