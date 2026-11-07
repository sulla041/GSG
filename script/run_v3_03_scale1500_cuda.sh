#!/usr/bin/env bash
# V3.03 1500-Design 工业级全量大实验启动脚本
# 意涵专属启动脚本：[1500设计 / 4096节点 / T=64, 3000步长程大练兵]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data/yiyh/dac2027/envs/v2ex_cuda/bin/python3"
OUTPUT_DIR="/data/yiyh/dac2027/runs/v3_03_scale1500_T64_3000step"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "🌟 启动 V3.03 1500-Design 工业级全量大实验 (T=64, 3000步) 🌟"
echo "  - GPU: CUDA_VISIBLE_DEVICES=${GPU_ID} (NVIDIA H100)"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - 数据底座: CC-Tensor v3 1500_4096_v2 (1200训练图 100% 常驻显存)"
echo "  - 核心架构: 6大等宽平衡通道 (24维) + 关系精炼双通道 (12维)"
echo "  - 实验规模: dynamics_steps=64, steps=3000, max_nodes=4096"
echo "=========================================================="

exec "${PYTHON_BIN}" -u script/train_v3_03_compact.py \
  --output "${OUTPUT_DIR}" \
  --max-target-cap 1.5 \
  --steps 3000 \
  --dynamics-steps 64 \
  --learning-rate 1e-3 \
  --eval-every 100 \
  --seed 20261103 \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
