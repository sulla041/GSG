#!/usr/bin/env bash
# 意涵 32-边平衡分层电表网络 (Hierarchical Meter Network) 专精启动脚本
# 架构特性：[Cell -> L1 电表(<=32节点) -> Root 电表(<=32节点)，入度恒定 O(1)]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data/yiyh/dac2027/envs/v2ex_cuda/bin/python3"
GPU_ID="${1:-0}"
STEPS="${2:-3000}"
LR="${3:-1e-3}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUTPUT_DIR="/data/yiyh/dac2027/runs/v3_04_p2_hierarchical_meter_${STEPS}step"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "🌟 启动 32-边平衡分层电表网络 (Hierarchical Meter) 大实验 🌟"
echo "  - GPU: CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "  - 意涵架构: 32-叉规约金字塔 (入度严格 O(1)，无单点爆炸)"
echo "  - 任务类型: 图级总面积专精预测 (Total Area, um^2)"
echo "  - 训练步数: ${STEPS} 步, 学习率: ${LR}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "=========================================================="

exec "${PYTHON_BIN}" -u script/train_v3_04_p2_hierarchical_meter.py \
  --output "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
