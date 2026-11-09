#!/usr/bin/env bash
# 意涵 32-边平衡分层电表网络 (Hierarchical Meter) 全图关键路径延迟 (Critical Path Delay) 专精启动脚本
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data/yiyh/dac2027/envs/v2ex_cuda/bin/python3"
GPU_ID="${1:-0}"
STEPS="${2:-3000}"
LR="${3:-1e-3}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUTPUT_DIR="/data/yiyh/dac2027/runs/v3_04_p2_hierarchical_critical_timing_${STEPS}step"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "🌟 启动 32-边平衡分层电表 (Critical Path Delay) 大实验 🌟"
echo "  - GPU: CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "  - 意涵架构: 32-叉规约金字塔 (全图时序端点自底向上求 Max)"
echo "  - 任务类型: 芯片最高主频极限 (Critical Path Max Delay, ps)"
echo "  - 训练步数: ${STEPS} 步, 学习率: ${LR}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "=========================================================="

exec "${PYTHON_BIN}" -u script/train_v3_04_p2_critical_timing.py \
  --output "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
