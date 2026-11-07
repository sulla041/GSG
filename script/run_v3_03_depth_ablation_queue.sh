#!/usr/bin/env bash
# V3.03 动力学深度 T 层数系统性消融流水线 (T in 8, 16, 24, 32, 48, 64, 96)
# 意涵专属启动脚本：[探究波前射程对端到端因果时序的本质影响]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data/yiyh/dac2027/envs/v2ex_cuda/bin/python3"
BASE_DIR="/data/yiyh/dac2027/runs/ablation_depth_cc_tensor"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

DEPTH_LIST=(8 16 24 32 48 64 96)
STEPS=600

echo "=========================================================="
echo "🌟 启动 V3.03 动力学层数 T 系统性消融大实验 🌟"
echo "  - 设备: GPU ${GPU_ID} (NVIDIA H100)"
echo "  - 数据底座: CC-Tensor v3 (100% 显存常驻)"
echo "  - 消融梯度: T = (${DEPTH_LIST[*]})"
echo "  - 统一训练步数: ${STEPS} 步 (Cosine 退火)"
echo "=========================================================="

mkdir -p "${BASE_DIR}"

for T in "${DEPTH_LIST[@]}"; do
  EXP_DIR="${BASE_DIR}/T${T}"
  mkdir -p "${EXP_DIR}"
  echo -e "\n----------------------------------------------------------"
  echo "🚀 开始消融实验: 动力学深度 T=${T} (steps=${STEPS})"
  echo "----------------------------------------------------------"
  "${PYTHON_BIN}" -u script/train_v3_03_compact.py \
    --output "${EXP_DIR}" \
    --max-target-cap 1.0 \
    --steps "${STEPS}" \
    --dynamics-steps "${T}" \
    --learning-rate 1e-3 \
    --eval-every 50 \
    --seed 20261103 \
    2>&1 | tee "${EXP_DIR}/train.log"
  echo "✅ T=${T} 消融实验完成！"
done

echo -e "\n🎉 全部动力学深度消融实验圆满完成！输出目录: ${BASE_DIR}"
