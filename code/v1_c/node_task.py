"""从终态点集读取每个节点的一个连续任务。"""

import torch
from torch import nn

from .setattn import SETATTN


class NodeTaskHead(nn.Module):
    """使用集合上下文输出 `[B, V]` 节点标量预测。"""

    def __init__(self, width: int, pair_width: int, hidden_width: int) -> None:
        """创建节点级回归头。"""
        super().__init__()
        self.context = SETATTN(width, pair_width, hidden_width)
        self.head = nn.Sequential(nn.Linear(width, hidden_width), nn.GELU(),
                                  nn.Linear(hidden_width, 1))

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """返回每个有效节点的标量预测。"""
        return self.head(self.context(points, node_mask)).squeeze(-1)
