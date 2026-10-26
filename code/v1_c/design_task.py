"""从终态点集读取一个设计级标量任务。"""

import torch
from torch import nn

from .setattn import SETATTN


class DesignTaskHead(nn.Module):
    """使用 masked set pooling 输出一个标量。"""

    def __init__(self, width: int, pair_width: int, hidden_width: int) -> None:
        """创建集合上下文和标量回归头。"""
        super().__init__()
        self.context = SETATTN(width, pair_width, hidden_width)
        self.head = nn.Sequential(nn.Linear(width, hidden_width), nn.GELU(),
                                  nn.Linear(hidden_width, 1))

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """返回 `[B]` 设计任务预测。"""
        context = self.context(points, node_mask)
        weights = node_mask.unsqueeze(-1).to(context.dtype)
        pooled = (context * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)
