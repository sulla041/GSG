"""V0 公式 1：节点特征预编码。"""

import torch
from torch import nn


class ENCS(nn.Module):
    """仅使用 MLP 计算 s(0, i)。"""

    def __init__(self, node_width: int, point_width: int, hidden_width: int) -> None:
        """创建节点升维映射。"""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(node_width, hidden_width), nn.GELU(),
                                 nn.Linear(hidden_width, point_width))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """返回 `[B, V, d]` 初始点集。"""
        if nodes.ndim != 3:
            raise ValueError("nodes must have shape [B, V, d_x]")
        return self.mlp(nodes)
