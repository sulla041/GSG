"""V0 公式 1：关系特征预编码。"""

import torch
from torch import nn


class ENCR(nn.Module):
    """仅使用 MLP 计算 re(j, i)。"""

    def __init__(self, relation_width: int, pair_width: int, hidden_width: int) -> None:
        """创建关系升维映射。"""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(relation_width, hidden_width), nn.GELU(),
                                 nn.Linear(hidden_width, pair_width))

    def forward(self, relations: torch.Tensor) -> torch.Tensor:
        """返回 `[B, V, V, d_p]` 关系状态。"""
        if relations.ndim != 4:
            raise ValueError("relations must have shape [B, V, V, d_r]")
        return self.mlp(relations)
