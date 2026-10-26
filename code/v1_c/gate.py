"""V0 公式 2：标量 Gate。"""

import torch
from torch import nn


class Gate(nn.Module):
    """实现 a(j, i) = Sigmoid(Linear([p(j, i) || re(j, i)]))。"""

    def __init__(self, pair_width: int, hidden_width: int) -> None:
        """创建点对标量门控网络。"""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * pair_width, hidden_width), nn.GELU(),
                                 nn.Linear(hidden_width, 1))

    def forward(self, pair_state: torch.Tensor, relation_state: torch.Tensor) -> torch.Tensor:
        """返回范围为 `(0, 1)` 的软门控权重。"""
        return torch.sigmoid(self.mlp(torch.cat((pair_state, relation_state), dim=-1)))
