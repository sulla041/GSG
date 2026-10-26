"""V0 公式 2：ENCO 偏移编码。"""

import torch
from torch import nn


class ENCO(nn.Module):
    """实现 u(j, i) = ENCO([p(j, i) || re(j, i)])。"""

    def __init__(self, pair_width: int, point_width: int, hidden_width: int) -> None:
        """创建点对偏移输出网络。"""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * pair_width, hidden_width), nn.GELU(),
                                 nn.Linear(hidden_width, point_width))

    def forward(self, pair_state: torch.Tensor, relation_state: torch.Tensor) -> torch.Tensor:
        """拼接点对状态和关系状态，生成 u(j, i)。"""
        return self.mlp(torch.cat((pair_state, relation_state), dim=-1))
