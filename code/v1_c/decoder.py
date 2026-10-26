"""V0 公式 3：终态点集解码器。"""

import torch
from torch import nn

from .pairattn import PAIRATTN
from .setattn import SETATTN


class Decoder(nn.Module):
    """使用独立 SETATTN_out 和 PAIRATTN_out 输出目标图。"""

    def __init__(self, width: int, node_width: int, pair_width: int,
                 hidden_width: int, relation_width: int) -> None:
        """创建节点头和关系头。"""
        super().__init__()
        self.setattn = SETATTN(width, pair_width, hidden_width)
        self.node_head = nn.Linear(width, node_width)
        self.pairattn = PAIRATTN(width, pair_width, hidden_width, relation_width)

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 x'(i) 和 r'(i, j)。"""
        nodes = self.node_head(self.setattn(points, node_mask)) * node_mask.unsqueeze(-1)
        return nodes, self.pairattn(points, node_mask)
