"""V0 公式 2：逐层独立的集合变换。"""

import torch
from torch import nn

from .enco import ENCO
from .gate import Gate
from .pairattn import PAIRATTN


class EvolutionLayer(nn.Module):
    """执行 p、u、a、o 和 LayerNorm 残差更新。"""

    def __init__(self, width: int, pair_width: int, hidden_width: int) -> None:
        """每层独立持有三个算子和一个 LayerNorm。"""
        super().__init__()
        self.pairattn = PAIRATTN(width, pair_width, hidden_width)
        self.enco = ENCO(pair_width, width, hidden_width)
        self.gate = Gate(pair_width, hidden_width)
        self.norm = nn.LayerNorm(width)

    def forward(self, points: torch.Tensor, relation_state: torch.Tensor,
                node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """聚合源节点 j 后更新目标节点 i。"""
        pair_state = self.pairattn(points, node_mask)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        offset = self.enco(pair_state, relation_state)
        gate = self.gate(pair_state, relation_state) * pair_mask.unsqueeze(-1)
        total = (gate * offset).sum(dim=1)
        return self.norm(points + total) * node_mask.unsqueeze(-1), total
