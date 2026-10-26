"""Virtual timing 的全局、query-conditioned relation readout。"""

import torch
from torch import nn

from .setattn import SETATTN


class VirtualTimingHead(nn.Module):
    """只针对输入关系中的 query pair 输出一个连续值。"""

    def __init__(self, point_width: int, relation_width: int,
                 pair_width: int, hidden_width: int) -> None:
        """建立带全局节点上下文和原始关系输入的关系头。"""
        super().__init__()
        self.context = SETATTN(point_width, pair_width, hidden_width)
        self.source = nn.Linear(point_width, pair_width, bias=False)
        self.target = nn.Linear(point_width, pair_width, bias=False)
        input_width = 3 * pair_width + relation_width
        self.readout = nn.Sequential(nn.Linear(input_width, hidden_width), nn.GELU(),
                                     nn.Linear(hidden_width, 1))

    def forward(self, points: torch.Tensor, relations: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        """返回 `[B,N,N,1]`，关系输入直接包含 `r_query`。"""
        context = self.context(points, node_mask)
        source = self.source(context)[:, :, None, :]
        target = self.target(context)[:, None, :, :]
        source_pair = source.expand(-1, -1, points.size(1), -1)
        target_pair = target.expand(-1, points.size(1), -1, -1)
        pair = torch.cat((source_pair, target_pair, source_pair - target_pair, relations), dim=-1)
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        return self.readout(pair) * valid.unsqueeze(-1)
