"""共享的三模态坐标偏移动力学。"""

import torch
from torch import nn

from .pair import AttnPair
from .setattn import SetAttention


class CoordinateDynamics(nn.Module):
    """同一套参数在每个迭代步重复计算关系条件偏移。"""

    def __init__(self, width, pair_width, hidden_width):
        """构造 SETATTN、Attn-Pair 和 f(r,pair,pos)。"""
        super().__init__()
        self.set_attention = SetAttention(width, width, hidden_width)
        self.pair = AttnPair(width, pair_width, hidden_width)
        self.offset = nn.Sequential(
            nn.Linear(2 * pair_width + width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, width)
        )
        self.route = nn.Sequential(nn.Linear(2 * pair_width, hidden_width), nn.GELU(),
                                   nn.Linear(hidden_width, 1))

    def forward(self, points, relation_state, node_mask):
        """返回确定性 DeltaS 和坐标上下文 H，不归一化写回坐标。"""
        context = self.set_attention(points, node_mask)
        pair = self.pair(context, node_mask)
        position = points[:, :, None, :] - points[:, None, :, :]
        offset = self.offset(torch.cat((relation_state, pair, position), dim=-1))
        route = torch.sigmoid(self.route(torch.cat((relation_state, pair), dim=-1)))
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        messages = offset * route * pair_mask.unsqueeze(-1)
        count = pair_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        delta = messages.sum(dim=1) / count
        return delta * node_mask.unsqueeze(-1), context
