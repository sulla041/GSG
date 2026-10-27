"""Attn-Pair 点对关系编码，只读取 SETATTN 输出 H。"""

import torch
from torch import nn


class AttnPair(nn.Module):
    """用非共享 source/target 投影编码有序节点对。"""

    def __init__(self, width, pair_width, hidden_width):
        """创建乘积、反向差分和值状态的点对编码器。"""
        super().__init__()
        self.source = nn.Linear(width, pair_width, bias=False)
        self.target = nn.Linear(width, pair_width, bias=False)
        self.value = nn.Linear(width, pair_width, bias=False)
        self.encoder = nn.Sequential(nn.Linear(3 * pair_width + 2 * width, hidden_width),
                                     nn.GELU(), nn.Linear(hidden_width, pair_width))

    def forward(self, context, node_mask):
        """返回 source-first 的 pair[j,i]，无第二次 pair attention。"""
        source = self.source(context)[:, :, None, :]
        target = self.target(context)[:, None, :, :]
        value = self.value(context)[:, :, None, :].expand(
            -1, -1, context.size(1), -1)
        source_context = context[:, :, None, :].expand(-1, -1, context.size(1), -1)
        target_context = context[:, None, :, :].expand(-1, context.size(1), -1, -1)
        features = torch.cat((source * target, source - target, value,
                              source_context, target_context), dim=-1)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        return self.encoder(features) * pair_mask.unsqueeze(-1)
