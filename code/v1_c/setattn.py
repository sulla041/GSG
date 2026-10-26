"""V0 的集合上下文算子。"""

import torch
from torch import nn


class SETATTN(nn.Module):
    """从点集产生全局节点上下文。"""

    def __init__(self, width: int, attention_width: int, hidden_width: int) -> None:
        """创建单头注意力、残差归一化和集合 MLP。"""
        super().__init__()
        self.query = nn.Linear(width, attention_width, bias=False)
        self.key = nn.Linear(width, attention_width, bias=False)
        self.value = nn.Linear(width, attention_width, bias=False)
        self.output = nn.Linear(attention_width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden_width), nn.GELU(),
                                 nn.Linear(hidden_width, width))

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """只让有效节点参与 key/value，并清零 padding 输出。"""
        query, key, value = self.query(points), self.key(points), self.value(points)
        scores = query @ key.transpose(-1, -2) / (query.size(-1) ** 0.5)
        scores = scores.masked_fill(~node_mask[:, None, :], torch.finfo(scores.dtype).min)
        context = self.output(torch.softmax(scores, dim=-1) @ value)
        output = self.norm(points + context)
        return (output + self.mlp(output)) * node_mask.unsqueeze(-1)
