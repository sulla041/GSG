"""不修改坐标主状态的集合级全局上下文。"""

import torch
from torch import nn


class SetAttention(nn.Module):
    """对有效点集做标准自注意力并输出 H。"""

    def __init__(self, width, attention_width, hidden_width):
        """构造无 LayerNorm 的 attention 和残差前馈映射。"""
        super().__init__()
        self.query = nn.Linear(width, attention_width, bias=False)
        self.key = nn.Linear(width, attention_width, bias=False)
        self.value = nn.Linear(width, attention_width, bias=False)
        self.output = nn.Linear(attention_width, width, bias=False)
        self.ffn = nn.Sequential(nn.Linear(width, hidden_width), nn.GELU(),
                                 nn.Linear(hidden_width, width))

    def forward(self, points, node_mask):
        """只返回上下文 H，输入坐标 points 保持不变。"""
        query, key, value = self.query(points), self.key(points), self.value(points)
        scores = query @ key.transpose(-1, -2) / (query.size(-1) ** 0.5)
        key_mask = node_mask[:, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        context = self.output(torch.softmax(scores, dim=-1) @ value)
        result = points + context
        result = result + self.ffn(result)
        return result * node_mask.unsqueeze(-1)
