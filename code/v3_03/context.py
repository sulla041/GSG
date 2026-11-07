"""可选集合上下文，供 H_i 消融诊断。"""

import torch
from torch import nn


class SetContext(nn.Module):
    """保留坐标残差的上下文分支，不归一化写回坐标。"""

    def __init__(self, width, hidden):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                 nn.Linear(hidden, width))

    def forward(self, points, mask):
        """屏蔽 padding key/query；全 padding 图返回零。"""
        scores = self.query(points) @ self.key(points).transpose(-1, -2)
        scores = scores / points.size(-1) ** 0.5
        scores = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min)
        state = points + self.output(scores.softmax(-1) @ self.value(points))
        return (state + self.ffn(state)).masked_fill(~mask[..., None], 0)


def make_context(mode, width, hidden):
    """none 使用草案的原始点对状态，setattn 用于后续 H_i 对照。"""
    if mode not in ("none", "setattn"):
        raise ValueError("context must be none or setattn")
    return SetContext(width, hidden) if mode == "setattn" else None
