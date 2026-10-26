"""V0 的有向 PAIRATTN 算子。"""

import torch
from torch import nn

from .setattn import SETATTN


class PAIRATTN(nn.Module):
    """由源点、目标点及集合上下文产生 p(j, i)。"""

    def __init__(self, width: int, pair_width: int, hidden_width: int,
                 output_width: int | None = None) -> None:
        """源/目标投影不共享；输出宽度可用于关系解码。"""
        super().__init__()
        output_width = pair_width if output_width is None else output_width
        self.context = SETATTN(width, pair_width, hidden_width)
        self.source = nn.Linear(width, pair_width, bias=False)
        self.target = nn.Linear(width, pair_width, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * pair_width, hidden_width), nn.GELU(),
                                 nn.Linear(hidden_width, output_width))

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """返回有序点对张量 `[B, V, V, d_p]` 或 `[B, V, V, d_r]`。"""
        context = self.context(points, node_mask)
        source = self.source(context)[:, :, None, :]
        target = self.target(context)[:, None, :, :]
        pair = torch.cat((source * target, source - target), dim=-1)
        output = self.mlp(pair)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        return output * pair_mask.unsqueeze(-1)
