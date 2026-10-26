"""从终态点集读取一个二元关系任务。"""

import torch
from torch import nn

from .pairattn import PAIRATTN


class RelationTaskHead(nn.Module):
    """使用有向 PAIRATTN 输出二元关系 logits。"""

    def __init__(self, width: int, pair_width: int, hidden_width: int) -> None:
        """创建关系分类头。"""
        super().__init__()
        self.head = PAIRATTN(width, pair_width, hidden_width, 2)

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """返回 `[B, V, V, 2]` 关系 logits。"""
        return self.head(points, node_mask)
