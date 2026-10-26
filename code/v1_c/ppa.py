"""V1-C 的节点 PPA 回归头和关系 PPA 预测头。"""

import torch
from torch import nn

from .pairattn import PAIRATTN
from .setattn import SETATTN


class PPAHead(nn.Module):
    """从终态点集预测节点连续标签和稠密关系标签。"""

    def __init__(self, width: int, pair_width: int, hidden_width: int,
                 node_labels: int, relation_types: int, relation_labels: int) -> None:
        """建立独立的节点回归、关系类别和关系数值头。"""
        super().__init__()
        self.context = SETATTN(width, pair_width, hidden_width)
        self.node_head = nn.Sequential(nn.Linear(width, hidden_width), nn.GELU(),
                                       nn.Linear(hidden_width, node_labels))
        self.relation_head = PAIRATTN(width, pair_width, hidden_width,
                                      relation_types + relation_labels)

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor):
        """返回节点回归值、关系类别 logits 和关系连续值。"""
        context = self.context(points, node_mask)
        node_values = self.node_head(context) * node_mask.unsqueeze(-1)
        relation_output = self.relation_head(points, node_mask)
        relation_class = relation_output[..., :4]
        relation_values = relation_output[..., 4:]
        return node_values, relation_class, relation_values
