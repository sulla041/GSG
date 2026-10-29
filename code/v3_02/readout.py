"""纯终态读出：rout 输出关系，off 输出节点残差。"""

from torch import nn

from .interactions import PairOperations, aggregate


class IntrinsicReadout(nn.Module):
    """读出接口仅接收终态 S 和有效节点 mask，不接收原始关系 R。"""

    def __init__(self, width, hidden, node_output, relation_output, context="none"):
        super().__init__()
        # 使用相同的点对运算基，包含显式源与目标状态
        self.operations = PairOperations(width, hidden, context)
        inputs = self.operations.output_width
        self.rout = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(),
                                  nn.Linear(hidden, relation_output))
        self.off = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(),
                                 nn.Linear(hidden, width))
        self.node_head = nn.Linear(width, node_output)

    def forward(self, points, mask, include_nodes=False):
        """hop 任务仅执行 rout；节点任务按需启用独立 off 分支。"""
        features = self.operations(points, mask)
        pairs = mask[:, :, None] & mask[:, None, :]
        relations = self.rout(features).masked_fill(~pairs[..., None], 0)
        result = {"relations": relations}
        if include_nodes:
            node_state = points + aggregate(self.off(features), mask)
            result["node_state"] = node_state.masked_fill(~mask[..., None], 0)
            result["nodes"] = self.node_head(node_state).masked_fill(~mask[..., None], 0)
        return result
