"""V3.02 广义点对运算：显式包含独立源状态与目标状态，赋予模型消息传递选择权。"""

import torch
from torch import nn

from .context import make_context


class PairOperations(nn.Module):
    """生成包含独立源/目标状态以及几何交互的点对特征。"""

    def __init__(self, width, hidden, context="none"):
        super().__init__()
        self.context = make_context(context, width, hidden)
        self.source = nn.Linear(width, width, bias=False)
        self.target = nn.Linear(width, width, bias=False)
        # 包含：source(d), target(d), difference(d), product(d), directed(d), distance(1)
        self.output_width = 5 * width + 1

    def forward(self, points, mask, return_components=False):
        """点对交互：保留独立 s_j、s_i，并结合相对几何运算。"""
        points = points.masked_fill(~mask[..., None], 0)
        context = self.context(points, mask) if self.context is not None else points
        num_nodes = points.size(1)
        # 显式广播源节点与目标节点状态
        source_state = points[:, :, None, :].expand(-1, -1, num_nodes, -1)
        target_state = points[:, None, :, :].expand(-1, num_nodes, -1, -1)
        difference = points[:, :, None, :] - points[:, None, :, :]
        product = context[:, :, None, :] * context[:, None, :, :]
        directed = self.source(context)[:, :, None, :] - self.target(context)[:, None, :, :]
        distance = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
        pairs = mask[:, :, None] & mask[:, None, :]
        components = {
            "source_state": source_state,
            "target_state": target_state,
            "difference": difference,
            "product": product,
            "directed": directed,
            "distance": distance,
        }
        components = {name: value.masked_fill(~pairs[..., None], 0)
                      for name, value in components.items()}
        result = torch.cat(tuple(components.values()), -1)
        return (result, components) if return_components else result


def aggregate(pair_contributions, mask):
    """汇总所有有效点对的相互作用位移：直接求和，不除以总节点数。"""
    pairs = mask[:, :, None] & mask[:, None, :]
    pair_contributions = pair_contributions.masked_fill(~pairs[..., None], 0)
    return pair_contributions.sum(1)


class PriorInteraction(nn.Module):
    """offre：点对运算与先验关系编码共同生成位移。"""

    def __init__(self, width, hidden, context="none"):
        super().__init__()
        self.operations = PairOperations(width, hidden, context)
        features = self.operations.output_width + width
        self.offset = nn.Sequential(nn.Linear(features, hidden), nn.GELU(),
                                    nn.Linear(hidden, width))
        self.route = nn.Sequential(nn.Linear(features, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))

    def _compute_field(self, features, mask):
        """同一条 MLP 路径计算 pairwise field，便于做输入块零化诊断。"""
        gate = self.route(features).sigmoid()
        pair_offset = self.offset(features)
        pair_contributions = pair_offset * gate
        pairs = mask[:, :, None] & mask[:, None, :]
        pair_contributions = pair_contributions.masked_fill(~pairs[..., None], 0)
        return aggregate(pair_contributions, mask), pair_contributions, gate, pair_offset

    def forward(self, points, relations, mask, return_diagnostics=False):
        """单个 offre 跨步共享；可返回详细敏感度统计。"""
        operation_features, components = self.operations(points, mask, True)
        blocks = {**components, "reij": relations}
        features = torch.cat((operation_features, relations), -1)
        field, pair_contributions, gate, pair_offset = self._compute_field(features, mask)
        if not return_diagnostics:
            return field, pair_contributions, gate
        details = {}
        pairs = mask[:, :, None] & mask[:, None, :]
        gate_values = gate.squeeze(-1)[pairs]
        contribution_norm = pair_contributions.norm(dim=-1)[pairs]
        centered_gate = gate_values - gate_values.mean()
        centered_contribution = contribution_norm - contribution_norm.mean()
        gate_corr = (centered_gate * centered_contribution).mean()
        gate_corr = gate_corr / (centered_gate.square().mean().sqrt() *
                                 centered_contribution.square().mean().sqrt()).clamp_min(1e-12)
        ungated_field = aggregate(pair_offset, mask)
        details.update({
            "pair_offset_rms": pair_offset[pairs].square().mean().sqrt(),
            "pair_contribution_rms": pair_contributions[pairs].square().mean().sqrt(),
            "gate_std": gate_values.std(unbiased=False),
            "gate_field_delta_rms": (field - ungated_field).square().mean().sqrt(),
            "gate_contribution_norm_corr": gate_corr,
        })
        for name, block in blocks.items():
            ablated = torch.cat([torch.zeros_like(value) if key == name else value
                                 for key, value in blocks.items()], -1)
            ablated_field = self._compute_field(ablated, mask)[0]
            details[f"{name}_input_rms"] = block.square().mean().sqrt()
            details[f"{name}_field_delta_rms"] = (field - ablated_field).square().mean().sqrt()
        return field, pair_contributions, gate, details
