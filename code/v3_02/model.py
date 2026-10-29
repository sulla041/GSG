"""V3.02：图条件化共享动力学，将关系内化为节点点集。"""

import torch
from torch import nn

from .diagnostics import step_metrics
from .interactions import PriorInteraction
from .noise import SymmetryNoise
from .readout import IntrinsicReadout


class V3_02Model(nn.Module):
    """V3.02 模型：纯净动力学累加，取消归一化与步长系数。"""

    def __init__(self, node_width=1, relation_width=3, state_width=32, hidden_width=64,
                 steps=32, context="none", node_output=1,
                 relation_output=1, noise_temperature=1., noise_max=.01, noise_stop=1e-6):
        super().__init__()
        if steps < 1 or min(state_width, hidden_width) < 1:
            raise ValueError("steps and widths must be positive")
        self.steps, self.state_width = steps, state_width
        self.node_width, self.relation_width = node_width, relation_width
        self.enc_x = nn.Sequential(nn.Linear(node_width, hidden_width), nn.GELU(),
                                   nn.Linear(hidden_width, state_width))
        self.enc_r = nn.Sequential(nn.Linear(relation_width, hidden_width), nn.GELU(),
                                   nn.Linear(hidden_width, state_width))
        self.offre = PriorInteraction(state_width, hidden_width, context)
        self.noise = SymmetryNoise(noise_temperature, noise_max, noise_stop)
        self.readout = IntrinsicReadout(state_width, hidden_width, node_output,
                                        relation_output, context)

    def vector_field(self, points, encoded_relations, mask):
        """计算同一系统的确定性场，关系已经过 enc_r 编码。"""
        return self.offre(points, encoded_relations, mask)[0]

    def embed(self, nodes, relations, mask, noises=None, return_trace=False,
              return_states=False):
        """节点与关系编码，包含自适应噪声和跨步参数共享。"""
        if nodes.ndim != 3 or nodes.shape[-1] != self.node_width:
            raise ValueError("invalid node feature shape")
        b, n = nodes.shape[:2]
        if relations.shape != (b, n, n, self.relation_width) or mask.shape != (b, n):
            raise ValueError("relation/mask shape does not match nodes")
        if mask.dtype != torch.bool:
            raise ValueError("node mask must be bool")
        if noises is not None and len(noises) != self.steps:
            raise ValueError("noise sequence length must equal steps")
        pairs = mask[:, :, None] & mask[:, None, :]
        nodes = nodes.masked_fill(~mask[..., None], 0)
        relations = relations.masked_fill(~pairs[..., None], 0)
        points = self.enc_x(nodes).masked_fill(~mask[..., None], 0)
        encoded = self.enc_r(relations).masked_fill(~pairs[..., None], 0)
        scale = torch.full_like(points[..., 0], self.noise.noise_max)
        trace, states = [], [points] if return_states else []
        for step in range(self.steps):
            interaction = self.offre(points, encoded, mask, return_trace)
            if return_trace:
                field, pair_contributions, gate, details = interaction
            else:
                field, pair_contributions, gate = interaction
            noise = None if noises is None else noises[step]
            if noise is not None and noise.shape != points.shape:
                raise ValueError("noise shape must match state")
            next_points, scale, injected = self.noise(
                points + field, mask, scale, noise)
            if return_trace:
                row = step_metrics(points, next_points, field, pair_contributions, gate,
                                   injected, scale, mask, 1.0, details)
                row["step"] = torch.tensor(step + 1, device=points.device)
                trace.append(row)
            points = next_points
            if return_states:
                states.append(points)
        if return_trace and return_states:
            return points, trace, states
        if return_trace:
            return points, trace
        if return_states:
            return points, states
        return points

    def decode(self, points, mask, include_nodes=False):
        """仅从终态读取目标图，节点分支根据任务选择。"""
        return self.readout(points, mask, include_nodes)

    def forward(self, nodes, relations, mask, noises=None, return_trace=False,
                include_nodes=False):
        """前向返回终态与关系读出；min_hop 只是单通道任务兼容名。"""
        state = self.embed(nodes, relations, mask, noises, return_trace)
        points, trace = state if return_trace else (state, None)
        result = self.decode(points, mask, include_nodes)
        result["embedding"] = points
        if result["relations"].size(-1) == 1:
            result["min_hop"] = result["relations"]
        return (result, trace) if return_trace else result
