"""自包含 V2-EX 共享坐标动力学及最终 dense 任务读出。"""

import torch
from torch import nn

from .dynamics import CoordinateDynamics
from .noise import SymmetryNoise
from .pair import AttnPair
from .setattn import SetAttention


class V2ExModel(nn.Module):
    """最终输出可训练，中间迭代仅返回 detached 诊断。"""

    def __init__(self, node_width=4, relation_width=5, point_width=32,
                 pair_width=16, hidden_width=64, steps=8, step_size=0.1,
                 noise_temperature=1.0, noise_max=1e-2, noise_stop=1e-6,
                 graph_centering=False):
        """所有时间步共享一个 CoordinateDynamics 实例。"""
        super().__init__()
        self.steps, self.step_size = steps, step_size
        self.graph_centering = graph_centering
        self.encs = nn.Sequential(nn.Linear(node_width, hidden_width), nn.GELU(),
                                  nn.Linear(hidden_width, point_width))
        self.encr = nn.Sequential(nn.Linear(relation_width, hidden_width), nn.GELU(),
                                  nn.Linear(hidden_width, pair_width))
        self.dynamics = CoordinateDynamics(point_width, pair_width, hidden_width)
        self.noise = SymmetryNoise(noise_temperature, noise_max, noise_stop)
        self.final_context = SetAttention(point_width, point_width, hidden_width)
        self.final_pair = AttnPair(point_width, pair_width, hidden_width)
        self.node_head = nn.Linear(2 * point_width, node_width)
        self.relation_head = nn.Sequential(nn.Linear(pair_width, hidden_width), nn.GELU(),
                                           nn.Linear(hidden_width, 2))
        self.hop_head = nn.Linear(pair_width, 1)

    def forward(self, nodes, relations, node_mask, noises=None, return_trace=False):
        """执行共享动力学；noises 可注入固定噪声以便复现和排列测试。"""
        if noises is not None and len(noises) != self.steps:
            raise ValueError("noises 长度必须等于迭代步数")
        points = self.encs(nodes) * node_mask.unsqueeze(-1)
        relation_state = self.encr(relations)
        previous_scale = torch.full_like(points[..., 0], self.noise.noise_max)
        trace = []
        for step in range(self.steps):
            delta, _ = self.dynamics(points, relation_state, node_mask)
            noise = None if noises is None else noises[step]
            next_points, scale, injected = self.noise(
                points + self.step_size * delta, node_mask, previous_scale, noise
            )
            if self.graph_centering:
                center = (next_points * node_mask.unsqueeze(-1)).sum(1, keepdim=True)
                center = center / node_mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
                next_points = (next_points - center) * node_mask.unsqueeze(-1)
            trace.append({
                "update_norm": (next_points - points).norm().detach(),
                "raw_offset_norm": delta.norm().detach(),
                "noise_norm": injected.norm().detach(),
                "coordinate_norm": next_points.norm().detach(),
                "noise_scale": scale.detach()
            })
            points, previous_scale = next_points, scale
        context = self.final_context(points, node_mask)
        pair = self.final_pair(context, node_mask)
        relation_logits = self.relation_head(pair)
        hop = self.hop_head(pair)
        node_output = self.node_head(torch.cat((points, context), dim=-1))
        output = {"reachable": relation_logits, "min_hop": hop, "nodes": node_output}
        return (output, tuple(trace), points) if return_trace else output
