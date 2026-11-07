"""V3.03 模型：每层动态 2D 物理空间读出与有界几何近邻演化（纯确定性无噪声）。"""

import torch
from torch import nn

from .interactions import BoundedPriorInteraction, sample_k_dynamic_neighbors
from .noise import SymmetryNoise
from .readout import IntrinsicReadout


class V3_03Model(nn.Module):
    """V3.03 模型：每层即时读取 2D 空间位置，动态更新几何近邻拓扑，彻底移除破坏性高斯噪声。"""

    def __init__(self, node_width=62, relation_width=1, state_width=32, hidden_width=64,
                 steps=8, k=32, context="none", node_output=2,
                 relation_output=1, noise_temperature=1., noise_max=0.0, noise_stop=1e-6):
        super().__init__()
        if steps < 1 or min(state_width, hidden_width) < 1:
            raise ValueError("steps and widths must be positive")
        self.steps, self.state_width = steps, state_width
        self.node_width, self.relation_width = node_width, relation_width
        self.k = k

        self.enc_x = nn.Sequential(
            nn.Linear(node_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width)
        )
        self.enc_r = nn.Sequential(
            nn.Linear(relation_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width)
        )
        self.offre = BoundedPriorInteraction(state_width, hidden_width, k=k, context=context)
        # 默认关闭噪声 (noise_max=0.0)，确保 100% 确定性纯净动力学
        self.noise = SymmetryNoise(noise_temperature, noise_max, noise_stop)
        self.readout = IntrinsicReadout(state_width, hidden_width, node_output,
                                        relation_output, context)

    def embed(self, nodes, relations, mask, die_size=None, noises=None, return_trace=False):
        """每步纯净累加演化，支持 die_size 空间感知。"""
        b, n = nodes.shape[:2]
        nodes = nodes.masked_fill(~mask[..., None], 0)
        points = self.enc_x(nodes).masked_fill(~mask[..., None], 0)

        scale = torch.full_like(points[..., 0], self.noise.noise_max)
        for step in range(self.steps):
            with torch.no_grad():
                current_coords = self.readout(points, mask, die_size=die_size, include_nodes=True)["nodes"]

            idx, rel_feat, n_mask = sample_k_dynamic_neighbors(
                relations, mask, coords=current_coords, k=self.k
            )
            encoded_sparse_rel = self.enc_r(rel_feat).masked_fill(~n_mask[..., None], 0)

            field, _, _ = self.offre(
                points, encoded_sparse_rel, idx, n_mask, mask
            )
            # 纯净动力学直接累加：若无噪声配置，直接等价于 points + field
            noise = None if noises is None else noises[step]
            points, scale, _ = self.noise(points + field, mask, scale, noise)

        return points

    def forward(self, nodes, relations, mask, die_size=None, noises=None, include_nodes=True,
                return_trace=False):
        """仅在最后一层进行解码输出与监督训练。"""
        final_points = self.embed(nodes, relations, mask, die_size=die_size, noises=noises, return_trace=return_trace)
        return self.readout(final_points, mask, die_size=die_size, include_nodes=include_nodes)
