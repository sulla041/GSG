"""纯终态读出：rout 输出稀疏或稠密点对关系/相对距离，off 输出节点残差。

第一性原理：
1. 继承 V3.02 原生几何血统，与动力学核 offre 共享同构点对基底：
   [s_j, s_i, s_j - s_i, s_j * s_i, ||s_j - s_i||]；
2. 摒弃任何破坏物理模长的外置归一化，末层偏置初始化为 ~0.03，梯度 100% 贯通回骨干；
3. 支持稀疏点对切片 decode_pairs(points, pairs)，在 O(N) 复杂度下直接预测内禀物理距离。
"""

import torch
from torch import nn


class IntrinsicReadout(nn.Module):
    """V3.03 原生内生读出头：原生支持节点坐标预测与稀疏点对物理距离读出。"""

    def __init__(self, width, hidden, node_output=2, relation_output=1, context="none"):
        super().__init__()
        self.width = width
        # 节点预测分支 (状态 + die_size)
        self.node_mlp = nn.Sequential(
            nn.Linear(width + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, node_output)
        )
        # 点对关系/物理距离原生分支 (4 * width + 1 维同构基底)
        in_pair_dim = 4 * width + 1
        self.rout = nn.Sequential(
            nn.Linear(in_pair_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, relation_output)
        )
        # 偏置初始化在芯片物理欧氏距离真实均值 ~0.03
        nn.init.constant_(self.rout[-1].bias, 0.03)

    def decode_pairs(self, points, pairs):
        """对稀疏采样点对 (M, 2) 提取原生同构几何基底并直接输出物理距离。"""
        if points.dim() == 3: points = points[0]
        s_i = points[pairs[:, 0]]
        s_j = points[pairs[:, 1]]
        diff = s_j - s_i
        dist_s = torch.sqrt(torch.sum(diff**2, dim=-1, keepdim=True) + 1e-6)
        # 原生点对基底：源、目标、差分、哈达玛积、欧氏模长 (与动力学核绝对同构)
        feat = torch.cat([s_j, s_i, diff, s_j * s_i, dist_s], dim=-1)
        pred_dist = self.rout(feat).squeeze(-1)
        return pred_dist

    def forward(self, points, mask, die_size=None, include_nodes=True):
        """节点预测基于粒子终态 S 与芯片物理尺寸。"""
        result = {}
        if include_nodes:
            b, n, d = points.shape
            if die_size is not None:
                ds = die_size.unsqueeze(1).expand(-1, n, -1)
            else:
                ds = torch.zeros(b, n, 2, device=points.device)
            inp = torch.cat([points, ds], dim=-1)
            nodes = self.node_mlp(inp).masked_fill(~mask[..., None], 0)
            result["nodes"] = nodes
            result["node_state"] = points
        return result
