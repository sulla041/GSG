"""DeepGate2 纯真有向边动力学退化模型：纯单向拓扑边做功，边特征标量为 1.0。"""

import torch
from torch import nn


class DeepGateExactEdgeInteraction(nn.Module):
    """纯真有向边相互作用层：仅沿显式 DAG 有向拓扑边传递力场。"""

    def __init__(self, state_width=32, hidden_width=64):
        super().__init__()
        self.state_width = state_width
        self.norm = nn.LayerNorm(state_width)

        # 点对基底: s_j (d) + s_i (d) + (s_j - s_i) (d) + dist (1) + edge_feat (1) = 3*d + 2
        in_dim = 3 * state_width + 2
        self.joint_head = nn.Sequential(
            nn.Linear(in_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width + 1)
        )

    def forward(self, points, edge_index):
        """沿真实边集合 edge_index [2, E] 前向传递力场位移。"""
        n, d = points.shape
        src, dst = edge_index[0], edge_index[1]
        e = edge_index.size(1)

        p_normed = self.norm(points)
        s_j = p_normed[src] # 驱动源门
        s_i = p_normed[dst] # 受力宿门
        diff = s_j - s_i
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-6)
        edge_attr = torch.ones(e, 1, device=points.device) # 纯真边存在性标记 1.0

        pair_input = torch.cat([s_j, s_i, diff, dist, edge_attr], dim=-1)
        joint_out = self.joint_head(pair_input)

        pair_offset = joint_out[:, :self.state_width]
        alpha = torch.tanh(joint_out[:, self.state_width:self.state_width + 1])
        edge_displacement = pair_offset * alpha

        field = torch.zeros(n, d, device=points.device)
        field.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), edge_displacement)
        return field


class DeepGateExactModel(nn.Module):
    """DeepGate2 端到端纯真有向边演化模型：无作弊纯拓扑信号概率预测。"""

    def __init__(self, node_width=3, state_width=32, hidden_width=64, steps=8):
        super().__init__()
        self.steps = steps
        self.enc_x = nn.Sequential(
            nn.Linear(node_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width)
        )
        self.interaction = DeepGateExactEdgeInteraction(state_width, hidden_width)
        self.readout = nn.Sequential(
            nn.Linear(state_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid() # 严格映射到概率区间 [0.0, 1.0]
        )

    def forward(self, nodes, edge_index):
        points = self.enc_x(nodes)
        for _ in range(self.steps):
            field = self.interaction(points, edge_index)
            points = points + field
        return self.readout(points).squeeze(-1)
