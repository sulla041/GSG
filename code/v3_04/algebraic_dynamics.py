"""V3.04 多通道代数运算注入动力学核心：
- 严格保持经典欧拉自由位移与 Softmax 局部竞争的物理累加模式；
- 在偏差计算输入端显式注入 Hadamard 逐元素乘积通道 (sj * si) 与几何投影通道；
- 赋予动力学网络做布尔乘法收缩的直接代数算力，独立演进，不影响 V3.03 时序代码。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import segment_softmax, NUM_BASE_TYPES


class AlgebraicInteraction(nn.Module):
    """V3.04 代数增强相互作用核：输入注入乘积通道与几何投影，输出自由位移与竞争权重。"""

    def __init__(self, state_width=32, hidden_width=64):
        super().__init__()
        self.state_width = state_width
        # 基底构成：sj(d) + si(d) + diff(d) + prod(d) + dist(1) + cos_sim(1) = 4*d + 2
        in_dim = 4 * state_width + 2
        self.joint_head = nn.Sequential(
            nn.Linear(in_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width + 1)
        )
        nn.init.normal_(self.joint_head[-1].weight, std=0.01)
        nn.init.zeros_(self.joint_head[-1].bias)

    def forward(self, points, edge_index, source_anchor_mask=None):
        num_nodes, d = points.shape
        src, dst = edge_index[0], edge_index[1]
        s_j, s_i = points[src], points[dst]
        diff = s_j - s_i
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-6)

        # 意涵关键代数通道注入：逐元素乘积通道与余弦夹角投影
        prod = s_j * s_i
        norm_j = torch.sqrt(torch.sum(s_j ** 2, dim=-1, keepdim=True) + 1e-6)
        norm_i = torch.sqrt(torch.sum(s_i ** 2, dim=-1, keepdim=True) + 1e-6)
        cos_sim = torch.sum(prod, dim=-1, keepdim=True) / (norm_j * norm_i)

        pair_input = torch.cat([s_j, s_i, diff, prod, dist, cos_sim], dim=-1)
        joint_out = self.joint_head(pair_input)

        free_disp = joint_out[:, :self.state_width]
        comp_logit = joint_out[:, self.state_width:self.state_width + 1]

        weights = segment_softmax(comp_logit, dst, num_nodes)
        weighted_disp = free_disp * weights

        field = torch.zeros(num_nodes, d, device=points.device)
        field.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), weighted_disp)
        if source_anchor_mask is not None:
            field = field.masked_fill(source_anchor_mask.unsqueeze(-1), 0.0)
        return field


class V304AlgebraicModel(nn.Module):
    """V3.04 独立多通道代数动力学模型：支持分类读出模式。"""

    def __init__(self, state_width=32, hidden_width=64, steps=24, num_bins=100):
        super().__init__()
        self.steps, self.state_width, self.num_bins = steps, state_width, num_bins

        self.base_emb = nn.Embedding(NUM_BASE_TYPES, 4)
        self.pin_emb = nn.Embedding(19, 4)
        nn.init.normal_(self.base_emb.weight, std=0.02)
        nn.init.normal_(self.pin_emb.weight, std=0.02)

        self.enc_x = nn.Sequential(nn.Linear(8, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))
        self.interaction = AlgebraicInteraction(state_width, hidden_width)

        if self.num_bins > 1:
            self.readout_head = nn.Linear(state_width, num_bins)
            self.register_buffer("bin_centers", torch.linspace(0, 1, num_bins))
        else:
            self.readout_head = nn.Sequential(nn.Linear(state_width, 1), nn.Sigmoid())

    def forward(self, g):
        base_feat = self.base_emb(g["node_base_id"].long())
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        points = self.enc_x(torch.cat([base_feat, pin_feat], dim=-1))

        ei = g["edge_index"]
        source_anchor = g.get("source_anchor_mask", None)
        for _ in range(self.steps):
            field = self.interaction(points, ei, source_anchor_mask=source_anchor)
            points = points + field

        logits = self.readout_head(points)
        if self.num_bins > 1:
            # 分类模式：加权平均 bin centers 得到连续概率
            probs = torch.softmax(logits, dim=-1)
            return (probs * self.bin_centers).sum(dim=-1)
        else:
            return logits.squeeze(-1)
