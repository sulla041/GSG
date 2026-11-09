"""V3.04 细分类注入与布尔代数对齐动力学模型 (针对 S1 真实漏电)：
- 与信号概率算法 100% 对齐：保持纯正向欧拉因果位移与 Hadamard 乘积通道；
- 突破 01/10 非对称性：为引脚端口配置高分辨率 8 维嵌入，并在相互作用核中显式拼接源/宿引脚角色；
- 独立于 V3.03 体系，专用于攻克微观状态加权真实漏电功耗。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import segment_softmax, NUM_BASE_TYPES


class AlignedPinAwareInteraction(nn.Module):
    """引脚端口角色感知 + 代数乘积相互作用核。"""

    def __init__(self, state_width=32, pin_feat_width=8, hidden_width=64):
        super().__init__()
        self.state_width = state_width
        # 输入基底：sj(d) + si(d) + diff(d) + prod(d) + dist(1) + cos_sim(1) + pin_src(p) + pin_dst(p)
        in_dim = 4 * state_width + 2 + 2 * pin_feat_width
        self.joint_head = nn.Sequential(
            nn.Linear(in_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width + 1)
        )
        nn.init.normal_(self.joint_head[-1].weight, std=0.01)
        nn.init.zeros_(self.joint_head[-1].bias)

    def forward(self, points, edge_index, pin_emb_feat, source_anchor_mask=None):
        num_nodes, d = points.shape
        src, dst = edge_index[0], edge_index[1]
        s_j, s_i = points[src], points[dst]
        diff = s_j - s_i
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-6)

        # 代数乘积与几何投影
        prod = s_j * s_i
        norm_j = torch.sqrt(torch.sum(s_j ** 2, dim=-1, keepdim=True) + 1e-6)
        norm_i = torch.sqrt(torch.sum(s_i ** 2, dim=-1, keepdim=True) + 1e-6)
        cos_sim = torch.sum(prod, dim=-1, keepdim=True) / (norm_j * norm_i)

        # 意涵核心指导：显式注入入射边源引脚与宿引脚细分类角色 (彻底破除 01/10 对称性混淆！)
        p_src = pin_emb_feat[src]
        p_dst = pin_emb_feat[dst]

        pair_input = torch.cat([s_j, s_i, diff, prod, dist, cos_sim, p_src, p_dst], dim=-1)
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


class S1AlignedDynamicsModel(nn.Module):
    """S1 细分类增强动力学模型：高分辨率节点种类分类 + 引脚角色注入。"""

    def __init__(self, state_width=32, hidden_width=64, steps=16):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 节点细分类扩展：门母元 (16维) + 实体类别 (8维) + 引脚端口 (8维)
        self.kind_emb = nn.Embedding(10, 8)
        self.base_emb = nn.Embedding(NUM_BASE_TYPES, 16)
        self.pin_emb = nn.Embedding(19, 8)

        nn.init.normal_(self.kind_emb.weight, std=0.02)
        nn.init.normal_(self.base_emb.weight, std=0.02)
        nn.init.normal_(self.pin_emb.weight, std=0.02)

        # 节点编码器：8 + 16 + 8 = 32 维
        self.enc_x = nn.Sequential(nn.Linear(32, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))
        self.interaction = AlignedPinAwareInteraction(state_width, pin_feat_width=8, hidden_width=hidden_width)

        # 标量漏电读出头 (线性输出对数标准归一化标量)
        self.readout = nn.Sequential(
            nn.Linear(state_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1)
        )

    def forward(self, g):
        k_feat = self.kind_emb(g["node_kind_id"].long())
        b_feat = self.base_emb(g["node_base_id"].long())
        p_feat = self.pin_emb(g["node_pin_id"].long())

        points = self.enc_x(torch.cat([k_feat, b_feat, p_feat], dim=-1))
        ei = g["edge_index"]
        source_anchor = g.get("source_anchor_mask", None)

        for _ in range(self.steps):
            field = self.interaction(points, ei, pin_emb_feat=p_feat, source_anchor_mask=source_anchor)
            points = points + field

        return self.readout(points).squeeze(-1)
