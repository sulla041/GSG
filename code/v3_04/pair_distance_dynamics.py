"""V3.04 物理点对距离测距仪 (Physical Distance Probe) 动力学模型：
- 严格落实意涵指示：Pair 输入和计算位移完全相同 (sj, si, diff, dist)，仅少一个 r_feat；
- 专用于直接预测节点 Pair 间的真实曼哈顿物理连线距离；
- 规模与 D1 严格绝对对齐 (state_width=32, hidden_width=64, 6大等宽平衡通道 24维)。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class PhysicalDistanceProbeModel(nn.Module):
    """基于两两节点 Pair 交互特征的物理距离测距模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=16):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 6 大等宽平衡通道 (严格对齐 D1: 每通道 4 维，总计 24 维)
        self.kind_emb = nn.Embedding(10, 4)
        self.base_emb = nn.Embedding(NUM_BASE_TYPES, 4)
        self.drive_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.pin_emb = nn.Embedding(19, 4)
        self.out_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.in_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())

        nn.init.normal_(self.kind_emb.weight, std=0.02)
        nn.init.normal_(self.base_emb.weight, std=0.02)
        nn.init.normal_(self.pin_emb.weight, std=0.02)

        self.enc_x = nn.Sequential(nn.Linear(24, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))

        # 拓扑动力学演化核 (保持 D1 纯正位移场做功)
        self.rel_type_emb = nn.Embedding(24, 6)
        self.rel_num_enc = nn.Sequential(nn.Linear(2, 6), nn.GELU(), nn.Linear(6, 6))
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        # 意涵核心武器：Pair 测距读出头
        # 输入与计算位移完全同构：[sj, si, diff, dist] = 3 * state_width + 1，仅少 r_feat！
        pair_in_dim = 3 * state_width + 1
        self.pair_distance_head = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1)
        )

    def forward(self, g, pair_index):
        # 1. 节点 6 大平衡通道编码
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))
        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

        # 2. D1 原生有向边拓扑场演化 T 步
        ei = g["edge_index"]
        src, dst = ei[0], ei[1]
        rel_type_feat = self.rel_type_emb(g["edge_type_id"].long())
        rel_num_in = torch.stack([g["node_log_out_deg"][src], g["node_log_in_deg"][dst]], dim=-1)
        rel_num_feat = self.rel_num_enc(rel_num_in)
        r_feat = torch.cat([rel_type_feat, rel_num_feat], dim=-1)

        source_anchor = g.get("source_anchor_mask", None)
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, ei, r_feat, source_anchor_mask=source_anchor)
            points = points + field

        # 3. 对目标两两节点 Pair 提取同构输入并直接测距
        p_src, p_dst = pair_index[0], pair_index[1]
        s_j, s_i = points[p_src], points[p_dst]
        diff = s_j - s_i
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-6)

        pair_input = torch.cat([s_j, s_i, diff, dist], dim=-1)
        return self.pair_distance_head(pair_input).squeeze(-1)
