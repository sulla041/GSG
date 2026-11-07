"""CC-Tensor v3 专用极速紧凑动力学核心：
- 节点 6 大平衡通道 (24维)：Kind(4) + Base(4) + Drive(4) + Pin(4) + OutDeg(4) + InDeg(4)
- 关系精炼双通道 (12维)：RelType(6) + 阻抗直连(6)
- 纯张量直接查表，彻底消除动态 argmax、预查表 LUT 与度数统计。
"""

import torch
from torch import nn
from .edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class CompactExactEdgeModel(nn.Module):
    """适配 CC-Tensor v3 的极速动力学模型，输入全为紧凑整型与标量。"""

    def __init__(self, state_width=32, hidden_width=64, steps=64):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 1. 意涵 6 大等宽平衡节点通道 (每通道 4 维，总计 24 维)
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

        # 2. 关系双通道 (6维类别 + 6维阻抗 = 12维，支持最大 24 种微观拓扑关系)
        self.rel_type_emb = nn.Embedding(24, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        self.rel_num_enc = nn.Sequential(nn.Linear(2, 6), nn.GELU(), nn.Linear(6, 6))

        # 3. 动力学相互作用核与双轨读出
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)
        self.direct_jump = nn.Linear(state_width, 1)
        nn.init.normal_(self.direct_jump.weight, std=0.01)
        nn.init.zeros_(self.direct_jump.bias)
        self.aux_readout = nn.Sequential(nn.Linear(state_width, hidden_width), nn.GELU(), nn.Linear(hidden_width, 1))
        nn.init.zeros_(self.aux_readout[-1].weight)
        nn.init.zeros_(self.aux_readout[-1].bias)

    def forward(self, g):
        # 极速嵌入查表与线性映射 (零计算浪费)
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))

        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

        # 关系阻抗直连切片
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

        return (self.direct_jump(points) + self.aux_readout(points.detach())).squeeze(-1)
