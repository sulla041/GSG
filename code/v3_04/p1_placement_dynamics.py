"""V3.04 P1 物理布局二维坐标预测模型 (100% 传承 D1 巅峰动力学血统)：
- 继承 D1 6 大等宽平衡通道 (Kind, Base, Drive, Pin, OutDeg, InDeg, 严格 24 维)；
- 继承精炼关系双通道 (类别 6 维 + 阻抗 6 维 = 12 维)；
- 继承自由高维位移与局部 Softmax 排他性竞争；
- 双通道线性直通投影至 (x, y) 物理平面。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class P1PlacementModel(nn.Module):
    """端到端 P1 物理版图二维坐标 (x, y) 动力学预测模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=24):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 6 大等宽平衡通道 (每通道 4 维，总计 24 维，与 D1 100% 对齐)
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

        # 关系双通道 (类别 6 维 + 阻抗 6 维 = 12 维)
        self.rel_type_emb = nn.Embedding(24, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        self.rel_num_enc = nn.Sequential(nn.Linear(2, 6), nn.GELU(), nn.Linear(6, 6))

        # 核心相互作用核 (自由位移 + 局部 Softmax 竞争)
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        # 二维物理坐标直通读出头 (输出归一化后的 [x, y] 相对二维坐标)
        self.xy_readout = nn.Sequential(
            nn.Linear(state_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 2)
        )

    def forward(self, g):
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))

        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

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

        return self.xy_readout(points) # [N, 2]
