"""V3.04 线网 HPWL 关系增强动力学模型 (规模与 D1 严格绝对对齐)：
- 严格对齐 D1 巅峰维度：state_width=32, hidden_width=64, 6大等宽平衡通道 (24维)；
- 扩充关系特征：正向边 + 反向边 + 同网兄弟近邻边，作为显式独立关系类别；
- 关系数值特征注入 Net 扇出规模 k 与端点阻抗 (6维数值阻抗 + 6维关系Embedding = 12维)。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class D1AlignedHPWLDynamicsModel(nn.Module):
    """规模严格与 D1 对齐的线网 HPWL 动力学预测模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=24):
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

        # 关系双通道 (扩展关系类别 6 维 + 物理阻抗 6 维 = 12 维，与 D1 100% 对齐)
        self.rel_type_emb = nn.Embedding(48, 6) # 支持正向与反向关系扩展 (最大索引覆盖至 16+16=32 以上)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        self.rel_num_enc = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 6))

        # 核心相互作用核 (严格对齐 D1)
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        # 意涵核心武器：关系级物理距离读出头 (Edge Readout Head)
        # 输入：源节点隐状态 (state_width) + 宿节点隐状态 (state_width) + 关系特征 (12维)
        self.edge_hpwl_readout = nn.Sequential(
            nn.Linear(state_width * 2 + 12, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1)
        )

    def forward(self, g, edge_index, edge_type_id, rel_num_feat_raw):
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))

        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

        rel_type_feat = self.rel_type_emb(edge_type_id.long())
        rel_num_feat = self.rel_num_enc(rel_num_feat_raw)
        r_feat = torch.cat([rel_type_feat, rel_num_feat], dim=-1)

        source_anchor = g.get("source_anchor_mask", None)
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, edge_index, r_feat, source_anchor_mask=source_anchor)
            points = points + field

        # 关系级读出：拼接源节点、宿节点及其关系特征，直接读出物理线长
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([points[src], points[dst], r_feat], dim=-1)
        return self.edge_hpwl_readout(edge_input).squeeze(-1)

