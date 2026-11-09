"""V3.04 电表节点（Meter Node）整图总面积 (Total Area) 预测动力学模型：
- 意涵战略创新：引入电表节点 (Meter Node) 作为全局物理探针，同构化图级读出。
- 物理机制：全图所有 Standard Cell (kind=0) 均有向连接至电表节点，注入电表专用连接关系。
- 架构特性：保持纯因果动力学骨干，电表节点终态向量直接线性映射整图总面积 (Area)。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class MeterNodeAreaDynamicsModel(nn.Module):
    """电表节点专用整图总面积 (Total Area) 动力学回归模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=16, k=8):
        super().__init__()
        self.steps, self.state_width, self.k = steps, state_width, k

        # 1. 节点 6 大平衡通道 (扩容支持电表节点专有类别)
        self.kind_emb = nn.Embedding(11, 4)
        self.base_emb = nn.Embedding(NUM_BASE_TYPES + 2, 4)
        self.drive_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.pin_emb = nn.Embedding(20, 4)
        self.out_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.in_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())

        self.enc_x = nn.Sequential(
            nn.Linear(24, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width)
        )

        # 2. 交互核与关系嵌入
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)
        self.rel_type_emb = nn.Embedding(16, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        self.rel_num_enc = nn.Sequential(nn.Linear(2, 6), nn.GELU(), nn.Linear(6, 6))
        
        # K 邻居聚合核
        self.k_agg = nn.Sequential(nn.Linear(state_width * 2, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))

        # 3. 电表节点单节点面积读出头
        self.meter_readout = nn.Sequential(
            nn.Linear(state_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1)
        )
        nn.init.normal_(self.meter_readout[-1].weight, std=0.01)
        nn.init.constant_(self.meter_readout[-1].bias, 0.45)

    def forward(self, g, meter_node_idx, k_neighbor_idx):
        # 编码平衡通道
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))
        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))
        
        # K-邻居聚合
        n_feat = points[k_neighbor_idx]
        agg = self.k_agg(torch.cat([points.unsqueeze(1).expand(-1, self.k, -1), n_feat], dim=-1))
        points = points + agg.mean(dim=1)

        ei = g["edge_index"]
        eti = g["edge_type_id"].long()
        src, dst = ei[0], ei[1]

        type_feat = self.rel_type_emb(eti)
        num_feat = self.rel_num_enc(torch.cat([g["node_log_out_deg"][src].unsqueeze(-1),
                                               g["node_log_in_deg"][dst].unsqueeze(-1)], dim=-1))
        rel_feat = torch.cat([type_feat, num_feat], dim=-1)

        # 动力学演化
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, ei, rel_feat)
            points = points + field

        return self.meter_readout(points[meter_node_idx]).squeeze(-1)
