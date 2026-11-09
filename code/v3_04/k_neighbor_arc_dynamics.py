"""V3.04 K-邻居增强版物理门弧延迟 (Cell Arc Delay) 动力学模型：
- 意涵战略升级：增加 K 邻居容量以捕获门内多引脚同伴（Sibling Pins）的协同做功与物理偏置。
- 架构：在全图因果演化基础上，注入 K=6 个门内微星团（Cell Micro-Cluster）邻域聚合通道。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class KNeighborCellArcDelayDynamicsModel(nn.Module):
    """K-邻居协同增强版门弧延迟 Pair 读出动力学模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=16, k=6):
        super().__init__()
        self.steps, self.state_width, self.k = steps, state_width, k

        # 1. 节点 6 大平衡通道 (24 维)
        self.kind_emb = nn.Embedding(10, 4)
        self.base_emb = nn.Embedding(NUM_BASE_TYPES, 4)
        self.drive_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.pin_emb = nn.Embedding(19, 4)
        self.out_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.in_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())

        self.enc_x = nn.Sequential(nn.Linear(24, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))

        # 2. 交互核
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)
        self.rel_type_emb = nn.Embedding(14, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)

        # 3. K-邻居聚合核 (物理星团交互)
        self.k_agg = nn.Sequential(nn.Linear(state_width * 2, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))

        # 4. 读出物理先验 (Base-ID + Pin + Sense + RC)
        self.readout_cell_emb = nn.Embedding(NUM_BASE_TYPES, 8)
        self.readout_src_pin_emb = nn.Embedding(19, 4)
        self.readout_dst_pin_emb = nn.Embedding(19, 4)
        self.readout_sense_emb = nn.Embedding(5, 4)
        self.load_rc_enc = nn.Sequential(nn.Linear(4, 8), nn.GELU())

        pair_in_dim = state_width * 4 + 20 + 8
        self.pair_readout = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1)
        )
        nn.init.normal_(self.pair_readout[-1].weight, std=0.01)

    def forward(self, g, arc_pairs, arc_cell_bases, arc_senses, arc_load_rcs, k_neighbor_idx):
        """
        - k_neighbor_idx: [N, K] 预计算的局部邻居索引矩阵。
        """
        # 节点编码
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))
        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

        # K-邻居聚合
        n_feat = points[k_neighbor_idx] # [N, K, D]
        agg = self.k_agg(torch.cat([points.unsqueeze(1).expand(-1, self.k, -1), n_feat], dim=-1))
        points = points + agg.mean(dim=1)

        # 演化
        ei = g["edge_index"]
        rel_feat = torch.cat([self.rel_type_emb(g["edge_type_id"].long()), 
                              torch.zeros(ei.shape[1], 6, device=ei.device)], dim=-1)
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, ei, rel_feat)
            points = points + field

        # Pair 读出
        u, v = arc_pairs[0], arc_pairs[1]
        p_u, p_v = points[u], points[v]
        pair_tensor = torch.cat([p_u, p_v, p_u * p_v, (p_v - p_u)], dim=-1)
        
        physics_prior = torch.cat([
            self.readout_cell_emb(arc_cell_bases.long()),
            self.readout_src_pin_emb(g["node_pin_id"][u].long()),
            self.readout_dst_pin_emb(g["node_pin_id"][v].long()),
            self.readout_sense_emb(arc_senses.long()),
            self.load_rc_enc(arc_load_rcs)
        ], dim=-1)

        return self.pair_readout(torch.cat([pair_tensor, physics_prior], dim=-1)).squeeze(-1)
