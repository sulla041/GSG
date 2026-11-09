"""V3.04 32-边分层平衡电表网络全图总静态漏电功耗 (Total Leakage Power) 预测动力学模型：
- 意涵战略创新：利用 32-边金字塔树状规约网络，同构积分全图所有晶体管的微观亚阈值与栅极漏电。
- 架构特性：入度严格恒定 O(1)，无单点爆炸；Root 电表直接线性读出全图静态漏电功耗 (W)。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class HierarchicalMeterLeakageDynamicsModel(nn.Module):
    """分层平衡电表金字塔全图静态漏电功耗 (Total Leakage Power) 动力学回归模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=16):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 1. 节点 6 大平衡通道 (扩容支持分层电表专属类别)
        self.kind_emb = nn.Embedding(11, 4)
        self.base_emb = nn.Embedding(NUM_BASE_TYPES + 3, 4)
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

        # 3. 顶层主电表 (Root Meter) 单节点漏电读出头
        self.root_meter_readout = nn.Sequential(
            nn.Linear(state_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1)
        )
        nn.init.normal_(self.root_meter_readout[-1].weight, std=0.01)
        # 归一化初始偏置约在 0.30 附近 (均值 4.53uW / 最大值 14.6uW)
        nn.init.constant_(self.root_meter_readout[-1].bias, 0.30)

    def forward(self, g, root_meter_idx):
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))

        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

        ei = g["edge_index"]
        eti = g["edge_type_id"].long()
        src, dst = ei[0], ei[1]

        type_feat = self.rel_type_emb(eti)
        num_feat = self.rel_num_enc(torch.cat([g["node_log_out_deg"][src].unsqueeze(-1),
                                               g["node_log_in_deg"][dst].unsqueeze(-1)], dim=-1))
        rel_feat = torch.cat([type_feat, num_feat], dim=-1)

        # 纯正动力学波前在金字塔中自底向上逐级积分全图静态做功
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, ei, rel_feat)
            points = points + field

        return self.root_meter_readout(points[root_meter_idx]).squeeze(-1)
