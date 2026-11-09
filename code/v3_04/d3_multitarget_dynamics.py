"""V3.04 D3 全功能多目标数值回归模型 (Multi-Target Timing Dynamics Model)：
- 继承 D1/D2 巅峰因果 RC 动力学架构 (32维隐状态, 64隐藏层, 4维边RC寄生阻抗特征注入)；
- 支持后布线时序全谱系数值回归：同时输出 [Arrival, Slack, Slew] 等多维关键物理指标；
- 专为丰富功能库与学术论文全方位评估打造。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class MultiTargetCausalRCDynamicsModel(nn.Module):
    """支持多目标后布线物理时序数值回归的因果 RC 动力学模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=32, num_targets=3):
        super().__init__()
        self.steps, self.state_width, self.num_targets = steps, state_width, num_targets

        # 1. 意涵 6 大等宽平衡节点通道 (严格 24 维)
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

        # 2. 关系边双通道 (6维类别 + 6维连续物理阻抗，包含 4 维 RC 特征)
        self.rel_type_emb = nn.Embedding(24, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        self.rel_num_enc = nn.Sequential(nn.Linear(6, 6), nn.GELU(), nn.Linear(6, 6))

        # 3. 动力学相互作用核
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        # 4. 多目标读出头 (num_targets: 0=Arrival, 1=Slack, 2=Slew)
        self.direct_jump = nn.Linear(state_width, num_targets)
        nn.init.normal_(self.direct_jump.weight, std=0.01)
        nn.init.zeros_(self.direct_jump.bias)
        self.aux_readout = nn.Sequential(nn.Linear(state_width, hidden_width), nn.GELU(), nn.Linear(hidden_width, num_targets))
        nn.init.zeros_(self.aux_readout[-1].weight)
        nn.init.zeros_(self.aux_readout[-1].bias)

    def forward(self, g):
        # 节点编码映射 (24 维)
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))
        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

        # 关系边阻抗特征构建 (2维度数 + 4维真实RC特征)
        ei = g["edge_index"]
        src, dst = ei[0], ei[1]
        rel_type_feat = self.rel_type_emb(g["edge_type_id"].long())
        base_num = torch.stack([g["node_log_out_deg"][src], g["node_log_in_deg"][dst]], dim=-1)
        rc_feat = g["edge_rc_features"] if "edge_rc_features" in g else torch.zeros(ei.shape[1], 4, device=ei.device)
        rel_num_feat = self.rel_num_enc(torch.cat([base_num, rc_feat], dim=-1))
        r_feat = torch.cat([rel_type_feat, rel_num_feat], dim=-1)

        source_anchor = g.get("source_anchor_mask", None)
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, ei, r_feat, source_anchor_mask=source_anchor)
            points = points + field

        return self.direct_jump(points) + self.aux_readout(points.detach())
