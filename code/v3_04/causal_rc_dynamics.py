"""V3.04 因果 RC 寄生特征增强动力学模型 (Causal RC Dynamics Model)：
- 严格继承 D1 纯正因果动力学血统 (32维隐状态, 64隐藏层, 6大等宽平衡通道 24维)；
- 支持消融受控注入：通过 use_rc 参数灵活切换“纯逻辑基线”与“4维后布线 RC 寄生增强”；
- 4维 RC 物理通道：net_via_count(通孔阻抗) + net_segment_count(走线级数) + net_wire_length(物理长度) + net_total_cap(对地电容)。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class CausalRCDynamicsModel(nn.Module):
    """支持 4 维边 RC 寄生特征注入的极速紧凑因果动力学模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=32, use_rc=True):
        super().__init__()
        self.steps, self.state_width, self.use_rc = steps, state_width, use_rc

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

        # 2. 关系特征通道：6维类别 + 6维连续物理阻抗
        self.rel_type_emb = nn.Embedding(24, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        # 输入维度：基础度数(2维) + (4维RC特征 if use_rc else 0)
        num_in_dim = 2 + (4 if use_rc else 0)
        self.rel_num_enc = nn.Sequential(nn.Linear(num_in_dim, 6), nn.GELU(), nn.Linear(6, 6))

        # 3. 动力学相互作用核与双轨时序读出
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)
        self.direct_jump = nn.Linear(state_width, 1)
        nn.init.normal_(self.direct_jump.weight, std=0.01)
        nn.init.zeros_(self.direct_jump.bias)
        self.aux_readout = nn.Sequential(nn.Linear(state_width, hidden_width), nn.GELU(), nn.Linear(hidden_width, 1))
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

        # 关系边阻抗特征构建
        ei = g["edge_index"]
        src, dst = ei[0], ei[1]
        rel_type_feat = self.rel_type_emb(g["edge_type_id"].long())

        base_num = torch.stack([g["node_log_out_deg"][src], g["node_log_in_deg"][dst]], dim=-1)
        if self.use_rc and "edge_rc_features" in g:
            num_in = torch.cat([base_num, g["edge_rc_features"]], dim=-1)
        else:
            num_in = base_num

        rel_num_feat = self.rel_num_enc(num_in)
        r_feat = torch.cat([rel_type_feat, rel_num_feat], dim=-1)

        source_anchor = g.get("source_anchor_mask", None)
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, ei, r_feat, source_anchor_mask=source_anchor)
            points = points + field

        return (self.direct_jump(points) + self.aux_readout(points.detach())).squeeze(-1)
