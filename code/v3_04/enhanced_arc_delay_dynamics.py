"""V3.04 增强版门弧延迟 (Cell Arc Delay) 动力学模型：
- 意涵第一性原理升级：
  - 前向：整图因果拓扑动力学演化保持全局场浸润；
  - 读出：Pair 读出头显式注入 [母元门类型 Base-ID + 输入端口 Pin-ID + 输出端口 Pin-ID] 黄金三元组；
  - 彻底打破黑盒迷雾，直接给模型点亮晶体管固有开关延时的物理基准台阶！
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class EnhancedCellArcDelayDynamicsModel(nn.Module):
    """母元门与端口先验注入的增强版门弧延迟 Pair 读出动力学模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=16):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 1. 节点 6 大等宽平衡通道 (24 维)
        self.kind_emb = nn.Embedding(10, 4)
        self.base_emb = nn.Embedding(NUM_BASE_TYPES, 4)
        self.drive_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.pin_emb = nn.Embedding(19, 4)
        self.out_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.in_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())

        self.enc_x = nn.Sequential(nn.Linear(24, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))

        # 2. 关系边通道 (12 维)
        self.rel_type_emb = nn.Embedding(24, 6)
        self.rel_num_enc = nn.Sequential(nn.Linear(6, 6), nn.GELU(), nn.Linear(6, 6))

        # 3. 动力学演化核
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        # 4. 读出增强专用的母门类型与端口高维先验 Embedding (8维+4维+4维)
        self.readout_cell_emb = nn.Embedding(NUM_BASE_TYPES, 8)
        self.readout_src_pin_emb = nn.Embedding(19, 4)
        self.readout_dst_pin_emb = nn.Embedding(19, 4)

        # Pair 读出输入：状态点对[128维] + 物理先验三元组[16维] = 144 维
        pair_in_dim = state_width * 4 + 16
        self.pair_readout = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1)
        )
        nn.init.normal_(self.pair_readout[-1].weight, std=0.01)
        nn.init.zeros_(self.pair_readout[-1].bias)

    def forward(self, g, arc_pairs, arc_cell_bases):
        """
        - g: 完整电路图张量；
        - arc_pairs: [2, M] 包含 M 条 Timing Arc 的 (from_pin, to_pin) 节点索引；
        - arc_cell_bases: [M] 每条时序弧所属母元门的 base_id。
        """
        # 节点编码
        k_feat = self.kind_emb(g["node_kind_id"].long())
        base_feat = self.base_emb(g["node_base_id"].long())
        drive_feat = self.drive_enc(g["node_drive_val"].unsqueeze(-1))
        pin_feat = self.pin_emb(g["node_pin_id"].long())
        out_feat = self.out_deg_enc(g["node_log_out_deg"].unsqueeze(-1))
        in_feat = self.in_deg_enc(g["node_log_in_deg"].unsqueeze(-1))
        points = self.enc_x(torch.cat([k_feat, base_feat, drive_feat, pin_feat, out_feat, in_feat], dim=-1))

        # 关系特征
        ei = g["edge_index"]
        src, dst = ei[0], ei[1]
        rel_type = self.rel_type_emb(g["edge_type_id"].long())
        base_num = torch.stack([g["node_log_out_deg"][src], g["node_log_in_deg"][dst]], dim=-1)
        rc_feat = g["edge_rc_features"] if "edge_rc_features" in g else torch.zeros(ei.shape[1], 4, device=ei.device)
        rel_feat = torch.cat([rel_type, self.rel_num_enc(torch.cat([base_num, rc_feat], dim=-1))], dim=-1)

        # 全图动力学演化
        source_anchor = g.get("source_anchor_mask", None)
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, ei, rel_feat, source_anchor_mask=source_anchor)
            points = points + field

        # 意涵经典 PAIRATTN 交互张量 (128维)
        u, v = arc_pairs[0], arc_pairs[1]
        p_u, p_v = points[u], points[v]
        pair_tensor = torch.cat([p_u, p_v, p_u * p_v, (p_v - p_u)], dim=-1)

        # 注入母元门类型与引脚端口黄金物理先验 (16维)
        cell_feat = self.readout_cell_emb(arc_cell_bases.long())
        src_pin_feat = self.readout_src_pin_emb(g["node_pin_id"][u].long())
        dst_pin_feat = self.readout_dst_pin_emb(g["node_pin_id"][v].long())
        physics_prior = torch.cat([cell_feat, src_pin_feat, dst_pin_feat], dim=-1)

        # 联合读出
        full_pair_feat = torch.cat([pair_tensor, physics_prior], dim=-1)
        return self.pair_readout(full_pair_feat).squeeze(-1)
