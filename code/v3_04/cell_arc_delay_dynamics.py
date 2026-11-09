"""V3.04 门弧延迟 (Cell Arc Delay) 动力学模型 (Pair 模式读出)：
- 架构哲学 (意涵第一性原理)：
  - 输入与前向演化：100% 保持整图纯净因果 DAG 演化 (获得全局拓扑与阻抗场浸润)；
  - 任务读出：Pair 模式读出头 (Pairwise Readout Head)，从终态点集读取 (from_pin, to_pin) 交互特征并直接回归晶体管固有翻转延时。
"""

import torch
from torch import nn
from code.v3_03.edge_only_dynamics import FreeCausalInteraction, NUM_BASE_TYPES


class CellArcDelayDynamicsModel(nn.Module):
    """全图动力学演化 + Pair 模式读出的微观门弧延迟回归模型。"""

    def __init__(self, state_width=32, hidden_width=64, steps=16):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 1. 意涵 6 大等宽平衡节点编码通道 (24 维)
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

        # 2. 关系边通道 (12 维)
        self.rel_type_emb = nn.Embedding(24, 6)
        self.rel_num_enc = nn.Sequential(nn.Linear(6, 6), nn.GELU(), nn.Linear(6, 6))

        # 3. 动力学演化核 (输入依然是完整图结构)
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        # 4. 意涵 Pair 读出头 (从终态隐状态读取 (src, dst) 相互作用并预测 Delay)
        # 输入：src 特征(32) + dst 特征(32) + Hadamard乘积(32) + 几何差分(32) = 128 维
        pair_in_dim = state_width * 4
        self.pair_readout = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1)
        )
        nn.init.normal_(self.pair_readout[-1].weight, std=0.01)
        nn.init.zeros_(self.pair_readout[-1].bias)

    def forward(self, g, arc_pairs):
        """
        - g: 完整电路图张量；
        - arc_pairs: [2, M] 包含 M 条真实 Timing Arc 的 (from_pin, to_pin) 节点索引。
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

        # 意涵经典 PAIRATTN 读出：提取指定 Timing Arc 端点对并计算交互张量
        u, v = arc_pairs[0], arc_pairs[1]
        p_u, p_v = points[u], points[v]
        pair_feat = torch.cat([p_u, p_v, p_u * p_v, (p_v - p_u)], dim=-1)
        return self.pair_readout(pair_feat).squeeze(-1)
