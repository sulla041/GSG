"""V3.03 最终精炼平衡动力学：
1. 节点特征：6大等宽平衡通道 (每通道严格 4 维，总维度 24 维，解耦门逻辑类别与驱动倍率)；
2. 关系特征：纯真正向精炼双通道 (6维共享类别 + 6维物理阻抗 = 12维)；
3. 动力学核：五重全展开基底 + 无归一化自由高维位移 + 局部 Softmax 极值竞争汇聚。
"""

import math
import torch
from torch import nn


# 28 种活跃标准单元门列表
CELL_MASTERS = [
    'AND2_X1', 'AND3_X1', 'AND4_X1', 'AOI211_X1', 'AOI21_X1', 'AOI221_X1',
    'AOI222_X1', 'AOI22_X1', 'DFF_X1', 'INV_X1', 'MUX2_X1', 'NAND2_X1',
    'NAND3_X1', 'NAND4_X1', 'NOR2_X1', 'NOR3_X1', 'NOR4_X1', 'OAI211_X1',
    'OAI21_X1', 'OAI221_X1', 'OAI222_X1', 'OAI22_X1', 'OAI33_X1', 'OR2_X1',
    'OR3_X1', 'OR4_X1', 'XNOR2_X1', 'XOR2_X1'
]

# 预构建 BaseType (0..27) 与 DriveStrength 映射表
def build_master_lookup_tables():
    base_names = []
    drives = []
    for m in CELL_MASTERS:
        if '_X' in m:
            b, d = m.split('_X')
            d = float(d)
        else:
            b, d = m, 1.0
        base_names.append(b)
        drives.append(d)

    unique_bases = sorted(list(set(base_names)))
    base_to_id = {b: i for i, b in enumerate(unique_bases)}

    # master_id (0..34, 其中 0 为无 master 占位) -> base_id (0..len(unique_bases))
    # 0 代表 null_base
    master_to_base = [0] * 35
    master_to_drive = [0.0] * 35

    for idx, (b, d) in enumerate(zip(base_names, drives)):
        m_id = idx + 1 # 1-based
        if m_id < 35:
            master_to_base[m_id] = base_to_id[b] + 1 # 1-based, 0 为 none
            master_to_drive[m_id] = math.log2(max(d, 1.0)) / 5.0 # 对数归一化

    return len(unique_bases) + 1, master_to_base, master_to_drive

NUM_BASE_TYPES, MASTER_TO_BASE, MASTER_TO_DRIVE = build_master_lookup_tables()


def segment_softmax(logits, dst, num_nodes):
    """分段防溢出 Softmax：在流入同一宿节点的所有关系边之间展开排他性竞争 (权重和为 1)。"""
    max_logit = torch.full((num_nodes, 1), -1e9, device=logits.device)
    max_logit.scatter_reduce_(0, dst.unsqueeze(-1), logits, reduce="amax", include_self=False)
    shifted_exp = torch.exp((logits - max_logit[dst]).clamp(max=80.0))
    sum_exp = torch.zeros(num_nodes, 1, device=logits.device)
    sum_exp.scatter_add_(0, dst.unsqueeze(-1), shifted_exp)
    return shifted_exp / (sum_exp[dst] + 1e-6)


class FreeCausalInteraction(nn.Module):
    """意涵统一自由位移相互作用核：输入 (sj, si, sj-si, dist, r_feat)，直接预测自由位移与竞争权重。"""

    def __init__(self, state_width=32, relation_feat_width=12, hidden_width=64):
        super().__init__()
        self.state_width = state_width
        in_dim = 3 * state_width + 1 + relation_feat_width
        self.joint_head = nn.Sequential(
            nn.Linear(in_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width + 1)
        )
        nn.init.normal_(self.joint_head[-1].weight, std=0.01)
        nn.init.zeros_(self.joint_head[-1].bias)

    def forward(self, points, edge_index, r_feat, source_anchor_mask=None):
        num_nodes, d = points.shape
        src, dst = edge_index[0], edge_index[1]
        s_j, s_i = points[src], points[dst]
        diff = s_j - s_i
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-6)

        pair_input = torch.cat([s_j, s_i, diff, dist, r_feat], dim=-1)
        joint_out = self.joint_head(pair_input)

        free_displacement = joint_out[:, :self.state_width]
        comp_logit = joint_out[:, self.state_width:self.state_width + 1]

        weights = segment_softmax(comp_logit, dst, num_nodes)
        weighted_displacement = free_displacement * weights

        field = torch.zeros(num_nodes, d, device=points.device)
        field.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), weighted_displacement)
        if source_anchor_mask is not None:
            field = field.masked_fill(source_anchor_mask.unsqueeze(-1), 0.0)
        return field, weighted_displacement, weights


class ExactEdgeModel(nn.Module):
    """意涵 6 通道等宽平衡动力学模型：
    - 节点: 6 大等宽通道 (Kind=4, BaseType=4, DriveStrength=4, Pin=4, OutDeg=4, InDeg=4，总维度 24 维)
    - 关系: 精炼双通道 (Type=6, Impedance=6，总维度 12 维)
    """

    def __init__(self, node_width=65, relation_width=16, state_width=32,
                 hidden_width=64, steps=16, cost_k=1.0, node_output=2, **kwargs):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 意涵第一性原理：6 大等宽平衡节点通道 (每通道严格 4 维，平权物理阻抗与驱动强度)
        self.kind_emb = nn.Embedding(10, 4)
        self.base_emb = nn.Embedding(NUM_BASE_TYPES, 4)
        self.drive_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.pin_emb = nn.Embedding(19, 4)
        self.out_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.in_deg_enc = nn.Sequential(nn.Linear(1, 4), nn.GELU())

        nn.init.normal_(self.kind_emb.weight, std=0.02)
        nn.init.normal_(self.base_emb.weight, std=0.02)
        nn.init.normal_(self.pin_emb.weight, std=0.02)

        # 预注册查找张量（提升 GPU 索引效率）
        self.register_buffer("m_to_base_lut", torch.tensor(MASTER_TO_BASE, dtype=torch.long))
        self.register_buffer("m_to_drive_lut", torch.tensor(MASTER_TO_DRIVE, dtype=torch.float32).unsqueeze(-1))

        # 节点总维度: 4*6 = 24 维！极其紧凑凝聚！
        self.enc_x = nn.Sequential(nn.Linear(24, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))

        # 关系特征通道: 6维类别 + 6维阻抗 = 12 维
        self.rel_type_emb = nn.Embedding(14, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        self.rel_num_enc = nn.Sequential(nn.Linear(2, 6), nn.GELU(), nn.Linear(6, 6))

        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        self.direct_jump = nn.Linear(state_width, node_output)
        nn.init.normal_(self.direct_jump.weight, std=0.01)
        nn.init.zeros_(self.direct_jump.bias)
        self.aux_readout = nn.Sequential(nn.Linear(state_width, hidden_width), nn.GELU(), nn.Linear(hidden_width, node_output))
        nn.init.zeros_(self.aux_readout[-1].weight)
        nn.init.zeros_(self.aux_readout[-1].bias)

    def extract_balanced_node_features(self, nodes, log_out, log_in):
        """意涵第一性原理：从 73 维原始输入中纯张量解耦出 6 大等宽平衡通道 (严格 24 维)。"""
        raw_cats = nodes[:, :63]
        
        # 1. 实体类别 ID (0..9)
        k_id = raw_cats[:, :10].argmax(dim=-1)
        k_feat = self.kind_emb(k_id) # (N, 4)

        # 2 & 3. 门类型解耦：基础逻辑类型 (BaseType) 与 驱动强度倍率 (DriveStrength)
        m_slice = raw_cats[:, 10:44]
        has_m = m_slice.sum(dim=-1) > 0
        raw_m_id = torch.where(has_m, m_slice.argmax(dim=-1) + 1, torch.zeros_like(k_id))
        
        # 通过预设 LUT 张量查表解耦
        base_id = self.m_to_base_lut[raw_m_id]
        drive_val = self.m_to_drive_lut[raw_m_id]

        base_feat = self.base_emb(base_id) # (N, 4)
        drive_feat = self.drive_enc(drive_val) # (N, 4)

        # 4. 引脚端口 ID (0..18)
        p_slice = raw_cats[:, 44:62]
        has_p = p_slice.sum(dim=-1) > 0
        p_id = torch.where(has_p, p_slice.argmax(dim=-1) + 1, torch.zeros_like(k_id))
        p_feat = self.pin_emb(p_id) # (N, 4)

        # 5 & 6. 连续物理阻抗：负载出度电容与扇入汇聚争夺
        out_feat = self.out_deg_enc(log_out) # (N, 4)
        in_feat = self.in_deg_enc(log_in) # (N, 4)

        # 6 大等宽通道全平衡拼接: 4 * 6 = 24 维！
        return torch.cat([k_feat, base_feat, drive_feat, p_feat, out_feat, in_feat], dim=-1)

    def build_forward_rel_channels(self, edge_index, edge_attr, log_out, log_in):
        """纯前向因果关系双通道构建：6维类别 + 6维阻抗 = 12维。"""
        src, dst = edge_index[0], edge_index[1]
        type_id = edge_attr[:, :14].argmax(dim=-1)
        num_feat = torch.cat([log_out[src], log_in[dst]], dim=-1)

        type_feat = self.rel_type_emb(type_id)
        num_feat = self.rel_num_enc(num_feat)
        r_feat = torch.cat([type_feat, num_feat], dim=-1)
        return edge_index, r_feat

    def forward(self, nodes, edge_index, edge_attr, source_anchor_mask=None, **kwargs):
        num_nodes = nodes.size(0)
        src, dst = edge_index[0], edge_index[1]
        out_deg = torch.zeros(num_nodes, 1, device=nodes.device)
        in_deg = torch.zeros(num_nodes, 1, device=nodes.device)
        out_deg.scatter_add_(0, src.unsqueeze(-1), torch.ones_like(src.unsqueeze(-1), dtype=torch.float32))
        in_deg.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst.unsqueeze(-1), dtype=torch.float32))
        log_out, log_in = torch.log1p(out_deg), torch.log1p(in_deg)

        # 1. 意涵 6 大等宽通道平衡映射 (严格 24 维)
        balanced_node_in = self.extract_balanced_node_features(nodes, log_out, log_in)
        points = self.enc_x(balanced_node_in)

        # 2. 关系特征通道构建 (12 维)
        cur_ei, r_feat = self.build_forward_rel_channels(edge_index, edge_attr, log_out, log_in)

        # 3. 32 步纯张量飞速循环，零开销
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, cur_ei, r_feat, source_anchor_mask=source_anchor_mask)
            points = points + field

        return self.direct_jump(points) + self.aux_readout(points.detach())
