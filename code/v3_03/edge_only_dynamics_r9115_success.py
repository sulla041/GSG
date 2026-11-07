"""V3.03 纯净前向因果动力学：五重全展开基底 + [关系类别通道 Embedding + 连续物理阻抗通道] 联合编码。"""

import torch
from torch import nn


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
    """DeepGate 风格紧凑物理通道 + 精炼关系通道 [共享类别 (6维) + 连续物理阻抗 (6维)]。"""

    def __init__(self, node_width=65, relation_width=16, state_width=32,
                 hidden_width=64, steps=16, cost_k=1.0, node_output=2, **kwargs):
        super().__init__()
        self.steps, self.state_width = steps, state_width

        # 节点通道 Embedding (34 维)
        self.kind_emb = nn.Embedding(10, 8)
        self.master_emb = nn.Embedding(35, 16)
        self.pin_emb = nn.Embedding(19, 8)
        nn.init.normal_(self.kind_emb.weight, std=0.02)
        nn.init.normal_(self.master_emb.weight, std=0.02)
        nn.init.normal_(self.pin_emb.weight, std=0.02)
        self.enc_x = nn.Sequential(nn.Linear(34, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))

        # 意涵第一性原理：极简精炼关系通道！
        # 通道 1: 拓扑类别 (14 种) -> 仅 6 维紧凑 Embedding
        self.rel_type_emb = nn.Embedding(14, 6)
        nn.init.normal_(self.rel_type_emb.weight, std=0.02)
        # 通道 2: 连续物理数值 (源出度、宿入度) -> 6 维线性映射
        self.rel_num_enc = nn.Sequential(nn.Linear(2, 6), nn.GELU(), nn.Linear(6, 6))

        # 关系总维度: 6 (类别) + 6 (物理阻抗) = 12 维纯净信噪比通道！
        self.interaction = FreeCausalInteraction(state_width, relation_feat_width=12, hidden_width=hidden_width)

        self.direct_jump = nn.Linear(state_width, node_output)
        nn.init.normal_(self.direct_jump.weight, std=0.01)
        nn.init.zeros_(self.direct_jump.bias)
        self.aux_readout = nn.Sequential(nn.Linear(state_width, hidden_width), nn.GELU(), nn.Linear(hidden_width, node_output))
        nn.init.zeros_(self.aux_readout[-1].weight)
        nn.init.zeros_(self.aux_readout[-1].bias)

    def extract_channels(self, nodes):
        raw_cats = nodes[:, :63]
        k_id = raw_cats[:, :10].argmax(dim=-1)
        m_slice = raw_cats[:, 10:44]
        m_id = torch.where(m_slice.sum(dim=-1) > 0, m_slice.argmax(dim=-1) + 1, torch.zeros_like(k_id))
        p_slice = raw_cats[:, 44:62]
        p_id = torch.where(p_slice.sum(dim=-1) > 0, p_slice.argmax(dim=-1) + 1, torch.zeros_like(k_id))
        return k_id, m_id, p_id

    def build_forward_rel_channels(self, edge_index, edge_attr, log_out, log_in):
        """意涵第一性原理：纯前向因果关系通道化表征，回归 STA 本质。"""
        src, dst = edge_index[0], edge_index[1]
        
        # 1. 拓扑类型 ID: [0..13]
        type_id = edge_attr[:, :14].argmax(dim=-1) # (E,)
        # 2. 连续物理阻抗
        num_feat = torch.cat([log_out[src], log_in[dst]], dim=-1)

        # 3. 双通道融合: 6维类别 + 6维阻抗 = 12 维精炼特征！
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

        k_id, m_id, p_id = self.extract_channels(nodes)
        dense_in = torch.cat([self.kind_emb(k_id), self.master_emb(m_id), self.pin_emb(p_id), log_in, log_out], dim=-1)
        points = self.enc_x(dense_in)

        # 循环外一次性构建纯净前向因果关系通道特征 (12 维)
        cur_ei, r_feat = self.build_forward_rel_channels(edge_index, edge_attr, log_out, log_in)

        # 32 步张量飞速循环，零开销
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, cur_ei, r_feat, source_anchor_mask=source_anchor_mask)
            points = points + field

        return self.direct_jump(points) + self.aux_readout(points.detach())
