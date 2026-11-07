"""V3.03 向量敏感偏移动力学纯净版：差分向量直通 + O(k^2) 同伴竞争 + 纯物理累加(无门控阻尼)。"""

import torch
from torch import nn
from torch.nn import functional as F


class SiblingCompetitionKernel(nn.Module):
    """O(k^2) 局部同伴竞争仲裁核：流入同一目标节点的所有入射边在两两横向博弈中自发涌现权重。"""

    def __init__(self, state_width=32, comp_width=16):
        super().__init__()
        self.q = nn.Linear(state_width, comp_width, bias=False)
        self.k = nn.Linear(state_width, comp_width, bias=False)
        self.v = nn.Linear(state_width, comp_width, bias=False)
        self.out_proj = nn.Linear(comp_width, state_width)

    def forward(self, points, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n, s_src = points.size(0), points[src]
        q, k, v = self.q(s_src), self.k(s_src), self.v(s_src)
        k_sum = torch.zeros(n, k.size(-1), device=points.device)
        k_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, k.size(-1)), k)
        v_sum = torch.zeros(n, v.size(-1), device=points.device)
        v_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, v.size(-1)), v)
        score = (q * k_sum[dst]).sum(dim=-1, keepdim=True) / (k.size(-1) ** 0.5)
        return self.out_proj(torch.tanh(score) * v_sum[dst])


class SoftplusCausalInteraction(nn.Module):
    """输入向量敏感偏移相互作用核：输入差分向量直通注入驱动偏移，纯 Softplus 保持做功。"""

    def __init__(self, state_width=32, relation_width=16, hidden_width=64, cost_k=1.0):
        super().__init__()
        self.state_width, self.cost_k = state_width, cost_k
        self.sibling_comp = SiblingCompetitionKernel(state_width, comp_width=16)
        self.enc_r = nn.Sequential(
            nn.Linear(relation_width, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width)
        )
        self.vec_trans = nn.Linear(state_width, state_width, bias=False)
        nn.init.orthogonal_(self.vec_trans.weight, gain=0.5)
        self.joint_head = nn.Sequential(
            nn.Linear(5 * state_width + 1, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width + 1)
        )

    def forward(self, points, edge_index, edge_attr, source_anchor_mask=None):
        n, d = points.shape
        src, dst = edge_index[0], edge_index[1]
        s_j, s_i = points[src], points[dst]
        diff = s_j - s_i
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-6)
        r_feat = self.enc_r(edge_attr)
        sibling_ctx = self.sibling_comp(points, edge_index)

        joint_out = self.joint_head(torch.cat([s_j, s_i, diff, dist, r_feat, sibling_ctx], dim=-1))
        sem_offset = joint_out[:, :self.state_width]
        energy_logit = joint_out[:, self.state_width:self.state_width + 1]

        # 意涵第一性原理：显式将唯一差分向量注入偏移计算，打破饱和，恢复几何灵敏度！
        pair_offset = torch.tanh(self.vec_trans(diff) + sem_offset)
        alpha = F.softplus(energy_logit, beta=self.cost_k)
        edge_displacement = pair_offset * alpha

        field = torch.zeros(n, d, device=points.device)
        field.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), edge_displacement)
        if source_anchor_mask is not None:
            field = field.masked_fill(source_anchor_mask.unsqueeze(-1), 0.0)
        return field, edge_displacement, alpha


class ExactEdgeModel(nn.Module):
    """DeepGate 风格紧凑物理通道 + 向量敏感偏移动力学模型 (纯净直接累加版)。"""

    def __init__(self, node_width=65, relation_width=16, state_width=32,
                 hidden_width=64, steps=16, cost_k=1.0, node_output=2):
        super().__init__()
        self.steps, self.state_width = steps, state_width
        self.kind_emb = nn.Embedding(10, 8)
        self.master_emb = nn.Embedding(35, 16)
        self.pin_emb = nn.Embedding(19, 8)
        nn.init.normal_(self.kind_emb.weight, std=0.02)
        nn.init.normal_(self.master_emb.weight, std=0.02)
        nn.init.normal_(self.pin_emb.weight, std=0.02)

        self.enc_x = nn.Sequential(nn.Linear(34, hidden_width), nn.GELU(), nn.Linear(hidden_width, state_width))
        self.interaction = SoftplusCausalInteraction(state_width, relation_width, hidden_width, cost_k=cost_k)

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
        edge_input = torch.cat([edge_attr[:, :14], log_out[src], log_in[dst]], dim=-1)

        # 意涵第一性原理：纯净物理直接累加，彻底废除任何人工节流门控！
        for _ in range(self.steps):
            field, _, _ = self.interaction(points, edge_index, edge_input, source_anchor_mask=source_anchor_mask)
            points = points + field

        return self.direct_jump(points) + self.aux_readout(points.detach())
