"""V3.03 有界点对交互机制：支持每层动态 2D 几何近邻采样与 V3.02 纯净开放动力学。"""

import torch
from torch import nn

from .context import make_context


def sample_k_dynamic_neighbors(relations, mask, coords=None, k=32):
    """纯张量 GPU 并行构建 K-邻域：真实拓扑边绝对优先保底 + 剩余名额独立均匀随机探索。

    优先级打分机制 (彻底打破同构对称与空间粘连死锁)：
    1. 自身自环：score = 10000.0 (最高优先级必选)
    2. 真实拓扑边 (前 10 维)：score = 1000.0 + rand (保底全部入选)
    3. 隐式点对：score = rand (0~1 独立随机抽取，确保同类节点选到完全不同源)
    4. 无效 padding 节点：score = -1e9 (直接剔除)
    """
    b, n, _, rel_dim = relations.shape
    device = relations.device
    cur_k = min(k, n)

    # 目标 i 为 dim=1，源 j 为 dim=2。严格仅将前 10 维视作显式真边
    has_rel = (relations[..., :10].abs().sum(dim=-1) > 1e-6).transpose(1, 2)
    diag = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0).expand(b, -1, -1)
    valid_pair = (mask.unsqueeze(1) & mask.unsqueeze(2)).transpose(1, 2)

    # 基础优先级：每节点独立均匀随机，彻底打破空间位置相同带来的 Tie-breaking 死锁
    priority = torch.rand(b, n, n, device=device)
    priority = torch.where(has_rel, priority + 1000.0, priority)
    priority = torch.where(diag, torch.tensor(10000.0, device=device), priority)
    priority = torch.where(valid_pair, priority, torch.tensor(-1e9, device=device))

    # 取 top-k 最高优先级邻居: [b, n, k]
    _, indices = torch.topk(priority, k=cur_k, dim=-1)

    b_idx = torch.arange(b, device=device)[:, None, None].expand(-1, n, cur_k)
    i_idx = torch.arange(n, device=device)[None, :, None].expand(b, -1, cur_k)
    rel_feat = relations[b_idx, indices, i_idx]

    n_mask = mask.unsqueeze(-1).expand(-1, -1, cur_k) & mask[b_idx, indices]
    return indices, rel_feat, n_mask


class BoundedPairOperations(nn.Module):
    """意涵精炼基底：features = [s_j, s_i, s_j - s_i, dist]。
    彻底移除冗余的点乘与矩阵剪变，保留最核心的施力源状态、受力宿状态、相对差向量以及空间欧氏距离。
    """

    def __init__(self, width, hidden, context="none"):
        super().__init__()
        self.context = make_context(context, width, hidden)
        self.output_width = 3 * width + 1

    def forward(self, points, neighbor_indices, neighbor_mask):
        """计算精炼点对基底：[s_j, s_i, s_j - s_i, dist]。"""
        b, n, d = points.shape
        k = neighbor_indices.shape[-1]
        context = self.context(points, neighbor_mask.any(dim=-1)) if self.context is not None else points
        batch_idx = torch.arange(b, device=points.device)[:, None, None].expand(-1, n, k)

        s_i = points.unsqueeze(2).expand(-1, -1, k, -1)
        s_j = points[batch_idx, neighbor_indices]

        diff = s_j - s_i
        dist = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)

        components = {
            "s_j": s_j, "s_i": s_i, "diff": diff, "dist": dist,
        }
        res = torch.cat([v.masked_fill(~neighbor_mask[..., None], 0) for v in components.values()], -1)
        return res, components


class BoundedPriorInteraction(nn.Module):
    """V3.03 意涵能量成本动力学：基于成本 k 的 Softplus 过程生成万向位移作用系数。"""

    def __init__(self, state_width, hidden_width, k=32, cost_k=1.0, context="none"):
        super().__init__()
        self.k = k
        self.cost_k = cost_k  # 能量成本平滑参数 beta = k
        self.state_width = state_width
        self.operations = BoundedPairOperations(state_width, hidden_width, context)
        # 输入维度: 3 * state_width + 1 (dist) + state_width (re_ji)
        features_dim = self.operations.output_width + state_width

        # 同步联合网络：一步吐出 32 维万向偏移位移 + 1 维能量作用势能 logit
        self.joint_head = nn.Sequential(
            nn.Linear(features_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width + 1)
        )

    def forward(self, points, encoded_sparse_rel, neighbor_indices, neighbor_mask, mask):
        """能量成本位移积累：alpha_ji = Softplus(logit_ji, beta=cost_k)"""
        op_features, _ = self.operations(points, neighbor_indices, neighbor_mask)
        joint_input = torch.cat((op_features, encoded_sparse_rel), dim=-1)

        joint_out = self.joint_head(joint_input)
        pair_offset = joint_out[..., :self.state_width]            # 自由 32 维万向偏移位移
        energy_logit = joint_out[..., self.state_width:self.state_width+1]
        
        # 意涵第一性原理：基于成本 k 的非负、无封顶平滑能量作用系数
        alpha = torch.nn.functional.softplus(energy_logit, beta=self.cost_k)

        pair_contributions = (pair_offset * alpha).masked_fill(~neighbor_mask[..., None], 0)
        field = pair_contributions.sum(dim=2).masked_fill(~mask[..., None], 0)
        return field, pair_contributions, alpha
