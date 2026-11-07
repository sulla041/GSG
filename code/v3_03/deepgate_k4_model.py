"""DeepGate2 K=4 三元关系特征模型：关系特征包含 [正向, 反向, 无连接] 三个维度。"""

import torch
from torch import nn


def sample_k4_directional_neighbors(edge_index, num_nodes, k=4):
    """为每个节点构建 K=4 邻居采样，显式抽取并区分正向、反向与无连接。

    优先级规则：
    1. 正向入射驱动源 (j -> i)：最高优先级保底入选
    2. 反向输出负载端 (i -> j)：次高优先级保底入选
    3. 无连接全局随机探索：填补剩余配额
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]

    # 构建因果邻接优先级打分
    # priority[i, j]: 宿节点 i 挑选 源节点 j 的分数
    priority = torch.rand(num_nodes, num_nodes, device=device) # 基础随机 [0, 1)

    # 正向驱动边 j -> i: priority 大幅度提高
    priority[dst, src] += 200.0
    # 反向负载边 i -> j: priority 适度提高
    priority[src, dst] += 100.0

    # 取 top-k 邻居索引: [num_nodes, k]
    cur_k = min(k, num_nodes)
    _, indices = torch.topk(priority, k=cur_k, dim=-1)

    # 构建三元关系特征 [num_nodes, k, 3]:
    # 通道 0: 正向 (j -> i)
    # 通道 1: 反向 (i -> j)
    # 通道 2: 无连接 (unconnected)
    rel_feat = torch.zeros(num_nodes, cur_k, 3, device=device)
    i_nodes = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, cur_k)
    j_nodes = indices

    # 查表匹配正向与反向
    fwd_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
    bwd_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
    fwd_mask[dst, src] = True
    bwd_mask[src, dst] = True

    is_fwd = fwd_mask[i_nodes, j_nodes]
    is_bwd = bwd_mask[i_nodes, j_nodes]
    is_unconn = (~is_fwd) & (~is_bwd)

    rel_feat[..., 0] = is_fwd.float()
    rel_feat[..., 1] = is_bwd.float()
    rel_feat[..., 2] = is_unconn.float()

    return indices, rel_feat


class DirectionalK4Interaction(nn.Module):
    """K=4 三元关系相互作用层：点对基底与关系特征显式编码。"""

    def __init__(self, state_width=32, hidden_width=64):
        super().__init__()
        self.state_width = state_width
        self.norm = nn.LayerNorm(state_width)

        # 3 维关系特征编码: [正向, 反向, 无连接] -> state_width
        self.enc_r = nn.Sequential(
            nn.Linear(3, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width)
        )
        # 点对基底: s_j (d) + s_i (d) + (s_j - s_i) (d) + dist (1) + enc_r (d) = 4*d + 1
        in_dim = 4 * state_width + 1
        self.joint_head = nn.Sequential(
            nn.Linear(in_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width + 1)
        )

    def forward(self, points, neighbor_indices, rel_feat):
        """points: [N, d], neighbor_indices: [N, k], rel_feat: [N, k, 3]"""
        n, d = points.shape
        k = neighbor_indices.size(1)

        p_normed = self.norm(points)
        s_i = p_normed.unsqueeze(1).expand(-1, k, -1)     # [N, k, d]
        s_j = p_normed[neighbor_indices]                  # [N, k, d]
        diff = s_j - s_i
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-6)
        r_emb = self.enc_r(rel_feat)                      # [N, k, d]

        pair_input = torch.cat([s_j, s_i, diff, dist, r_emb], dim=-1)
        joint_out = self.joint_head(pair_input)

        pair_offset = joint_out[..., :self.state_width]
        alpha = torch.tanh(joint_out[..., self.state_width:self.state_width + 1])
        pair_contributions = pair_offset * alpha          # [N, k, d]

        # 纯净直接累加 (Sum) 聚合所有 K=4 邻域点对的做功
        field = pair_contributions.sum(dim=1)             # [N, d]
        return field


class DeepGateK4Model(nn.Module):
    """DeepGate2 K=4 三元关系动力学模型。"""

    def __init__(self, node_width=3, state_width=32, hidden_width=64, steps=8, k=4):
        super().__init__()
        self.steps = steps
        self.k = k

        self.enc_x = nn.Sequential(
            nn.Linear(node_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, state_width)
        )
        self.interaction = DirectionalK4Interaction(state_width, hidden_width)
        self.readout = nn.Sequential(
            nn.Linear(state_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid()
        )

    def forward(self, nodes, edge_index):
        points = self.enc_x(nodes)
        num_nodes = nodes.size(0)
        # 每步或固定采样 K=4 邻居
        indices, rel_feat = sample_k4_directional_neighbors(edge_index, num_nodes, k=self.k)
        for _ in range(self.steps):
            field = self.interaction(points, indices, rel_feat)
            points = points + field
        return self.readout(points).squeeze(-1)
