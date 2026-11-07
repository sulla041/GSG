"""V3.03 电路 P0 节点回归任务：平滑绝对与相对损失联合监督。"""

import torch
from torch.nn import functional as F


def loss(prediction, target, target_mask, die_size=None, scale_weight=1.0, repulsion_weight=0.2):
    """纯 X 坐标预测与纯线性无衰减排斥监督 (Linear Repulsion Loss)。

    第一性原理：
    1. 纯 X 轴回归 (Smooth L1)：集中全力攻克横向连续流向预测；
    2. 芯片宽度归一化误差：基于 W_die 进行物理尺度对齐；
    3. 纯线性无衰减点对排斥监督 (L_rep)：
       惩罚所有 X 方向间距小于真实间距的点对，使用纯线性 ReLU(dt - dp)，
       彻底根除 Smooth L1 平方衰减（梯度提升 50 倍！），强力击穿均值塌缩！
    """
    if not target_mask.any():
        raise ValueError("batch has no valid placement targets")
    # 提取纯 X 标量坐标 (形状归一化为 [N, 1])
    p = prediction[target_mask]
    t = target[target_mask]
    if p.dim() == 1: p = p.unsqueeze(-1)
    if t.dim() == 1: t = t.unsqueeze(-1)
    if t.size(-1) > 1: t = t[:, 0:1] # 只取 X 坐标
    if p.size(-1) > 1: p = p[:, 0:1]

    # 1. 纯 X 绝对误差
    l1_loss = F.smooth_l1_loss(p, t)

    # 2. 芯片宽度归一化损失
    rel_die_loss = torch.tensor(0.0, device=prediction.device)
    if die_size is not None:
        if die_size.dim() == 2 and die_size.size(0) == prediction.size(0):
            b_idx = torch.arange(prediction.size(0), device=prediction.device).unsqueeze(1).expand(-1, prediction.size(1))
            valid_b_idx = b_idx[target_mask]
            w_valid = die_size[valid_b_idx, 0:1].clamp_min(1e-4)
        else:
            w_valid = die_size[:, 0:1].clamp_min(1e-4)
        rel_die_loss = F.smooth_l1_loss(p / w_valid, t / w_valid)

    # 3. 纯线性点对排斥监督 (直接惩罚 X 方向拥挤度，梯度无衰减)
    rep_loss = torch.tensor(0.0, device=prediction.device)
    if repulsion_weight > 0.0:
        b_losses = []
        for b in range(prediction.size(0)):
            m = target_mask[b]
            m_count = m.sum().item()
            if m_count >= 4:
                pb = prediction[b, m, 0:1] if prediction.size(-1) > 1 else prediction[b, m]
                tb = target[b, m, 0:1] if target.size(-1) > 1 else target[b, m]
                if pb.dim() == 1: pb = pb.unsqueeze(-1)
                if tb.dim() == 1: tb = tb.unsqueeze(-1)
                # X 轴一维欧氏间距
                dp = (pb - pb.T).abs()
                dt = (tb - tb.T).abs()
                triu = torch.triu_indices(m_count, m_count, offset=1, device=pb.device)
                dp_pairs = dp[triu[0], triu[1]]
                dt_pairs = dt[triu[0], triu[1]]
                # 纯线性排斥惩罚: 只要 dp < dt，直接施加常数 1.0 反向梯度推开！
                rep_penalty = F.relu(dt_pairs - dp_pairs)
                b_losses.append(rep_penalty.mean())
        if b_losses:
            rep_loss = torch.stack(b_losses).mean()

    return l1_loss + scale_weight * rel_die_loss + repulsion_weight * rep_loss


@torch.no_grad()
def metrics(prediction, target, target_mask, die_size=None):
    """返回纯 X 物理绝对误差 MAE、宽度相对比率 (%) 与实时皮尔逊相关性 r(x)。"""
    p = prediction[target_mask]
    t = target[target_mask]
    if p.dim() == 1: p = p.unsqueeze(-1)
    if t.dim() == 1: t = t.unsqueeze(-1)
    if t.size(-1) > 1: t = t[:, 0:1]
    if p.size(-1) > 1: p = p[:, 0:1]

    abs_err = (p - t).abs().squeeze(-1)
    mape_err = (abs_err / t.squeeze(-1).clamp_min(1e-4)) * 100.0

    # 芯片宽度相对误差比率 (%)
    if die_size is not None:
        if die_size.dim() == 2 and die_size.size(0) == prediction.size(0):
            b_idx = torch.arange(prediction.size(0), device=prediction.device).unsqueeze(1).expand(-1, prediction.size(1))
            valid_b_idx = b_idx[target_mask]
            w_valid = die_size[valid_b_idx, 0].clamp_min(1e-4)
        else:
            w_valid = die_size[:, 0].clamp_min(1e-4)
        die_rel = (abs_err / w_valid) * 100.0
    else:
        die_rel = abs_err * 1000.0

    # 逐图计算 X 相关性
    rx_list = []
    for b in range(prediction.size(0)):
        m = target_mask[b]
        if m.sum() >= 4:
            pb = prediction[b, m, 0] if prediction.size(-1) > 1 else prediction[b, m].squeeze(-1)
            tb = target[b, m, 0] if target.size(-1) > 1 else target[b, m].squeeze(-1)
            rx = torch.corrcoef(torch.stack([pb, tb]))[0, 1].item()
            if not torch.isnan(torch.tensor(rx)): rx_list.append(rx)

    avg_rx = sum(rx_list) / len(rx_list) if rx_list else 0.0

    return {
        "valid_nodes": int(target_mask.sum()),
        "x_mae": abs_err.mean().item(),
        "x_die_rel_percent": die_rel.mean().item(),
        "r_x": avg_rx,
        "diag_x_mape_percent": mape_err.mean().item(),
    }
