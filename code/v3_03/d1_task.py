"""V3.03 D1 时序预测任务：Log-Space 掩码感知标准化 Huber 损失与多维物理诊断指标。"""

import torch
from torch.nn import functional as F

LOG_EPS = 1e-12


def d1_loss(norm_pred, target, target_mask, log_mean, log_std):
    """在对数标准化空间计算掩码感知的 Smooth L1 损失。"""
    if not target_mask.any():
        raise ValueError("batch has no valid timing targets")
    # 将真实物理延时映射到对数标准化空间：y_norm = (log(y) - mu) / sigma
    m = target_mask
    norm_t = (torch.log(target.clamp_min(LOG_EPS)) - log_mean) / log_std
    return F.smooth_l1_loss(norm_pred[m], norm_t[m])


@torch.no_grad()
def d1_metrics(norm_pred, target, target_mask, log_mean, log_std):
    """还原到物理空间计算 MAE、MAPE、R2、Pearson r、预测与真实方差及方差比。"""
    if not target_mask.any():
        return {
            "mae_ns": 0.0, "mape_percent": 0.0, "r_timing": 0.0,
            "r2_timing": 0.0, "pred_std": 0.0, "true_std": 0.0,
            "std_ratio": 0.0, "valid_nodes": 0
        }

    # 还原回物理纳秒尺度：p_raw = exp(p_norm * sigma + mu)
    raw_p = torch.exp(norm_pred * log_std + log_mean)
    m = target_mask
    p_valid = raw_p[m]
    t_valid = target[m]

    abs_err = (p_valid - t_valid).abs()
    mape = (abs_err / t_valid.clamp_min(1e-4)) * 100.0

    p_std = p_valid.std().item() if p_valid.numel() > 1 else 0.0
    t_std = t_valid.std().item() if t_valid.numel() > 1 else 0.0
    std_ratio = p_std / max(t_std, 1e-6)

    # 逐图评估首通道的相关系数 r 与 R2 决定系数
    r_list, r2_list = [], []
    for b in range(norm_pred.size(0)):
        mb = target_mask[b, :, 0]
        if mb.sum() >= 4:
            pb = raw_p[b, mb, 0]
            tb = target[b, mb, 0]
            # Pearson r
            r = torch.corrcoef(torch.stack([pb, tb]))[0, 1].item()
            if not torch.isnan(torch.tensor(r)):
                r_list.append(r)
            # R2 决定系数: 1 - SS_res / SS_tot
            ss_res = ((tb - pb) ** 2).sum()
            ss_tot = ((tb - tb.mean()) ** 2).sum().clamp_min(1e-6)
            r2 = (1.0 - ss_res / ss_tot).item()
            r2_list.append(r2)

    avg_r = sum(r_list) / len(r_list) if r_list else 0.0
    avg_r2 = sum(r2_list) / len(r2_list) if r2_list else 0.0

    return {
        "valid_nodes": int(target_mask.sum()),
        "mae_ns": abs_err.mean().item(),
        "mape_percent": mape.mean().item(),
        "r_timing": avg_r,
        "r2_timing": avg_r2,
        "pred_std": p_std,
        "true_std": t_std,
        "std_ratio": std_ratio,
    }
