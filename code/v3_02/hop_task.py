"""可达节点对跳数回归；与独立动力学分离的任务模块。"""

import torch
from torch.nn import functional as F


def reachable_mask(target, pair_mask, sentinel):
    """排除 padding 和不可达哨兵，保留自身零跳监督。"""
    return pair_mask.bool() & (target != sentinel)


def hop_loss(prediction, target, pair_mask, sentinel):
    """沿用 beta=1 的 Smooth L1，无 hop 重加权或额外辅助 loss。"""
    valid = reachable_mask(target, pair_mask, sentinel)
    if not valid.any():
        raise ValueError("no reachable pairs for supervision")
    return F.smooth_l1_loss(prediction[valid], target[valid].to(prediction.dtype))


@torch.no_grad()
def positive_hop_mape(prediction, target, pair_mask, sentinel):
    """正跳数 pair 的平均绝对百分比误差；无正跳数时返回 None。"""
    valid = reachable_mask(target, pair_mask, sentinel) & (target > 0)
    if not valid.any():
        return None
    error = (prediction[valid] - target[valid]).abs() / target[valid]
    return error.mean().item() * 100


def format_mape(value):
    """日志明确百分比单位；无有效分母时避免打印虚假的零误差。"""
    return "n/a" if value is None else f"{value:.2f}%"


@torch.no_grad()
def hop_metrics(prediction, target, pair_mask, sentinel):
    """以正跳数 MAPE 为主；0 跳单列 MAE，全部可达 pair 仍统计 MAE。"""
    valid = reachable_mask(target, pair_mask, sentinel)
    p, y = prediction[valid], target[valid]
    if y.numel() == 0:
        raise ValueError("no reachable pairs for metrics")
    error, rounded = (p - y).abs(), p.round().long()
    values, counts = rounded.unique(return_counts=True)
    per_hop = {}
    for hop in y.unique().tolist():
        selected = y == hop
        per_hop[str(hop)] = {"count": int(selected.sum()),
            "mae": error[selected].mean().item(),
            "mape_percent": error[selected].mean().item() / hop * 100 if hop > 0 else None,
            "accuracy": (rounded[selected] == y[selected]).float().mean().item(),
            "prediction_mean": p[selected].mean().item()}
    return {"reachable_pairs": y.numel(), "mae": error.mean().item(),
            "mape_percent": positive_hop_mape(prediction, target, pair_mask, sentinel),
            "positive_hop_pairs": int((y > 0).sum()), "zero_hop_pairs": int((y == 0).sum()),
            "zero_hop_mae": error[y == 0].mean().item() if (y == 0).any() else None,
            "rounded_exact_accuracy": (rounded == y).float().mean().item(),
            "prediction_min": p.min().item(), "prediction_max": p.max().item(),
            "rounded_prediction_counts": dict(zip(map(str, values.tolist()), counts.tolist())),
            "per_hop": per_hop}
