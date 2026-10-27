"""V2-EX dense reachability 最终监督与分 hop 诊断工具。"""

import torch
from torch.nn import functional as F


def reachable_loss(logits, target, pair_mask, positive_weight_cap=20.0):
    """只对有效有序 pair 计算 balanced BCE，不监督 hop 数值。"""
    logits = logits[..., 1] - logits[..., 0]
    valid = pair_mask.bool()
    truth = target.bool()
    positive = (truth & valid).sum().to(logits.dtype).clamp_min(1)
    negative = ((~truth) & valid).sum().to(logits.dtype)
    weight = (negative / positive).clamp(1.0, positive_weight_cap)
    per_pair = F.binary_cross_entropy_with_logits(
        logits, truth.to(logits.dtype), pos_weight=weight, reduction="none"
    )
    return (per_pair * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def reach_metrics(logits, target, pair_mask):
    """返回 masked accuracy、precision、recall 和 F1。"""
    prediction = logits[..., 1] > logits[..., 0]
    truth, valid = target.bool(), pair_mask.bool()
    tp = (prediction & truth & valid).sum().float()
    fp = (prediction & ~truth & valid).sum().float()
    fn = (~prediction & truth & valid).sum().float()
    correct = ((prediction == truth) & valid).sum().float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {"accuracy": (correct / valid.sum().clamp_min(1)).item(),
            "precision": precision.item(), "recall": recall.item(), "f1": f1.item()}
