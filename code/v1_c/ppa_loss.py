"""PPA 监督损失：显式使用节点、关系类别和 timing-arc 掩码。"""

import torch
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只对有效监督元素求平均，空掩码返回可反传的零。"""
    valid = mask.to(values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def _class_weights(target: torch.Tensor, mask: torch.Tensor,
                   classes: int) -> torch.Tensor:
    """用 inverse-sqrt 频率缓解稠密 r_null 的多数类偏置。"""
    values = target[mask]
    counts = torch.bincount(values, minlength=classes).to(torch.float32)
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].rsqrt()
    return weights / weights[present].mean().clamp_min(1e-12)


def ppa_loss(node_prediction: torch.Tensor, node_target: torch.Tensor,
             node_valid: torch.Tensor, relation_logits: torch.Tensor,
             relation_prediction: torch.Tensor, relation_class: torch.Tensor,
             relation_class_mask: torch.Tensor, relation_target: torch.Tensor,
             relation_valid: torch.Tensor, node_weight: float = 1.0,
             relation_weight: float = 1.0, timing_weight: float = 1.0):
    """返回总损失和分项指标，所有连续目标使用 SmoothL1。"""
    node_error = F.smooth_l1_loss(node_prediction, node_target, reduction="none")
    node_loss = _masked_mean(node_error, node_valid)
    classes = relation_logits.size(-1)
    class_error = F.cross_entropy(relation_logits.permute(0, 3, 1, 2), relation_class,
                                  weight=_class_weights(relation_class, relation_class_mask, classes),
                                  reduction="none")
    class_loss = _masked_mean(class_error, relation_class_mask)
    relation_error = F.smooth_l1_loss(relation_prediction, relation_target, reduction="none")
    timing_loss = _masked_mean(relation_error, relation_valid)
    total = node_weight * node_loss + relation_weight * class_loss + timing_weight * timing_loss
    return total, {"node_loss": node_loss, "relation_class_loss": class_loss,
                   "timing_loss": timing_loss}
