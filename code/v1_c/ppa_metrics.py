"""PPA 监督训练的分类准确率和连续值容差准确率。"""

import torch


def _accuracy(correct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算掩码下的准确率，空掩码返回零。"""
    valid = mask.to(correct.dtype)
    return (correct.to(valid.dtype) * valid).sum() / valid.sum().clamp_min(1.0)


def ppa_metrics(node_prediction: torch.Tensor, node_target: torch.Tensor,
                node_valid: torch.Tensor, relation_logits: torch.Tensor,
                relation_prediction: torch.Tensor, relation_class: torch.Tensor,
                relation_class_mask: torch.Tensor, relation_dense_mask: torch.Tensor,
                relation_target: torch.Tensor,
                relation_valid: torch.Tensor, critical_threshold: float = 0.0):
    """返回关系分类 accuracy、critical accuracy 和连续值一标准差准确率。"""
    relation_guess = relation_logits.argmax(-1)
    relation_acc = _accuracy(relation_guess == relation_class, relation_class_mask)
    dense_relation_acc = _accuracy(relation_guess == relation_class, relation_dense_mask)
    edge_mask = relation_class_mask & (relation_class != 0)
    edge_acc = _accuracy(relation_guess == relation_class, edge_mask)
    critical_guess = relation_prediction[..., 2] >= critical_threshold
    critical_target = relation_target[..., 2] >= critical_threshold
    critical_acc = _accuracy(critical_guess == critical_target, relation_valid[..., 2])
    predicted_edge = relation_guess != 0
    target_edge = relation_class != 0
    true_positive = (predicted_edge & target_edge & relation_class_mask).sum().float()
    precision = true_positive / (predicted_edge & relation_class_mask).sum().clamp_min(1)
    recall = true_positive / (target_edge & relation_class_mask).sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    node_close = (node_prediction - node_target).abs() <= 1.0
    timing_close = (relation_prediction - relation_target).abs() <= 1.0
    return {"relation_acc": relation_acc, "dense_relation_acc": dense_relation_acc,
            "non_null_relation_acc": edge_acc, "edge_precision": precision,
            "edge_recall": recall, "edge_f1": f1,
            "critical_acc": critical_acc,
            "node_within_1std": _accuracy(node_close, node_valid),
            "timing_within_1std": _accuracy(timing_close, relation_valid)}
