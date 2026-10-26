"""V1 终态点集上的节点和关系监督读出。"""

import torch
from torch import nn


class NodeRelationPredictor(nn.Module):
    """从 S_e 的独立终态读出预测节点类别和目标关系。"""

    def __init__(self, encoder: nn.Module, node_class_count: int) -> None:
        """创建节点任务头，关系任务复用独立 PAIRATTN_out。"""
        super().__init__()
        self.encoder = encoder
        width = encoder.decoder.setattn.norm.normalized_shape[0]
        self.node_head = nn.Linear(width, node_class_count)

    def forward(self, nodes: torch.Tensor, relations: torch.Tensor,
                node_mask: torch.Tensor, noises=None):
        """返回节点 logits、关系 logits 和终态点集。"""
        points, _, _, _, _ = self.encoder(nodes, relations, node_mask, noises)
        context = self.encoder.decoder.setattn(points, node_mask)
        node_logits = self.node_head(context) * node_mask.unsqueeze(-1)
        relation_logits = self.encoder.decoder.pairattn(points, node_mask)
        return node_logits, relation_logits, points


def masked_task_loss(node_logits: torch.Tensor, relation_logits: torch.Tensor,
                     batch: dict[str, torch.Tensor],
                     relation_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """仅在有效节点和有效有序点对上计算监督交叉熵。"""
    node_mask = batch["node_mask"]
    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    task_mask = batch["task_relation_mask"] & pair_mask
    node_loss = nn.functional.cross_entropy(node_logits[node_mask], batch["node_targets"][node_mask])
    relation_loss = nn.functional.cross_entropy(
        relation_logits[task_mask], batch["relation_targets"][task_mask],
        weight=None if relation_weight is None else relation_weight.to(relation_logits.device)
    )
    node_accuracy = (node_logits[node_mask].argmax(-1) == batch["node_targets"][node_mask]).float().mean()
    relation_accuracy = (relation_logits[task_mask].argmax(-1) == batch["relation_targets"][task_mask]).float().mean()
    relation_prediction = relation_logits[task_mask].argmax(-1)
    relation_target = batch["relation_targets"][task_mask]
    positive_target = relation_target == 1
    positive_prediction = relation_prediction == 1
    true_positive = (positive_target & positive_prediction).sum().float()
    edge_precision = true_positive / positive_prediction.sum().clamp_min(1)
    edge_recall = true_positive / positive_target.sum().clamp_min(1)
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall).clamp_min(1e-8)
    hidden_mask = batch["hidden_edge_mask"]
    hidden_prediction = relation_logits[hidden_mask].argmax(-1) == 1
    hidden_recall = (hidden_prediction.float().mean() if hidden_prediction.numel()
                     else torch.zeros((), device=node_logits.device))
    return node_loss + relation_loss, {
        "node_loss": node_loss.item(), "relation_loss": relation_loss.item(),
        "node_accuracy": node_accuracy.item(), "relation_accuracy": relation_accuracy.item(),
        "edge_precision": edge_precision.item(), "edge_recall": edge_recall.item(),
        "edge_f1": edge_f1.item(),
        "hidden_edge_recall": hidden_recall.item(),
    }
