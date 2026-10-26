"""V0 公式 4：可替换的相似度损失。"""

import torch
from torch import nn
from torch.nn import functional as F


class SimilarityLoss(nn.Module):
    """实现负相似度形式，但由调用者注入具体 similar。"""

    def forward(self, nodes_out: torch.Tensor, nodes_target: torch.Tensor,
                relations_out: torch.Tensor, relations_target: torch.Tensor,
                node_mask: torch.Tensor, similarity) -> torch.Tensor:
        """计算节点项与关系项的负相似度之和。"""
        node_score = similarity(nodes_out, nodes_target, node_mask.unsqueeze(-1))
        pair_mask = (node_mask[:, :, None] & node_mask[:, None, :]).unsqueeze(-1)
        relation_score = similarity(relations_out, relations_target, pair_mask)
        return -node_score - relation_score


class ReconstructionMSELoss(nn.Module):
    """使用 masked MSE 作为当前连续 V0 的训练主损失。"""

    def forward(self, nodes_out: torch.Tensor, nodes_target: torch.Tensor,
                relations_out: torch.Tensor, relations_target: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        """分别计算节点和有效有序节点对的平均 MSE。"""
        node_mask = node_mask.unsqueeze(-1)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        node_loss = masked_mse(nodes_out, nodes_target, node_mask)
        relation_loss = masked_mse(relations_out, relations_target, pair_mask)
        return node_loss + relation_loss


def masked_mse(prediction: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """仅在 mask 标记的元素上计算均方误差。"""
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(-1)
    valid = mask.expand_as(prediction).to(prediction.dtype)
    squared_error = (prediction - target).square()
    return (squared_error * valid).sum() / valid.sum().clamp_min(1.0)


def cosine_similarity(prediction: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """提供标准余弦相似度作为可替换接口的测试实现。"""
    score = F.cosine_similarity(prediction, target, dim=-1)
    valid = mask.squeeze(-1).to(score.dtype)
    return (score * valid).sum() / valid.sum().clamp_min(1.0)


def relation_similarity_loss(prediction: torch.Tensor, target: torch.Tensor,
                             node_mask: torch.Tensor) -> torch.Tensor:
    """计算有效关系对的可导余弦相近度损失。"""
    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    return 1.0 - cosine_similarity(prediction, target, pair_mask.unsqueeze(-1))


def v0_reconstruction_loss(nodes_out: torch.Tensor, nodes_target: torch.Tensor,
                           relations_out: torch.Tensor, relations_target: torch.Tensor,
                           node_mask: torch.Tensor, similarity_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 MSE 主项、关系余弦辅助项和可记录的分项。"""
    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    node_loss = masked_mse(nodes_out, nodes_target, node_mask)
    relation_loss = masked_mse(relations_out, relations_target, pair_mask)
    similarity_loss = relation_similarity_loss(relations_out, relations_target, node_mask)
    total = node_loss + relation_loss + similarity_weight * similarity_loss
    return total, {"node_loss": node_loss, "relation_loss": relation_loss,
                   "similarity_loss": similarity_loss}
