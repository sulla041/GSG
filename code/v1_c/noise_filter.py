"""V1 候选：按点集聚拢程度调节逐层噪声。"""

import torch
from torch import nn


class AdaptiveNoiseFilter(nn.Module):
    """聚拢节点获得更强噪声，分离节点获得更弱噪声。"""

    def __init__(self, temperature: float = 1.0, max_scale: float = 1e-2) -> None:
        """保存高斯相似度温度和噪声上限。"""
        super().__init__()
        self.temperature, self.max_scale = temperature, max_scale

    def forward(self, points: torch.Tensor, node_mask: torch.Tensor,
                noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """返回加噪点集和节点噪声强度。"""
        if noise is None:
            noise = torch.randn_like(points)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        eye = torch.eye(points.size(1), device=points.device, dtype=torch.bool)[None]
        peer_mask = pair_mask & ~eye
        distance = (points[:, :, None] - points[:, None, :]).square().mean(dim=-1)
        similarity = torch.exp(-distance / self.temperature) * peer_mask
        concentration = similarity.sum(-1) / peer_mask.sum(-1).clamp_min(1)
        scale = self.max_scale * concentration
        return (points + scale.unsqueeze(-1) * noise) * node_mask.unsqueeze(-1), scale
