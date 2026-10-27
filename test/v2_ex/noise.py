"""根据节点聚拢度单调退出的对称性破除噪声。"""

import torch
from torch import nn


class SymmetryNoise(nn.Module):
    """按当前聚拢度提议噪声，并受上一步尺度上限约束。"""

    def __init__(self, temperature=1.0, noise_max=1e-2, stop_scale=1e-6):
        """保存噪声调度参数，均为可登记的实验配置。"""
        super().__init__()
        self.temperature = temperature
        self.noise_max = noise_max
        self.stop_scale = stop_scale

    def forward(self, points, node_mask, previous_scale, noise=None):
        """返回噪声扰动状态、新尺度和实际注入噪声。"""
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        eye = torch.eye(points.size(1), device=points.device, dtype=torch.bool)[None]
        peer_mask = pair_mask & ~eye
        distance = (points[:, :, None] - points[:, None, :]).square().mean(-1)
        similarity = torch.exp(-distance / self.temperature) * peer_mask
        concentration = similarity.sum(-1) / peer_mask.sum(-1).clamp_min(1)
        proposed = self.noise_max * concentration
        scale = torch.minimum(previous_scale, proposed)
        scale = torch.where(scale >= self.stop_scale, scale, torch.zeros_like(scale))
        scale = scale * node_mask
        if noise is None:
            noise = torch.randn_like(points)
        injected = scale.unsqueeze(-1) * noise
        return (points + injected) * node_mask.unsqueeze(-1), scale, injected
