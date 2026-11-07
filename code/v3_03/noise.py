"""独立复制既有自适应噪声：聚拢度驱动、单调退出。"""

import torch
from torch import nn


class SymmetryNoise(nn.Module):
    """保持 V2-EX 当前噪声公式与参数，不导入旧模型代码。"""

    def __init__(self, temperature=1.0, noise_max=1e-2, stop_scale=1e-6):
        super().__init__()
        self.temperature, self.noise_max = temperature, noise_max
        self.stop_scale = stop_scale

    def forward(self, points, node_mask, previous_scale, noise=None):
        """根据点间均方距离提议尺度，由单调包络控制退出。"""
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        eye = torch.eye(points.size(1), device=points.device, dtype=torch.bool)[None]
        peer_mask = pair_mask & ~eye
        distance = (points[:, :, None] - points[:, None, :]).square().mean(-1)
        concentration = (torch.exp(-distance / self.temperature) * peer_mask).sum(-1)
        concentration = concentration / peer_mask.sum(-1).clamp_min(1)
        scale = torch.minimum(previous_scale, self.noise_max * concentration)
        scale = torch.where(scale >= self.stop_scale, scale, torch.zeros_like(scale))
        scale = scale * node_mask
        if noise is None:
            noise = torch.randn_like(points)
        injected = scale[..., None] * noise
        return (points + injected).masked_fill(~node_mask[..., None], 0), scale, injected
