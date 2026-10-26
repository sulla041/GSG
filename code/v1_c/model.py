"""V1：V0 关系演化结构加独立逐层自适应噪声。"""

import torch
from torch import nn

from .decoder import Decoder
from .encr import ENCR
from .encs import ENCS
from .evolution import EvolutionLayer
from .noise_filter import AdaptiveNoiseFilter
from .ppa import PPAHead
from .design_task import DesignTaskHead
from .node_task import NodeTaskHead
from .relation_task import RelationTaskHead
from .virtual_timing_head import VirtualTimingHead


class V1CModel(nn.Module):
    """每个 V1 层拥有独立演化参数和噪声滤波器。"""

    def __init__(self, node_width: int, relation_width: int, point_width: int,
                 pair_width: int, hidden_width: int, layers: int,
                 temperature: float = 1.0, max_noise: float = 1e-2) -> None:
        """创建 V1；只复用 V0 算子类定义，不共享 V0 参数。"""
        super().__init__()
        self.encs = ENCS(node_width, point_width, hidden_width)
        self.encr = ENCR(relation_width, pair_width, hidden_width)
        self.layers = nn.ModuleList([EvolutionLayer(point_width, pair_width, hidden_width)
                                     for _ in range(layers)])
        self.noise_filters = nn.ModuleList([
            AdaptiveNoiseFilter(temperature, max_noise) for _ in range(layers)
        ])
        self.decoder = Decoder(point_width, node_width, pair_width, hidden_width, relation_width)
        self.ppa = PPAHead(point_width, pair_width, hidden_width, 10, 4, 3)
        self.design_task = DesignTaskHead(point_width, pair_width, hidden_width)
        self.node_task = NodeTaskHead(point_width, pair_width, hidden_width)
        self.relation_task = RelationTaskHead(point_width, pair_width, hidden_width)
        self.virtual_timing = VirtualTimingHead(point_width, relation_width,
                                                pair_width, hidden_width)

    def forward(self, nodes: torch.Tensor, relations: torch.Tensor,
                node_mask: torch.Tensor, noises: tuple[torch.Tensor, ...] | None = None,
                return_ppa: bool = False, ppa_only: bool = False,
                task_only: bool = False, node_task_only: bool = False,
                relation_task_only: bool = False, virtual_timing_only: bool = False):
        """返回终态、重构输出、偏移和逐层噪声强度。"""
        points, relation_state = self.encs(nodes), self.encr(relations)
        deltas, scales = [], []
        for index, (layer, noise_filter) in enumerate(zip(self.layers, self.noise_filters)):
            points, delta = layer(points, relation_state, node_mask)
            noise = None if noises is None else noises[index]
            points, scale = noise_filter(points, node_mask, noise)
            deltas.append(delta)
            scales.append(scale)
        if return_ppa and ppa_only:
            return self.ppa(points, node_mask)
        if task_only:
            return self.design_task(points, node_mask)
        if node_task_only:
            return self.node_task(points, node_mask)
        if relation_task_only:
            return self.relation_task(points, node_mask)
        if virtual_timing_only:
            return self.virtual_timing(points, relations, node_mask)
        nodes_out, relations_out = self.decoder(points, node_mask)
        output = (points, nodes_out, relations_out, tuple(deltas), tuple(scales))
        if not return_ppa:
            return output
        return output + self.ppa(points, node_mask)
