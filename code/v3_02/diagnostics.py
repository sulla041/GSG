"""只读逐步诊断；所有观测张量脱离梯度图。"""

import torch


@torch.no_grad()
def step_metrics(before, after, field, pair_contributions, gate, injected, scale, mask, eta,
                 interaction_details=None):
    """区分确定性场、真实更新、噪声、中心化几何与消息抵消。"""
    nodes = mask.sum().clamp_min(1)
    coordinates = nodes * before.size(-1)
    pairs = mask[:, :, None] & mask[:, None, :]
    count = pairs.sum().clamp_min(1)
    center = after.sum(1, keepdim=True) / mask.sum(1).clamp_min(1)[:, None, None]
    centered = (after - center).masked_fill(~mask[..., None], 0)
    magnitude = pair_contributions.norm(dim=-1).sum(1)
    cancellation = pair_contributions.sum(1).norm(dim=-1) / magnitude.clamp_min(1e-12)
    displacement = after - before
    result = {
        "field_rms": (field.square().sum() / coordinates).sqrt(),
        "interaction_rms": ((eta * field).square().sum() / coordinates).sqrt(),
        "update_rms": (displacement.square().sum() / coordinates).sqrt(),
        "node_displacement_mean": displacement.norm(dim=-1).masked_select(mask).mean(),
        "node_displacement_max": displacement.norm(dim=-1).masked_fill(~mask, 0).max(),
        "node_field_displacement_mean": (eta * field).norm(dim=-1).masked_select(mask).mean(),
        "noise_rms": (injected.square().sum() / coordinates).sqrt(),
        "coordinate_rms": (after.square().sum() / coordinates).sqrt(),
        "centered_coordinate_rms": (centered.square().sum() / coordinates).sqrt(),
        "gate_mean": (gate.squeeze(-1) * pairs).sum() / count,
        "gate_saturation": (((gate.squeeze(-1) < .01) | (gate.squeeze(-1) > .99))
                            & pairs).sum() / count,
        "pair_contribution_cancellation": (cancellation * mask).sum() / nodes,
        "noise_scale_mean": scale.sum() / nodes,
        "noise_zero_fraction": ((scale == 0) & mask).sum() / nodes,
    }
    if interaction_details:
        result.update(interaction_details)
    return result
