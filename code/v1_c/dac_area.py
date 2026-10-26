"""DAC mapped-cell area downstream loss、指标和训练循环。"""

import time

import torch
from torch.nn import functional as F


def area_loss(prediction: torch.Tensor, target: torch.Tensor,
              valid: torch.Tensor, mean: float, std: float):
    """在 train 统计量下计算 masked 标准化 area SmoothL1。"""
    normalized = (target[..., 0] - mean) / std
    error = F.smooth_l1_loss(prediction[..., 0], normalized, reduction="none")
    mask = valid[..., 0].to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def area_metrics(prediction: torch.Tensor, target: torch.Tensor,
                 valid: torch.Tensor, graph_target: torch.Tensor,
                 node_mask: torch.Tensor, mean: float, std: float):
    """返回节点 normalized MAE/RMSE 和 graph area 相对误差。"""
    raw = prediction[..., 0] * std + mean
    mask = valid[..., 0] & node_mask
    delta = (raw - target[..., 0]).masked_select(mask)
    node_mae = delta.abs().mean() if delta.numel() else raw.sum() * 0.0
    node_rmse = delta.square().mean().sqrt() if delta.numel() else node_mae
    graph_pred = (raw * mask).sum(-1)
    graph_error = (graph_pred - graph_target).abs() / graph_target.abs().clamp_min(1e-12)
    return {"node_mae": node_mae, "node_rmse": node_rmse,
            "graph_relative_error": graph_error.mean()}


def run_area_epoch(model, loader, device: torch.device, mean: float, std: float,
                   optimizer=None, log_interval: float = 5.0):
    """运行一个 area epoch，训练时按五秒节流打印进度。"""
    training = optimizer is not None
    model.train(training)
    names = ("loss", "node_mae", "node_rmse", "graph_relative_error")
    totals = {name: 0.0 for name in names}
    next_log, steps = time.monotonic() + log_interval, 0
    for steps, batch in enumerate(loader, 1):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction, _, _ = model(batch["nodes"], batch["relations"],
                                     batch["node_mask"], return_ppa=True,
                                     ppa_only=True)
            loss = area_loss(prediction, batch["node_target"], batch["node_valid"],
                             mean, std)
            metrics = area_metrics(prediction, batch["node_target"],
                                   batch["node_valid"], batch["graph_area_um2"],
                                   batch["node_mask"], mean, std)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals["loss"] += loss.item()
        for name, value in metrics.items():
            totals[name] += value.item()
        if training and time.monotonic() >= next_log:
            print(f"progress batch={steps}/{len(loader)} loss={loss.item():.4e}", flush=True)
            next_log = time.monotonic() + log_interval
    count = max(steps, 1)
    return {name: value / count for name, value in totals.items()}
