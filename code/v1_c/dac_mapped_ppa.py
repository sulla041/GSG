"""DAC mapped-v2 area/power multi-task loss、指标和训练循环。"""

import time

import torch
from torch.nn import functional as F


LABELS = ("area", "internal", "switching", "leakage", "total_power")


def ppa_loss(prediction, target, valid, mean, std):
    """计算五个 mapped-cell 连续 PPA 通道的 masked 标准化损失。"""
    normalized = (target - mean) / std
    error = F.smooth_l1_loss(prediction[..., :5], normalized, reduction="none")
    mask = valid.to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def _channel_metrics(prediction, target, valid, mean, std):
    """返回每个 mapped-cell PPA 通道的原始尺度 MAE/RMSE。"""
    raw = prediction[..., :5] * std + mean
    result = {}
    for index, name in enumerate(LABELS):
        delta = (raw[..., index] - target[..., index]).masked_select(valid[..., index])
        result[f"{name}_mae"] = delta.abs().mean() if delta.numel() else raw.sum() * 0.0
        result[f"{name}_rmse"] = delta.square().mean().sqrt() if delta.numel() else result[f"{name}_mae"]
    return result, raw


def ppa_metrics(prediction, target, valid, graph_area, graph_power,
                node_mask, mean, std):
    """返回节点 PPA 指标和 area/total-power 图级相对误差。"""
    metrics, raw = _channel_metrics(prediction, target, valid, mean, std)
    area_pred = (raw[..., 0] * valid[..., 0] * node_mask).sum(-1)
    power_pred = (raw[..., 4] * valid[..., 4] * node_mask).sum(-1)
    metrics["graph_area_relative_error"] = (
        (area_pred - graph_area).abs() / graph_area.abs().clamp_min(1e-12)).mean()
    metrics["graph_power_relative_error"] = (
        (power_pred - graph_power).abs() / graph_power.abs().clamp_min(1e-12)).mean()
    return metrics


def run_ppa_epoch(model, loader, device, mean, std, optimizer=None,
                  log_interval=5.0):
    """运行一个 mapped PPA epoch，训练进度按五秒节流。"""
    training = optimizer is not None
    model.train(training)
    names = ["loss"] + [f"{x}_{y}" for x in LABELS for y in ("mae", "rmse")]
    names += ["graph_area_relative_error", "graph_power_relative_error"]
    totals = {name: 0.0 for name in names}
    next_log, steps = time.monotonic() + log_interval, 0
    for steps, batch in enumerate(loader, 1):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction, _, _ = model(batch["nodes"], batch["relations"],
                                     batch["node_mask"], return_ppa=True, ppa_only=True)
            loss = ppa_loss(prediction, batch["node_target"], batch["node_valid"],
                            mean, std)
            metrics = ppa_metrics(prediction, batch["node_target"], batch["node_valid"],
                                  batch["graph_area_um2"], batch["graph_total_power_w"],
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
