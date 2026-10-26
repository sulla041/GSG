"""DAC pin timing-v1 的 CELL_ARC 三通道 delay 训练循环。"""

import time

import torch
from torch.nn import functional as F


LABELS = ("rise_delay", "fall_delay", "worst_delay")


def timing_loss(prediction, target, valid, mean, std):
    """计算有效 CELL_ARC delay 的 masked 标准化 SmoothL1。"""
    valid = valid.expand_as(target)
    normalized = (target - mean) / std
    error = F.smooth_l1_loss(prediction, normalized, reduction="none")
    mask = valid.to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def timing_metrics(prediction, target, valid, mean, std):
    """返回 rise/fall/worst delay 的原始尺度 MAE/RMSE。"""
    valid = valid.expand_as(target)
    raw = prediction * std + mean
    result = {}
    for index, name in enumerate(LABELS):
        delta = (raw[..., index] - target[..., index]).masked_select(valid[..., index])
        result[f"{name}_mae"] = delta.abs().mean() if delta.numel() else raw.sum() * 0.0
        result[f"{name}_rmse"] = delta.square().mean().sqrt() if delta.numel() else result[f"{name}_mae"]
    return result


def run_timing_epoch(model, loader, device, mean, std, optimizer=None,
                     log_interval=5.0):
    """运行一个 pin timing epoch，训练进度按五秒节流。"""
    training = optimizer is not None
    model.train(training)
    names = ["loss"] + [f"{x}_{y}" for x in LABELS for y in ("mae", "rmse")]
    totals = {name: 0.0 for name in names}
    next_log, steps = time.monotonic() + log_interval, 0
    for steps, batch in enumerate(loader, 1):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            _, _, prediction = model(batch["nodes"], batch["relations"],
                                     batch["node_mask"], return_ppa=True, ppa_only=True)
            loss = timing_loss(prediction, batch["relation_target"],
                               batch["relation_valid"].unsqueeze(-1), mean, std)
            metrics = timing_metrics(prediction, batch["relation_target"],
                                     batch["relation_valid"].unsqueeze(-1), mean, std)
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
