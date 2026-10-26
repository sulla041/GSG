"""V1-C virtual timing relation 的 masked 单任务训练工具。"""

import time

import torch
from torch.nn import functional as F


def target_stats(dataset, channel: int, log_target: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """只用 train split 的有效 virtual relation 拟合尺度统计。"""
    values = []
    for design_id in dataset.ids:
        label = torch.load(dataset.label_root / design_id / "labels.pt",
                           map_location="cpu", weights_only=False)
        value = label["relation_target"][..., channel][label["relation_valid"]]
        if log_target:
            value = torch.log1p(value.clamp_min(0.0) * 1e9)
        values.append(value)
    merged = torch.cat(values)
    return merged.mean(), merged.std().clamp_min(1e-8)


def _loss(prediction, target, valid, channel, mean, std, log_target):
    """计算一个 virtual relation 通道的 masked Huber/BCE loss。"""
    truth = target[..., channel]
    prediction = prediction[..., 0]
    if log_target:
        truth = (torch.log1p(truth.clamp_min(0.0) * 1e9) - mean) / std
        error = F.smooth_l1_loss(prediction, truth, reduction="none")
    elif channel == 2:
        error = F.binary_cross_entropy_with_logits(prediction, truth, reduction="none")
    else:
        truth = (truth - mean) / std
        error = F.smooth_l1_loss(prediction, truth, reduction="none")
    mask = valid.to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def _metrics(prediction, target, valid, channel, mean, std, log_target):
    """还原到物理空间并计算 MAE、RMSE 和相对误差。"""
    truth = target[..., channel]
    prediction = prediction[..., 0]
    if channel == 2:
        raw = prediction.sigmoid()
    elif log_target:
        raw = (prediction * std + mean).expm1().clamp_min(0.0) / 1e9
    else:
        raw = prediction * std + mean
    delta = (raw - truth).masked_select(valid)
    mae = delta.abs().mean() if delta.numel() else raw.sum() * 0.0
    rmse = delta.square().mean().sqrt() if delta.numel() else mae
    ratio_mask = valid & truth.ne(0)
    ratio = ((raw - truth).abs() / truth.abs().clamp_min(1e-12)).masked_select(ratio_mask)
    return {"mae": mae, "rmse": rmse,
            "relative_error": ratio.mean() if ratio.numel() else raw.sum() * 0.0,
            "valid_relations": valid.sum().to(raw.dtype)}


def run_virtual_epoch(model, loader, device, channel, mean, std, log_target,
                      optimizer=None, log_interval=5.0):
    """运行一个 virtual timing epoch，保持现有五秒进度输出协议。"""
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "mae", "rmse", "relative_error", "valid_relations")}
    next_log, steps = time.monotonic() + log_interval, 0
    for steps, batch in enumerate(loader, 1):
        batch = {key: value.to(device, non_blocking=device.type == "cuda")
                 for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = model(batch["nodes"], batch["relations"], batch["node_mask"],
                               virtual_timing_only=True)
            loss = _loss(prediction, batch["relation_target"], batch["relation_valid"],
                         channel, mean, std, log_target)
            metrics = _metrics(prediction, batch["relation_target"], batch["relation_valid"],
                               channel, mean, std, log_target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals["loss"] += loss.item()
        for key in metrics:
            totals[key] += metrics[key].item()
        if training and time.monotonic() >= next_log:
            print(f"progress batch={steps}/{len(loader)} loss={loss.item():.4e}", flush=True)
            next_log = time.monotonic() + log_interval
    count = max(steps, 1)
    return {key: value / count for key, value in totals.items()}
