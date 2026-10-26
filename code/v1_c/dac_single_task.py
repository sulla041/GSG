"""DAC mapped-v2 单一连续节点任务的 loss、指标和训练循环。"""

import time

import torch
from torch.nn import functional as F

LOG_TARGET_EPS = 1e-12


def task_loss(prediction, target, valid, channel, mean, std):
    """在 log-space 计算指定 mapped-cell 标签的 masked Huber loss。"""
    normalized = (torch.log(target[..., channel].clamp_min(LOG_TARGET_EPS)) - mean) / std
    error = F.smooth_l1_loss(prediction[..., channel], normalized, reduction="none")
    mask = valid[..., channel].to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def task_metrics(prediction, target, valid, node_mask, channel, mean, std):
    """将 log-space 预测还原后计算原始尺度误差和相对误差。"""
    raw = (prediction[..., channel] * std + mean).exp()
    mask = valid[..., channel] & node_mask
    truth = target[..., channel]
    delta = (raw - truth).masked_select(mask)
    mae = delta.abs().mean() if delta.numel() else raw.sum() * 0.0
    rmse = delta.square().mean().sqrt() if delta.numel() else mae
    ratio_mask = mask & truth.ne(0)
    ratio = ((raw - truth).abs() / truth.abs().clamp_min(1e-12)).masked_select(ratio_mask)
    relative_error = ratio.mean() if ratio.numel() else raw.sum() * 0.0
    return {"mae": mae, "rmse": rmse, "relative_error": relative_error}


def run_task_epoch(model, loader, device, channel, mean, std, optimizer=None,
                   log_interval=5.0, measure_prediction_time=False):
    """运行一个单任务 epoch，训练时按五秒节流打印。"""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "relative_error": 0.0}
    next_log, steps = time.monotonic() + log_interval, 0
    inference_seconds = 0.0
    design_count = node_count = valid_node_count = 0
    for steps, batch in enumerate(loader, 1):
        batch = {key: value.to(device, non_blocking=device.type == "cuda")
                 for key, value in batch.items()}
        if measure_prediction_time and device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter() if measure_prediction_time else 0.0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction, _, _ = model(batch["nodes"], batch["relations"],
                                     batch["node_mask"], return_ppa=True, ppa_only=True)
        if measure_prediction_time:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start
            design_count += int(batch["node_mask"].size(0))
            node_count += int(batch["node_mask"].sum().item())
            valid_node_count += int((batch["node_valid"][..., channel]
                                     & batch["node_mask"]).sum().item())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss = task_loss(prediction, batch["node_target"], batch["node_valid"],
                             channel, mean, std)
            metrics = task_metrics(prediction, batch["node_target"], batch["node_valid"],
                                   batch["node_mask"], channel, mean, std)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals["loss"] += loss.item()
        totals["mae"] += metrics["mae"].item()
        totals["rmse"] += metrics["rmse"].item()
        totals["relative_error"] += metrics["relative_error"].item()
        if training and time.monotonic() >= next_log:
            print(f"progress batch={steps}/{len(loader)} loss={loss.item():.4e} "
                  f"relative_error={metrics['relative_error'].item():.2%}", flush=True)
            next_log = time.monotonic() + log_interval
    count = max(steps, 1)
    result = {key: value / count for key, value in totals.items()}
    if measure_prediction_time:
        total_us = inference_seconds * 1e6
        result.update({
            "inference_time_ms_total": inference_seconds * 1e3,
            "inference_time_us_per_design": total_us / max(design_count, 1),
            "inference_time_us_per_node": total_us / max(node_count, 1),
            "inference_time_us_per_valid_node": total_us / max(valid_node_count, 1),
        })
    return result
