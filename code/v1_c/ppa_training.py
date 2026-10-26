"""V1-C PPA 监督训练循环。"""

import time

import torch

from .ppa_loss import ppa_loss
from .ppa_metrics import ppa_metrics


def run_ppa_epoch(model, loader, device: torch.device,
                  optimizer: torch.optim.Optimizer | None = None,
                  log_interval: float = 5.0,
                  critical_threshold: float = 0.0) -> dict[str, float]:
    """运行一个 PPA epoch，并按 5 秒节流输出进度。"""
    training = optimizer is not None
    model.train(training)
    metric_names = ("loss", "node_loss", "relation_class_loss", "timing_loss",
                    "relation_acc", "dense_relation_acc", "non_null_relation_acc",
                    "edge_precision", "edge_recall", "edge_f1", "critical_acc",
                    "node_within_1std", "timing_within_1std")
    totals = {key: 0.0 for key in metric_names}
    next_log, steps = time.monotonic() + log_interval, 0
    for steps, batch in enumerate(loader, 1):
        batch = {key: value.to(device) for key, value in batch.items()}
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                  enabled=device.type == "cuda")
        with torch.set_grad_enabled(training), autocast:
            node_pred, relation_logits, relation_pred = model(
                batch["nodes"], batch["relations"], batch["node_mask"],
                return_ppa=True, ppa_only=True)
            loss, values = ppa_loss(
                node_pred, batch["node_target"], batch["node_valid"], relation_logits,
                relation_pred, batch["relation_class"], batch["relation_class_mask"],
                batch["relation_target"], batch["relation_valid"])
            metrics = ppa_metrics(
                node_pred, batch["node_target"], batch["node_valid"], relation_logits,
                relation_pred, batch["relation_class"], batch["relation_class_mask"],
                batch["relation_dense_mask"],
                batch["relation_target"], batch["relation_valid"], critical_threshold)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        totals["loss"] += loss.item()
        for key, value in values.items():
            totals[key] += value.item()
        for key, value in metrics.items():
            totals[key] += value.item()
        if training and time.monotonic() >= next_log:
            print(f"progress batch={steps}/{len(loader)} loss={loss.item():.4e}", flush=True)
            next_log = time.monotonic() + log_interval
    count = max(steps, 1)
    return {key: value / count for key, value in totals.items()}
