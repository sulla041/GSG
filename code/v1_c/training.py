"""V1 训练、验证和 checkpoint 工具。"""

from pathlib import Path
import time

import torch
from torch import nn

from .loss import SimilarityLoss, cosine_similarity, v0_reconstruction_loss


def run_epoch(model: nn.Module, loader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None,
              similarity_weight: float = 0.1, log_interval: float = 5.0) -> dict[str, float]:
    """运行 V1 epoch，并返回重构、相近度和平均噪声强度。"""
    training = optimizer is not None
    model.train(training)
    metric = SimilarityLoss()
    totals = {"loss": 0.0, "node_loss": 0.0, "relation_loss": 0.0,
              "similarity_loss": 0.0, "cosine": 0.0, "noise": 0.0}
    next_log = time.monotonic() + log_interval
    for step, raw in enumerate(loader, 1):
        batch = {name: value.to(device) for name, value in raw.items()}
        with torch.set_grad_enabled(training):
            _, nodes_out, relations_out, _, scales = model(
                batch["nodes"], batch["relations"], batch["node_mask"]
            )
            loss, values = v0_reconstruction_loss(
                nodes_out, batch["nodes"], relations_out, batch["relations"],
                batch["node_mask"], similarity_weight
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        cosine = metric(nodes_out, batch["nodes"], relations_out,
                        batch["relations"], batch["node_mask"], cosine_similarity)
        totals["loss"] += loss.item()
        totals["node_loss"] += values["node_loss"].item()
        totals["relation_loss"] += values["relation_loss"].item()
        totals["similarity_loss"] += values["similarity_loss"].item()
        totals["cosine"] += cosine.item()
        totals["noise"] += torch.cat(scales, dim=-1).mean().item()
        if training and time.monotonic() >= next_log:
            print(f"progress batch={step}/{len(loader)} loss={loss.item():.4e}", flush=True)
            next_log = time.monotonic() + log_interval
    count = max(len(loader), 1)
    return {name: value / count for name, value in totals.items()}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: dict[str, float]) -> None:
    """保存 V1 模型、优化器和验证指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "metrics": metrics}, path)
