"""训练 V1-C 的单一设计级 PPA 下游任务。"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "script"))

from dataset.design_task import DesignTaskDataset, load_design_task
from dataset.design_task import TASK_FIELDS
from train_v1_c_ppa import pair_paths
from v1_c.model import V1CModel


def fit_stats(pairs, liberty, task, max_nodes):
    """只用 train split 的有效标量拟合标准化参数。"""
    values = []
    for pair in pairs:
        sample = load_design_task(pair, liberty, task, max_nodes)
        if sample["task_valid"]:
            values.append(sample["task_target"].item())
    if not values:
        raise ValueError(f"no valid labels for task={task}")
    tensor = torch.tensor(values)
    return {"mean": tensor.mean().item(), "std": tensor.std(unbiased=False).clamp_min(1e-12).item()}


def collate_task(items):
    """将单任务样本整理为可变节点数 batch。"""
    width = max(item["nodes"].size(0) for item in items)
    result = {key: [] for key in ("nodes", "relations", "node_mask", "task_target", "task_valid")}
    for item in items:
        size = item["nodes"].size(0)
        result["nodes"].append(torch.nn.functional.pad(item["nodes"], (0, 0, 0, width - size)))
        result["relations"].append(torch.nn.functional.pad(item["relations"], (0, 0, 0, width - size, 0, width - size)))
        result["node_mask"].append(torch.nn.functional.pad(item["node_mask"], (0, width - size)))
        result["task_target"].append(item["task_target"])
        result["task_valid"].append(item["task_valid"])
    return {key: torch.stack(value) for key, value in result.items()}


def run_epoch(model, loader, device, optimizer=None):
    """运行单标量任务 epoch，返回 loss、MAE、RMSE 和容差 accuracy。"""
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "mae", "rmse", "accuracy")}
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(batch["nodes"], batch["relations"], batch["node_mask"], task_only=True)
            mask = batch["task_valid"].to(prediction.dtype)
            error = prediction - batch["task_target"]
            loss = (torch.nn.functional.smooth_l1_loss(prediction, batch["task_target"], reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
            totals["loss"] += loss.item()
            totals["mae"] += (error.abs() * mask).sum().item() / mask.sum().clamp_min(1.0).item()
            totals["rmse"] += (error.square() * mask).sum().div(mask.sum().clamp_min(1.0)).sqrt().item()
            totals["accuracy"] += ((error.abs() <= 1.0).to(mask.dtype) * mask).sum().item() / mask.sum().clamp_min(1.0).item()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    count = max(len(loader), 1)
    return {key: value / count for key, value in totals.items()}


def main():
    """执行一个明确的单任务训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_FIELDS), default="area")
    parser.add_argument("--source", default="timingpredict")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v1_c_design_task")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    liberty = ROOT / "dataset/raw/timingpredict_extracted/netlists/techlib/sky130_fd_sc_hd__ss_100C_1v60.lib"
    manifest = ROOT / "dataset/paired_graphs/manifest.json"
    pairs = pair_paths(manifest, args.source, args.limit, args.max_nodes)
    if len(pairs) < 2:
        raise ValueError("single-task training needs at least two eligible pairs")
    split = max(1, int(len(pairs) * 0.8))
    stats = fit_stats(pairs[:split], liberty, args.task, args.max_nodes)
    train = DesignTaskDataset(pairs[:split], liberty, args.task, args.max_nodes, stats)
    valid = DesignTaskDataset(pairs[split:], liberty, args.task, args.max_nodes, stats)
    train_loader = DataLoader(train, 1, shuffle=True, collate_fn=collate_task)
    valid_loader = DataLoader(valid, 1, collate_fn=collate_task)
    first = train[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V1CModel(first["nodes"].size(-1), 3, 16, 8, 32, layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({"task": args.task, "source": args.source, "seed": args.seed, "device": str(device), "stats": stats}, indent=2))
    best = float("inf")
    print(f"task={args.task} device={device} train={len(train)} valid={len(valid)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            valid_metrics = run_epoch(model, valid_loader, device)
        print(f"epoch={epoch}/{args.epochs} train={train_metrics['loss']:.4e} valid={valid_metrics['loss']:.4e} "
              f"val_mae={valid_metrics['mae']:.4f} val_rmse={valid_metrics['rmse']:.4f} "
              f"val_acc_1std={valid_metrics['accuracy']:.4f}", flush=True)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": valid_metrics, "stats": stats}, args.output / "best.pt")


if __name__ == "__main__":
    main()
