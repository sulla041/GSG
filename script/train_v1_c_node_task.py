"""训练 V1-C 的单一节点级 PPA 下游任务。"""

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

from dataset.node_task import NodeTaskDataset, collate_node_task, load_node_task
from dataset.ppa_supervised import NODE_LABELS
from dataset.ppa_supervised import _node_target
from train_v1_c_ppa import pair_paths
from v1_c.model import V1CModel


def fit_stats(pairs, liberty, task, max_nodes):
    """只用 train split 的有效节点标签拟合标准化参数。"""
    values = []
    index = NODE_LABELS.index(task)
    for number, pair in enumerate(pairs, 1):
        items = json.loads((pair / "target/fine/nodes.json").read_text())
        for item in items:
            value, valid = _node_target(item)
            if valid[index]:
                values.append(value[index])
        if number % 50 == 0 or number == len(pairs):
            print(f"stats_progress={number}/{len(pairs)}", flush=True)
    if not values:
        raise ValueError(f"no valid node labels for task={task}")
    tensor = torch.tensor(values)
    return {"mean": tensor.mean().item(), "std": tensor.std(unbiased=False).clamp_min(1e-12).item()}


def run_epoch(model, loader, device, optimizer=None):
    """运行节点级 masked regression epoch。"""
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "mae", "rmse", "accuracy")}
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(batch["nodes"], batch["relations"], batch["node_mask"], node_task_only=True)
            mask = batch["node_task_valid"].to(prediction.dtype)
            error = prediction - batch["node_task_target"]
            loss = (torch.nn.functional.smooth_l1_loss(prediction, batch["node_task_target"], reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
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
    """执行一个明确的节点级单任务训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=NODE_LABELS, default="area")
    parser.add_argument("--source", default="circuitnet3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v1_c_node_task")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    liberty = ROOT / "dataset/raw/timingpredict_extracted/netlists/techlib/sky130_fd_sc_hd__ss_100C_1v60.lib"
    print(f"preparing node task={args.task} source={args.source}", flush=True)
    pairs = pair_paths(ROOT / "dataset/paired_graphs/manifest.json", args.source, args.limit, args.max_nodes)
    if len(pairs) < 2:
        raise ValueError("node-task training needs at least two eligible pairs")
    print(f"eligible_pairs={len(pairs)}", flush=True)
    split = max(1, int(len(pairs) * 0.8))
    stats = fit_stats(pairs[:split], liberty, args.task, args.max_nodes)
    train = NodeTaskDataset(pairs[:split], liberty, args.task, args.max_nodes, stats)
    valid = NodeTaskDataset(pairs[split:], liberty, args.task, args.max_nodes, stats)
    train_loader = DataLoader(train, 1, shuffle=True, collate_fn=collate_node_task)
    valid_loader = DataLoader(valid, 1, collate_fn=collate_node_task)
    first = train[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V1CModel(first["nodes"].size(-1), 3, 16, 8, 32, layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({"task": args.task, "source": args.source, "seed": args.seed, "device": str(device), "stats": stats}, indent=2))
    best = float("inf")
    print(f"node_task={args.task} device={device} train={len(train)} valid={len(valid)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            metrics = run_epoch(model, valid_loader, device)
        print(f"epoch={epoch}/{args.epochs} val_loss={metrics['loss']:.4e} "
              f"val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f} "
              f"val_acc_1std={metrics['accuracy']:.4f}", flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics, "stats": stats}, args.output / "best.pt")


if __name__ == "__main__":
    main()
