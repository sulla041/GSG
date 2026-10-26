"""训练 V1-C 的单一关系级二元下游任务。"""

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

from dataset.relation_task import RELATION_TASKS, RelationTaskDataset, collate_relation
from train_v1_c_ppa import pair_paths
from v1_c.model import V1CModel


def run_epoch(model, loader, device, optimizer=None):
    """运行关系任务 epoch，报告 Cora 风格二元指标。"""
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "accuracy", "precision", "recall", "f1")}
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch["nodes"], batch["relations"], batch["node_mask"], relation_task_only=True)
            mask, target = batch["relation_task_mask"], batch["relation_task_target"]
            loss = torch.nn.functional.cross_entropy(logits[mask], target[mask])
            prediction = logits.argmax(-1)
            correct = (prediction == target) & mask
            positive = target == 1
            predicted_positive = prediction == 1
            true_positive = (positive & predicted_positive & mask).sum().float()
            precision = true_positive / (predicted_positive & mask).sum().clamp_min(1)
            recall = true_positive / (positive & mask).sum().clamp_min(1)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
            totals["loss"] += loss.item()
            totals["accuracy"] += correct[mask].float().mean().item()
            totals["precision"] += precision.item()
            totals["recall"] += recall.item()
            totals["f1"] += f1.item()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    count = max(len(loader), 1)
    return {key: value / count for key, value in totals.items()}


def main():
    """执行一个关系类型下游任务。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=RELATION_TASKS, default="cell_to_net")
    parser.add_argument("--source", default="circuitnet3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v1_c_relation_task")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    liberty = ROOT / "dataset/raw/timingpredict_extracted/netlists/techlib/sky130_fd_sc_hd__ss_100C_1v60.lib"
    print(f"preparing relation task={args.task} source={args.source}", flush=True)
    pairs = pair_paths(ROOT / "dataset/paired_graphs/manifest.json", args.source, args.limit, args.max_nodes)
    if len(pairs) < 2:
        raise ValueError("relation-task training needs at least two eligible pairs")
    split = max(1, int(len(pairs) * 0.8))
    train = RelationTaskDataset(pairs[:split], liberty, args.task, args.max_nodes)
    valid = RelationTaskDataset(pairs[split:], liberty, args.task, args.max_nodes)
    train_loader = DataLoader(train, 1, shuffle=True, collate_fn=collate_relation)
    valid_loader = DataLoader(valid, 1, collate_fn=collate_relation)
    first = train[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V1CModel(first["nodes"].size(-1), 3, 16, 8, 32, layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({"task": args.task, "source": args.source, "seed": args.seed, "device": str(device)}, indent=2))
    best = float("inf")
    print(f"relation_task={args.task} device={device} train={len(train)} valid={len(valid)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            metrics = run_epoch(model, valid_loader, device)
        print(f"epoch={epoch}/{args.epochs} val_loss={metrics['loss']:.4e} "
              f"val_acc={metrics['accuracy']:.4f} val_precision={metrics['precision']:.4f} "
              f"val_recall={metrics['recall']:.4f} val_f1={metrics['f1']:.4f}", flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics}, args.output / "best.pt")


if __name__ == "__main__":
    main()
