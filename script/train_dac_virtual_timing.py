"""训练 DAC2027 Virtual Arc Delay/Slack/Criticality 单任务。"""

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

from dataset.dac_virtual_timing_dataset import (DACVirtualTimingDataset,
                                                collate_virtual_timing)
from v1_c.model import V1CModel
from v1_c.virtual_timing import run_virtual_epoch, target_stats


TASKS = {"delay": (0, True), "slack": (1, False), "criticality": (2, False)}


def loader(dataset, shuffle, workers):
    """建立 full-node dynamic padding loader。"""
    return DataLoader(dataset, batch_size=1, shuffle=shuffle,
                      collate_fn=collate_virtual_timing, num_workers=workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0,
                      prefetch_factor=2 if workers > 0 else None)


def main() -> None:
    """执行 virtual timing 单任务 train/valid/test。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="delay")
    parser.add_argument("--root", type=Path, default=Path("/home/yiyh/graphmath/dac2027/processed/verilog_rtl_corpus"))
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/dac2027_virtual_timing_v1")
    parser.add_argument("--split-path", type=Path, default=ROOT / "dataset/virtual_timing_split.json")
    parser.add_argument("--feature-profile", choices=("base", "native_process", "pin_electrical"), default="native_process")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=2048)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    channel, log_target = TASKS[args.task]
    datasets = [DACVirtualTimingDataset(args.root, args.labels, split, args.feature_profile, args.max_nodes,
                                         args.split_path)
                for split in ("train", "val", "test")]
    limits = (args.limit_train, args.limit_valid, args.limit_test)
    for dataset, limit in zip(datasets, limits):
        if limit:
            dataset.ids = dataset.ids[:limit]
    train, valid, test = datasets
    mean, std = target_stats(train, channel, log_target) if channel != 2 else (torch.tensor(0.), torch.tensor(1.))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V1CModel(train[0]["nodes"].size(-1), train[0]["relations"].size(-1), 32, 16, 64, layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"device": str(device), "channel": channel, "log_target": log_target,
                            "train": len(train), "valid": len(valid), "test": len(test),
                            "mean": mean.item(), "std": std.item(),
                            "loss": "masked_Huber" if channel != 2 else "masked_BCE",
                            "target_transform": "log1p_delay_ns" if log_target else "standardized_raw"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    train_loader, valid_loader, test_loader = (loader(train, True, args.num_workers),
                                               loader(valid, False, args.num_workers),
                                               loader(test, False, args.num_workers))
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_virtual_epoch(model, train_loader, device, channel, mean.to(device), std.to(device), log_target, optimizer)
        with torch.inference_mode():
            valid_stats = run_virtual_epoch(model, valid_loader, device, channel, mean.to(device), std.to(device), log_target)
        print(f"epoch={epoch} train={train_stats} valid={valid_stats}", flush=True)
        if valid_stats["loss"] < best:
            best = valid_stats["loss"]
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "task": args.task, "channel": channel, "epoch": epoch}, args.output / "best.pt")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.inference_mode():
        result = run_virtual_epoch(model, test_loader, device, channel,
                                   checkpoint["mean"].to(device), checkpoint["std"].to(device), log_target)
    (args.output / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"test={result}", flush=True)


if __name__ == "__main__":
    main()
