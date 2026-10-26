"""在统一 paired graphs 上训练 V1-C 的 PPA 监督头。"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from dataset.ppa_supervised import PPAPairDataset, collate_ppa, load_ppa_pair
from v1_c.model import V1CModel
from v1_c.ppa_training import run_ppa_epoch


def _node_count(pair: Path) -> int:
    """读取 netlist 规模，避免在稠密张量化阶段才发现超限。"""
    data = json.loads((pair / "input/netlist.json").read_text())
    module = next(iter(data["modules"].values()))
    return len(module.get("cells", {})) + len(module.get("netnames", {}))


def pair_paths(manifest: Path, source: str, limit: int, max_nodes: int) -> list[Path]:
    """从正式 manifest 选择来源独立的 pair 目录。"""
    data = json.loads(manifest.read_text())
    items = [x for x in data["pairs"] if not source or x["source"] == source]
    items.sort(key=lambda x: hashlib.sha1(
        f'{x["source"]}:{x["source_id"]}'.encode()).hexdigest())
    paths = [manifest.parent / f'{x["source"]}__{x["source_id"]}' for x in items]
    paths = [path for path in paths if _node_count(path) <= max_nodes]
    return paths[:limit] if limit else paths


def fit_stats(pairs: list[Path], liberty: Path, max_nodes: int) -> dict:
    """仅使用训练 pair 的有效标签拟合逐通道标准化参数。"""
    total = {"node": torch.zeros(10), "relation": torch.zeros(3)}
    square = {"node": torch.zeros(10), "relation": torch.zeros(3)}
    count = {"node": torch.zeros(10), "relation": torch.zeros(3)}
    for pair in pairs:
        sample = load_ppa_pair(pair, liberty, max_nodes)
        for kind, values, valid in (("node", sample["node_target"], sample["node_valid"]),
                                    ("relation", sample["relation_target"], sample["relation_valid"])):
            mask = valid.to(values.dtype)
            dims = (0, 1) if kind == "node" else (0, 1, 2)
            total[kind] += (values * mask).sum(dims)
            square[kind] += (values.square() * mask).sum(dims)
            count[kind] += mask.sum(dims)
    result = {}
    for kind in ("node", "relation"):
        mean = total[kind] / count[kind].clamp_min(1.0)
        variance = (square[kind] / count[kind].clamp_min(1.0) - mean.square()).clamp_min(0.0)
        result[f"{kind}_mean"] = mean.tolist()
        result[f"{kind}_std"] = variance.sqrt().clamp_min(1e-12).tolist()
    return result


def main() -> None:
    """执行可复现的 PPA 训练/验证并保存配置与 checkpoint。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/paired_graphs/manifest.json")
    parser.add_argument("--liberty", type=Path, default=ROOT / "dataset/raw/timingpredict_extracted/netlists/techlib/sky130_fd_sc_hd__ss_100C_1v60.lib")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v1_c_ppa")
    parser.add_argument("--source", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = pair_paths(args.manifest, args.source, args.limit, args.max_nodes)
    if len(pairs) < 2:
        raise ValueError("PPA training needs at least two pair directories")
    split = max(1, int(len(pairs) * 0.8))
    train_pairs, valid_pairs = pairs[:split], pairs[split:]
    stats = fit_stats(train_pairs, args.liberty, args.max_nodes)
    train_set = PPAPairDataset(train_pairs, args.liberty, args.max_nodes, stats)
    valid_set = PPAPairDataset(valid_pairs, args.liberty, args.max_nodes, stats)
    first = train_set[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V1CModel(first["nodes"].size(-1), 3, 16, 8, 32, layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate_ppa)
    valid_loader = DataLoader(valid_set, 1, shuffle=False, collate_fn=collate_ppa)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) for key, value in vars(args).items()}
    config.update({"device": str(device), "train_pairs": len(train_set),
                   "valid_pairs": len(valid_set), "node_labels": 10,
                   "relation_types": 4, "relation_labels": 3})
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2))
    best = float("inf")
    critical_threshold = -stats["relation_mean"][2] / stats["relation_std"][2]
    print(f"device={device} train={len(train_set)} valid={len(valid_set)} seed={args.seed}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_ppa_epoch(model, train_loader, device, optimizer,
                                       critical_threshold=critical_threshold)
        with torch.no_grad():
            valid_metrics = run_ppa_epoch(model, valid_loader, device,
                                           critical_threshold=critical_threshold)
        print(f"epoch={epoch}/{args.epochs} train={train_metrics['loss']:.4e} "
              f"valid={valid_metrics['loss']:.4e} "
              f"node={train_metrics['node_loss']:.4e} "
              f"class={train_metrics['relation_class_loss']:.4e} "
              f"timing={train_metrics['timing_loss']:.4e} "
              f"val_rel_acc={valid_metrics['relation_acc']:.4f} "
              f"val_dense_rel_acc={valid_metrics['dense_relation_acc']:.4f} "
              f"val_edge_acc={valid_metrics['non_null_relation_acc']:.4f} "
              f"val_edge_f1={valid_metrics['edge_f1']:.4f} "
              f"val_edge_recall={valid_metrics['edge_recall']:.4f} "
              f"val_critical_acc={valid_metrics['critical_acc']:.4f} "
              f"val_node_1std={valid_metrics['node_within_1std']:.4f} "
              f"val_timing_1std={valid_metrics['timing_within_1std']:.4f}", flush=True)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "metrics": valid_metrics, "stats": stats}, args.output / "best.pt")


if __name__ == "__main__":
    main()
