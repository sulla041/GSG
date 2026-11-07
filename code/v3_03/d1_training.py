"""V3.03 D1 时序预测端到端训练入口：支持 Log-Space 标准化与丰富物理诊断指标。"""

import argparse
import json
import random
import sys
from pathlib import Path
import torch

from .d1_data import load_d1_tensors, compute_d1_log_stats, make_d1_batch
from .d1_task import d1_loss, d1_metrics
from .model import V3_03Model


def arguments():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="V3.03 D1 Log-Space Timing Training")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--max-nodes", type=int, default=1024)
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dynamics-steps", type=int, default=8)
    p.add_argument("--state-width", type=int, default=32)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, graphs, device, batch_size, log_mean, log_std):
    """评估模型在验证集或当前子集上的多维物理诊断指标。"""
    model.eval()
    rows = []
    for start in range(0, len(graphs), batch_size):
        batch = graphs[start:start + batch_size]
        nodes, relations, targets, target_mask, node_mask = make_d1_batch(batch, device)
        pred = model(nodes, relations, node_mask, include_nodes=True)["nodes"]
        rows.append(d1_metrics(pred, targets, target_mask, log_mean, log_std))
    model.train()
    total_valid = sum(r["valid_nodes"] for r in rows)
    if total_valid == 0:
        return {"mae_ns": 0.0, "mape_percent": 0.0, "r_timing": 0.0, "r2_timing": 0.0,
                "pred_std": 0.0, "true_std": 0.0, "std_ratio": 0.0, "valid_nodes": 0}
    return {
        "mae_ns": sum(r["mae_ns"] * r["valid_nodes"] for r in rows) / total_valid,
        "mape_percent": sum(r["mape_percent"] * r["valid_nodes"] for r in rows) / total_valid,
        "r_timing": sum(r["r_timing"] * r["valid_nodes"] for r in rows) / total_valid,
        "r2_timing": sum(r["r2_timing"] * r["valid_nodes"] for r in rows) / total_valid,
        "pred_std": sum(r["pred_std"] * r["valid_nodes"] for r in rows) / total_valid,
        "true_std": sum(r["true_std"] * r["valid_nodes"] for r in rows) / total_valid,
        "std_ratio": sum(r["std_ratio"] * r["valid_nodes"] for r in rows) / total_valid,
        "valid_nodes": total_valid,
    }


def main():
    """主训练执行流。"""
    args = arguments()
    print(f"python={sys.executable} torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("V3.03 D1 training requires a visible CUDA device; CPU fallback is disabled")

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    limit = 4 if args.smoke else None
    graphs, data_meta = load_d1_tensors(max_nodes=args.max_nodes, split=args.split, seed=args.seed, limit=limit)
    if args.smoke and args.steps > 20:
        args.steps, args.eval_every = 10, 2

    device = torch.device("cuda")
    # 计算训练集的 Log 统计量
    log_mean, log_std = compute_d1_log_stats(graphs)
    log_mean, log_std = log_mean.to(device), log_std.to(device)
    print(f"D1 Log-Space Stats: mean={log_mean.tolist()}, std={log_std.tolist()}", flush=True)

    model = V3_03Model(
        node_width=data_meta["node_width"],
        relation_width=data_meta["relation_width"],
        state_width=args.state_width,
        hidden_width=args.hidden_width,
        steps=args.dynamics_steps,
        k=args.k_neighbors,
        node_output=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)

    config = {
        **vars(args), "data": data_meta, "task": "D1_timing_arrival_log_space",
        "log_mean": log_mean.tolist(), "log_std": log_std.tolist(),
        "device": str(device), "gpu": torch.cuda.get_device_name()
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    best_r = -float("inf")
    history = []
    print(f"v3_03_d1_log_start split={args.split} graphs={len(graphs)} k={args.k_neighbors} gpu={config['gpu']}", flush=True)

    for step in range(1, args.steps + 1):
        batch = random.choices(graphs, k=min(args.batch_size, len(graphs)))
        nodes, relations, targets, target_mask, node_mask = make_d1_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        norm_pred = model(nodes, relations, node_mask, include_nodes=True)["nodes"]
        batch_loss = d1_loss(norm_pred, targets, target_mask, log_mean, log_std)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            val = evaluate(model, graphs, device, args.batch_size, log_mean, log_std)
            row = {"step": step, "loss": batch_loss.item(), **val}
            history.append(row)
            if val["r_timing"] > best_r:
                best_r = val["r_timing"]
                torch.save({"model": model.state_dict(), "config": config, "step": step}, args.output / "best.pt")
            print(f"step={step:4d} loss={batch_loss.item():.4e} mae={val['mae_ns']:.4f}ns "
                  f"mape={val['mape_percent']:.2f}% r={val['r_timing']:.4f} R2={val['r2_timing']:.4f} "
                  f"std_ratio={val['std_ratio']:.3f} (p_std={val['pred_std']:.3f}/t_std={val['true_std']:.3f})", flush=True)

    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"v3_03_d1_done best_r={best_r:.4f} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
