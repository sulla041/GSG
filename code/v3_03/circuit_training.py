"""V3.03 电路端到端训练入口：支持 K-有界动力学、极速加载与 MAPE 监督。"""

import argparse
import json
import random
import sys
from pathlib import Path
import torch

from .circuit_data import TENSOR_DIR, load_circuit_tensors, make_batch
from .circuit_task import loss, metrics
from .model import V3_03Model


def arguments():
    """解析命令行超参数配置。"""
    p = argparse.ArgumentParser(description="V3.03 K-Bounded Circuit Placement Dynamics Training")
    p.add_argument("--tensor-dir", type=Path, default=TENSOR_DIR)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--max-nodes", type=int, default=1024)
    p.add_argument("--k-neighbors", type=int, default=8, help="近邻采样数 (意涵决断: k=8 聚焦因果真边，抑制随机噪声稀释)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dynamics-steps", type=int, default=8)
    p.add_argument("--state-width", type=int, default=32)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--mape-weight", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, graphs, device, batch_size):
    """评估模型在验证集或当前子集上的纯 X 物理绝对误差、宽度相对比率与 X 相关性。"""
    model.eval()
    rows = []
    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start:start + batch_size]
        nodes, relations, targets, target_mask, node_mask, die_sizes = make_batch(batch_graphs, device)
        pred = model(nodes, relations, node_mask, die_size=die_sizes, include_nodes=True)["nodes"]
        rows.append(metrics(pred, targets, target_mask, die_size=die_sizes))
    model.train()
    total_valid = sum(r["valid_nodes"] for r in rows)
    if total_valid == 0:
        return {"x_mae": 0.0, "x_die_rel_percent": 0.0, "r_x": 0.0, "diag_x_mape_percent": 0.0, "valid_nodes": 0}
    return {
        "x_mae": sum(r["x_mae"] * r["valid_nodes"] for r in rows) / total_valid,
        "x_die_rel_percent": sum(r["x_die_rel_percent"] * r["valid_nodes"] for r in rows) / total_valid,
        "r_x": sum(r["r_x"] * r["valid_nodes"] for r in rows) / total_valid,
        "diag_x_mape_percent": sum(r["diag_x_mape_percent"] * r["valid_nodes"] for r in rows) / total_valid,
        "valid_nodes": total_valid,
    }


def main():
    """主训练执行流。"""
    args = arguments()
    print(f"python={sys.executable} torch={torch.__version__} "
          f"torch_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("V3.03 circuit training requires a visible CUDA device; CPU fallback is disabled")

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    limit = 4 if args.smoke else None
    graphs, data_meta = load_circuit_tensors(args.tensor_dir, args.max_nodes, args.split, args.seed, limit=limit)
    if args.smoke and args.steps == 1000:
        args.steps = 10
        args.eval_every = 2

    device = torch.device("cuda")
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
        **vars(args), "data": data_meta, "task": "P0_placement_xy",
        "device": str(device), "gpu": torch.cuda.get_device_name(),
        "dynamics": "v3_03_k_bounded_sum", "schema_version": "v3_03_circuit_p0_v1"
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    best_metric = float("inf")
    history = []
    print(f"v3_03_circuit_start split={args.split} graphs={len(graphs)} "
          f"k={args.k_neighbors} scale_weight={args.mape_weight} gpu={config['gpu']}", flush=True)

    for step in range(1, args.steps + 1):
        batch = random.choices(graphs, k=min(args.batch_size, len(graphs)))
        nodes, relations, targets, target_mask, node_mask, die_sizes = make_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(nodes, relations, node_mask, die_size=die_sizes, include_nodes=True)["nodes"]
        batch_loss = loss(pred, targets, target_mask, die_size=die_sizes, scale_weight=args.mape_weight)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            val = evaluate(model, graphs, device, args.batch_size)
            row = {"step": step, "loss": batch_loss.item(), **val}
            history.append(row)
            if val["x_die_rel_percent"] < best_metric:
                best_metric = val["x_die_rel_percent"]
                torch.save({"model": model.state_dict(), "config": config, "step": step},
                           args.output / "best_checkpoint.pt")
            (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(f"step={step:4d} | loss={batch_loss.item():.6f} | val_x_mae={val['x_mae']:.6f} "
                  f"| x_die_rel={val['x_die_rel_percent']:.2f}% | r(x)={val['r_x']:+.4f} "
                  f"| diag_x_mape={val['diag_x_mape_percent']:.2f}% | valid_nodes={val['valid_nodes']}", flush=True)

    print(f"v3_03_circuit_complete output={args.output} best_x_die_rel={best_metric:.2f}%", flush=True)
