"""DeepGate2 K=4 三元关系动力学训练执行流。"""

import argparse
import random
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

from .deepgate_data import load_deepgate2_circuits
from .deepgate_k4_model import DeepGateK4Model


@torch.no_grad()
def evaluate_circuits(model, circuits, device):
    """逐图评估信号翻转概率的物理指标。"""
    model.eval()
    p_all, t_all = [], []
    r_list, r2_list = [], []

    for c in circuits:
        x = c["nodes"].to(device)
        ei = c["edge_index"].to(device)
        t = c["prob"].to(device)

        p = model(x, ei)
        p_all.append(p)
        t_all.append(t)

        r = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
        if not torch.isnan(torch.tensor(r)): r_list.append(r)

        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum().clamp_min(1e-6)
        r2_list.append((1.0 - ss_res / ss_tot).item())

    model.train()
    all_p = torch.cat(p_all)
    all_t = torch.cat(t_all)
    diff = (all_p - all_t).abs()
    p_std = all_p.std().item()
    t_std = all_t.std().item()

    return {
        "mae": diff.mean().item(),
        "r": sum(r_list) / len(r_list) if r_list else 0.0,
        "r2": sum(r2_list) / len(r2_list) if r2_list else 0.0,
        "pred_std": p_std, "true_std": t_std,
        "std_ratio": p_std / max(t_std, 1e-6),
        "total_nodes": all_p.numel()
    }


def main():
    """主训练执行入口。"""
    p = argparse.ArgumentParser(description="DeepGate2 K=4 Tri-Relation Dynamics")
    p.add_argument("--output", type=Path, default=Path("/data/yiyh/dac2027/runs/deepgate2_k4_tri_rel_300step"))
    p.add_argument("--max-nodes", type=int, default=512)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--dynamics-steps", type=int, default=8)
    p.add_argument("--k-neighbors", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=20261103)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    train_circuits = load_deepgate2_circuits(max_nodes=args.max_nodes, limit=args.limit, split="train", seed=args.seed)
    val_circuits = load_deepgate2_circuits(max_nodes=args.max_nodes, limit=args.limit, split="val", seed=args.seed)

    model = DeepGateK4Model(node_width=3, state_width=32, hidden_width=64, steps=args.dynamics_steps, k=args.k_neighbors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)

    print(f"🌟 deepgate2_k4_start train={len(train_circuits)} val={len(val_circuits)} k={args.k_neighbors} steps={args.steps}", flush=True)

    for step in range(1, args.steps + 1):
        c = random.choice(train_circuits)
        x = c["nodes"].to(device)
        ei = c["edge_index"].to(device)
        t = c["prob"].to(device)

        optimizer.zero_grad(set_to_none=True)
        p_pred = model(x, ei)
        loss = F.smooth_l1_loss(p_pred, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            val = evaluate_circuits(model, val_circuits, device)
            print(f"step={step:4d} loss={loss.item():.4e} val_mae={val['mae']:.4f} "
                  f"val_r={val['r']:.4f} val_R2={val['r2']:.4f} std_ratio={val['std_ratio']:.3f} "
                  f"(p_std={val['pred_std']:.3f}/t_std={val['true_std']:.3f})", flush=True)

    print(f"deepgate2_k4_done output={args.output}", flush=True)


if __name__ == "__main__":
    main()
