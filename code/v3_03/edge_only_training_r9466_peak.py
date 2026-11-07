"""V3.03 纯真自发散开动力学训练执行流：纯净密集单步优化版。"""

import argparse
import random
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

from .d1_data import load_d1_tensors, compute_d1_log_stats
from .edge_only_dynamics import ExactEdgeModel

LOG_EPS = 1e-12


def edge_loss(norm_pred, target, target_mask, log_mean, log_std, mape_weight=0.1):
    # 1. 对数标准化空间绝对平滑损失
    norm_t = (torch.log(target.clamp_min(LOG_EPS)) - log_mean) / log_std
    l1_loss = F.smooth_l1_loss(norm_pred[target_mask], norm_t[target_mask])

    # 2. 意涵第一性原理：物理纳秒空间的平滑相对误差 (Smooth MAPE) 监督
    # 消除纯绝对误差对浅层节点的轻视，赋予全尺度平等的因果保序权重
    raw_p = torch.exp(norm_pred * log_std + log_mean)
    p_val, t_val = raw_p[target_mask], target[target_mask]
    rel_ratio = p_val / t_val.clamp_min(1e-4)
    # 当 p == t 时，rel_ratio == 1.0，误差为 0；使用 Smooth L1 杜绝极端分母数值震荡
    mape_loss = F.smooth_l1_loss(rel_ratio, torch.ones_like(rel_ratio), beta=0.1)

    return l1_loss + mape_weight * mape_loss


@torch.no_grad()
def evaluate_exact_edges(model, graphs, device, log_mean, log_std):
    model.eval()
    p_all, t_all = [], []
    r_list, r2_list = [], []

    for g in graphs:
        x, ei, ea = g["input_nodes"].to(device), g["input_edge_index"].to(device), g["input_edge_attr"].to(device)
        m, t = g["node_target_mask"][:, 10:12].to(device), g["node_targets"][:, 10:12].to(device)

        norm_p = model(x, ei, ea, source_anchor_mask=g.get("source_anchor_mask", None).to(device) if g.get("source_anchor_mask") is not None else None)
        raw_p = torch.exp(norm_p * log_std + log_mean)

        m0 = m[:, 0]
        if m0.sum() >= 4:
            pb, tb = raw_p[m0, 0], t[m0, 0]
            p_all.append(pb); t_all.append(tb)
            r = torch.corrcoef(torch.stack([pb, tb]))[0, 1].item()
            if not torch.isnan(torch.tensor(r)): r_list.append(r)
            ss_res = ((tb - pb) ** 2).sum()
            ss_tot = ((tb - tb.mean()) ** 2).sum().clamp_min(1e-6)
            r2_list.append((1.0 - ss_res / ss_tot).item())

    model.train()
    if not p_all:
        return {"mae_ns": 0.0, "mape_percent": 0.0, "r_timing": 0.0, "r2_macro": 0.0, "r2_pooled": 0.0, "pred_std": 0.0, "true_std": 0.0, "std_ratio": 0.0}

    all_p, all_t = torch.cat(p_all), torch.cat(t_all)
    diff = (all_p - all_t).abs()
    p_std, t_std = all_p.std().item(), all_t.std().item()
    pooled_ss_res = ((all_t - all_p) ** 2).sum()
    pooled_ss_tot = ((all_t - all_t.mean()) ** 2).sum().clamp_min(1e-6)
    r2_pooled = (1.0 - pooled_ss_res / pooled_ss_tot).item()

    return {
        "mae_ns": diff.mean().item(),
        "mape_percent": ((diff / all_t.clamp_min(1e-4)) * 100.0).mean().item(),
        "r_timing": sum(r_list) / len(r_list) if r_list else 0.0,
        "r2_macro": sum(r2_list) / len(r2_list) if r2_list else 0.0,
        "r2_pooled": r2_pooled,
        "pred_std": p_std, "true_std": t_std,
        "std_ratio": p_std / max(t_std, 1e-6)
    }


def main():
    p = argparse.ArgumentParser(description="Exact Directed Edge Only Dynamics")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-nodes", type=int, default=2048)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--dynamics-steps", type=int, default=16)
    p.add_argument("--cost-k", type=float, default=1.0)
    p.add_argument("--min-k", type=int, default=2, help="节点最小受力源保底数 (仅真边不足时精准补充无连接边)")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--max-target-cap", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    limit = 4 if args.smoke else None
    graphs, meta = load_d1_tensors(max_nodes=args.max_nodes, split="train", seed=args.seed, limit=limit)
    if args.max_target_cap is not None:
        raw_count = len(graphs)
        graphs = [g for g in graphs if g["node_targets"][g["node_target_mask"][:, 10], 10].max().item() < args.max_target_cap]
        print(f"🧹 样本净化: 剔除 max_arrival >= {args.max_target_cap}ns 的设计, 样本数 {raw_count} -> {len(graphs)}", flush=True)

    if args.smoke and args.steps > 20: args.steps, args.eval_every = 10, 2

    log_mean, log_std = compute_d1_log_stats(graphs)
    log_mean, log_std = log_mean.to(device), log_std.to(device)

    model = ExactEdgeModel(
        node_width=65, relation_width=17, state_width=32, hidden_width=64,
        steps=args.dynamics_steps, cost_k=args.cost_k, node_output=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)

    print(f"🌟 exact_edge_only_start device={device} graphs={len(graphs)} steps={args.steps} dynamics_steps={args.dynamics_steps}", flush=True)
    for step in range(1, args.steps + 1):
        g = random.choice(graphs)
        x, ei, ea = g["input_nodes"].to(device), g["input_edge_index"].to(device), g["input_edge_attr"].to(device)
        m, t = g["node_target_mask"][:, 10:12].to(device), g["node_targets"][:, 10:12].to(device)

        optimizer.zero_grad(set_to_none=True)
        norm_p = model(x, ei, ea, source_anchor_mask=g.get("source_anchor_mask", None).to(device) if g.get("source_anchor_mask") is not None else None)
        loss = edge_loss(norm_p, t, m, log_mean, log_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            val = evaluate_exact_edges(model, graphs, device, log_mean, log_std)
            print(f"step={step:4d} loss={loss.item():.4e} mae={val['mae_ns']:.4f}ns "
                  f"mape={val['mape_percent']:.2f}% r={val['r_timing']:.4f} "
                  f"R2_pool={val['r2_pooled']:.4f} (R2_macro={val['r2_macro']:.4f}) "
                  f"std_ratio={val['std_ratio']:.3f} (p_std={val['pred_std']:.3f}/t_std={val['true_std']:.3f})", flush=True)

    print(f"exact_edge_only_done output={args.output}", flush=True)


if __name__ == "__main__":
    main()
