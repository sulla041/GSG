"""CC-Tensor v3 专用训练与评估流水线：纯显存常驻、零计算浪费。"""

import argparse
import random
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from code.v3_03.compact_d1_data import load_compact_d1_tensors, compute_compact_d1_log_stats
from code.v3_03.compact_edge_dynamics import CompactExactEdgeModel

LOG_EPS = 1e-12


def compact_loss(norm_pred, target, target_mask, log_mean, log_std, mape_weight=0.1):
    norm_t = (torch.log(target.clamp_min(LOG_EPS)) - log_mean) / log_std
    l1 = F.smooth_l1_loss(norm_pred[target_mask], norm_t[target_mask])
    raw_p = torch.exp(norm_pred * log_std + log_mean)
    p_val, t_val = raw_p[target_mask], target[target_mask]
    ratio = p_val / t_val.clamp_min(1e-4)
    mape = F.smooth_l1_loss(ratio, torch.ones_like(ratio), beta=0.1)
    return l1 + mape_weight * mape


@torch.no_grad()
def evaluate_compact(model, graphs, log_mean, log_std):
    model.eval()
    p_all, t_all, r_list, r2_list = [], [], [], []
    for g in graphs:
        m, t = g["target_mask"], g["arrival_time"]
        norm_p = model(g)
        raw_p = torch.exp(norm_p * log_std + log_mean)
        if m.sum() >= 4:
            pb, tb = raw_p[m], t[m]
            p_all.append(pb); t_all.append(tb)
            r = torch.corrcoef(torch.stack([pb, tb]))[0, 1].item()
            if not torch.isnan(torch.tensor(r)): r_list.append(r)
            ss_res = ((tb - pb) ** 2).sum()
            ss_tot = ((tb - tb.mean()) ** 2).sum().clamp_min(1e-6)
            r2_list.append((1.0 - ss_res / ss_tot).item())
    model.train()
    if not p_all: return {}
    ap, at = torch.cat(p_all), torch.cat(t_all)
    diff = (ap - at).abs()
    p_std, t_std = ap.std().item(), at.std().item()
    p_res = ((at - ap) ** 2).sum()
    p_tot = ((at - at.mean()) ** 2).sum().clamp_min(1e-6)
    return {
        "mae_ns": diff.mean().item(), "mape": ((diff / at.clamp_min(1e-4)) * 100.0).mean().item(),
        "r": sum(r_list) / len(r_list) if r_list else 0.0,
        "r2_macro": sum(r2_list) / len(r2_list) if r2_list else 0.0,
        "r2_pool": (1.0 - p_res / p_tot).item(),
        "std_ratio": p_std / max(t_std, 1e-6), "p_std": p_std, "t_std": t_std
    }


def main():
    p = argparse.ArgumentParser(description="CC-Tensor v3 Training")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dynamics-steps", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--max-target-cap", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit = 4 if args.smoke else None
    graphs = load_compact_d1_tensors(split="train", seed=args.seed, limit=limit, device=device)
    raw_cnt = len(graphs)
    graphs = [g for g in graphs if g["target_mask"].any()]
    if args.max_target_cap is not None:
        graphs = [g for g in graphs if g["arrival_time"][g["target_mask"]].max().item() < args.max_target_cap]
    print(f"🧹 样本净化: 有效时序监督且 max_arrival < {args.max_target_cap}ns, 样本数 {raw_cnt} -> {len(graphs)}", flush=True)

    if args.smoke and args.steps > 20: args.steps, args.eval_every = 10, 2
    log_mean, log_std = compute_compact_d1_log_stats(graphs)
    log_mean, log_std = log_mean.to(device), log_std.to(device)

    model = CompactExactEdgeModel(steps=args.dynamics_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.learning_rate * 0.05)

    print(f"🌟 compact_exact_edge_start device={device} graphs={len(graphs)} steps={args.steps} T={args.dynamics_steps}", flush=True)
    for step in range(1, args.steps + 1):
        g = random.choice(graphs)
        opt.zero_grad(set_to_none=True)
        norm_p = model(g)
        loss = compact_loss(norm_p, g["arrival_time"], g["target_mask"], log_mean, log_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            val = evaluate_compact(model, graphs, log_mean, log_std)
            print(f"step={step:4d} loss={loss.item():.4e} mae={val['mae_ns']:.4f}ns "
                  f"mape={val['mape']:.2f}% r={val['r']:.4f} R2_pool={val['r2_pool']:.4f} "
                  f"(R2_macro={val['r2_macro']:.4f}) std_ratio={val['std_ratio']:.3f} "
                  f"(p_std={val['p_std']:.3f}/t_std={val['t_std']:.3f})", flush=True)


if __name__ == "__main__":
    main()
