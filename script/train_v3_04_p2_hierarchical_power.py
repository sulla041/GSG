"""P2 任务：基于意涵“32-边平衡分层电表网络”的整图总功耗 (Total Power) 训练流水线。
- 架构特性：Cell -> L1 电表 (每块 <= 32 节点) -> Root 电表 (<= 32 节点)，入度严格 O(1)。
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
import random
import torch
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor

from code.v3_04.hierarchical_meter_power_dynamics import HierarchicalMeterPowerDynamicsModel
from script.train_v3_04_p2_hierarchical_meter import inject_hierarchical_meter_network

def load_p2_power_dataset(d2_dir: Path, p2_dir: Path, split="train", limit=None, device="cpu"):
    p2_files = sorted(list(p2_dir.glob("*.pt")), key=lambda x: x.stem)
    c1, c2 = int(len(p2_files) * 0.8), int(len(p2_files) * 0.9)
    ranges = {"train": (0, c1), "val": (c1, c2), "test": (c2, len(p2_files))}
    cur_p2 = p2_files[ranges[split][0]:ranges[split][1]][:limit] if limit else p2_files[ranges[split][0]:ranges[split][1]]

    def load_one(p2_path):
        d2_path = d2_dir / f"{p2_path.stem}.pt"
        if not d2_path.exists(): return None
        p2_data = torch.load(p2_path, map_location="cpu", weights_only=False)
        gt, mask = p2_data.get("graph_targets"), p2_data.get("graph_target_mask")
        if gt is None or mask is None or not mask[3]: return None # mask[3] is power
        d2_data = torch.load(d2_path, map_location="cpu", weights_only=False)
        d2_data["power_target"] = float(gt[3].item()) # 原始单位 W
        return inject_hierarchical_meter_network(d2_data, branch_k=32, device=device)

    print(f"🚀 32 线程并发载入 P2 32-边分层电表网络功耗样本 [{split}]...", flush=True)
    with ThreadPoolExecutor(max_workers=32) as pool:
        graphs = [g for g in pool.map(load_one, cur_p2) if g is not None]
    return graphs

@torch.no_grad()
def evaluate_power(model, graphs, max_power):
    model.eval()
    p_all, t_all = [], []
    for g in graphs:
        pred_norm = model(g, g["root_meter_idx"])
        pred_w = pred_norm * max_power
        p_all.append(pred_w)
        t_all.append(torch.tensor(g["power_target"], device=pred_norm.device))
    model.train()
    
    preds, targets = torch.stack(p_all), torch.stack(t_all)
    mae = (preds - targets).abs().mean().item()
    mape = ((preds - targets).abs() / targets.clamp_min(1e-9)).mean().item() * 100.0
    r = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().clamp_min(1e-12).item()
    return {"mae": mae, "mape": mape, "r": r, "r2": 1.0 - ss_res / ss_tot}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d2_dir = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_04_d2_causal_rc_1500_1000_v1/D2")
    p2_dir = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_03_compact_tensors_parasitic_1500_4096_v1/P2")

    train_graphs = load_p2_power_dataset(d2_dir, p2_dir, split="train", device=device)
    val_graphs = load_p2_power_dataset(d2_dir, p2_dir, split="val", device=device)

    powers = torch.tensor([g["power_target"] for g in train_graphs], device=device)
    max_power = powers.max()
    print(f"🌟 32-边分层电表网络就绪: 训练集={len(train_graphs)}, 验证集={len(val_graphs)}, max_power={max_power.item():.2e}W")

    model = HierarchicalMeterPowerDynamicsModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        g = random.choice(train_graphs)
        pred_linear = model(g, g["root_meter_idx"])
        loss = F.smooth_l1_loss(pred_linear, torch.tensor(g["power_target"], device=device) / max_power, beta=0.1)
        (loss / 4).backward()
        if step % 4 == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

        if step % 100 == 0 or step == args.steps:
            m = evaluate_power(model, val_graphs, max_power)
            print(f"step={step:4d} loss={loss.item():.4e} mae={m['mae']:.2e}W mape={m['mape']:.2f}% r={m['r']:.4f} R2={m['r2']:.4f}", flush=True)

if __name__ == "__main__":
    main()
