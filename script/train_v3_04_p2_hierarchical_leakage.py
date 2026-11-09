"""P2 任务：基于意涵“32-边平衡分层电表网络”的全图总静态漏电功耗 (Total Leakage Power) 训练流水线。
- 架构特性：从全图所有 Cell 汇聚到 L1 电表 (<=32) -> Root 电表 (<=32)，入度严格 O(1)。
- 物理机制：纯静态物理漏电求和规约，彻底排除动态波形翻转率干扰，具备全局密集梯度。
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
import random
import re
import torch
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor

from code.v3_04.hierarchical_meter_leakage_dynamics import HierarchicalMeterLeakageDynamicsModel
from script.train_v3_04_p2_hierarchical_meter import inject_hierarchical_meter_network


def load_p2_leakage_dataset(d2_dir: Path, label_dir: Path, split="train", limit=None, device="cpu"):
    d2_files = sorted(list(d2_dir.glob("*.pt")), key=lambda x: x.stem)
    c1, c2 = int(len(d2_files) * 0.8), int(len(d2_files) * 0.9)
    ranges = {"train": (0, c1), "val": (c1, c2), "test": (c2, len(d2_files))}
    cur_d2 = d2_files[ranges[split][0]:ranges[split][1]][:limit] if limit else d2_files[ranges[split][0]:ranges[split][1]]

    def load_one(f_path):
        design = f_path.stem
        rpt_file = label_dir / design / "power.rpt"
        if not rpt_file.exists(): return None
        text = rpt_file.read_text()
        m = re.search(r'Total\s+([\d\.e\+-]+)\s+([\d\.e\+-]+)\s+([\d\.e\+-]+)\s+([\d\.e\+-]+)', text)
        if not m: return None
        leak_val = float(m.group(3)) # Total Leakage Power (Watts)
        if leak_val <= 0: return None

        d = torch.load(f_path, map_location="cpu", weights_only=False)
        d["leak_target"] = float(leak_val)
        res = inject_hierarchical_meter_network(d, branch_k=32, device=device)
        res["leak_target"] = float(leak_val)
        return res

    print(f"🚀 32 线程并发载入全图静态漏电样本 [{split}]...", flush=True)
    with ThreadPoolExecutor(max_workers=32) as pool:
        graphs = [g for g in pool.map(load_one, cur_d2) if g is not None]
    return graphs


@torch.no_grad()
def evaluate_leakage(model, graphs, max_scale):
    model.eval()
    p_all, t_all = [], []
    for g in graphs:
        pred_norm = model(g, g["root_meter_idx"])
        pred_w = pred_norm * max_scale
        p_all.append(pred_w)
        t_all.append(torch.tensor(g["leak_target"], device=pred_norm.device))
    model.train()

    preds, targets = torch.stack(p_all), torch.stack(t_all)
    # 转为微瓦 (uW) 评估
    preds_uw, targets_uw = preds * 1e6, targets * 1e6
    mae_uw = (preds_uw - targets_uw).abs().mean().item()
    mape = ((preds - targets).abs() / targets.clamp_min(1e-12)).mean().item() * 100.0
    r = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().clamp_min(1e-16).item()
    return {"mae_uw": mae_uw, "mape": mape, "r": r, "r2": 1.0 - ss_res / ss_tot}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d2_dir = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_04_d2_causal_rc_1500_1000_v1/D2")
    label_dir = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_expanded_ppa_labels_1500_v1")

    train_graphs = load_p2_leakage_dataset(d2_dir, label_dir, split="train", device=device)
    val_graphs = load_p2_leakage_dataset(d2_dir, label_dir, split="val", device=device)

    leaks = torch.tensor([g["leak_target"] for g in train_graphs], device=device)
    max_scale = leaks.max()
    print(f"🌟 静态漏电分层电表就绪: 训练集={len(train_graphs)}, 验证集={len(val_graphs)}, 均值={leaks.mean().item()*1e6:.2f}uW, 最大={max_scale.item()*1e6:.2f}uW")

    model = HierarchicalMeterLeakageDynamicsModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        g = random.choice(train_graphs)
        pred_linear = model(g, g["root_meter_idx"])
        loss = F.smooth_l1_loss(pred_linear, torch.tensor(g["leak_target"], device=device) / max_scale, beta=0.1)
        (loss / 4).backward()
        if step % 4 == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

        if step % 100 == 0 or step == args.steps:
            m = evaluate_leakage(model, val_graphs, max_scale)
            print(f"step={step:4d} loss={loss.item():.4e} mae={m['mae_uw']:.3f}uW mape={m['mape']:.2f}% r={m['r']:.4f} R2={m['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
