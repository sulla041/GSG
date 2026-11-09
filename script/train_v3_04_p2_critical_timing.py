"""P2 任务：基于意涵“32-边平衡分层电表网络”的全图关键路径最大到达时间 (Critical Path Delay) 训练流水线。
- 架构特性：从全图所有时序端点 (Target Pins) 汇聚到 L1 电表 (<=32) -> Root 电表 (<=32)，入度严格 O(1)。
- 物理机制：Softmax 竞争自动等价求解最长路径 Max，Root 电表直接读出全图时序最高主频极限。
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

from code.v3_04.hierarchical_meter_critical_timing import HierarchicalMeterCriticalTimingDynamicsModel


def inject_timing_hierarchical_meter(d, branch_k=32, device="cpu"):
    """为全图时序终点构建 32-边平衡金字塔电表网络。"""
    n = len(d["node_kind_id"])
    target_mask = d["target_mask"].cpu()
    target_idx = target_mask.nonzero().squeeze(-1)
    if len(target_idx) == 0: target_idx = torch.arange(n)

    num_targets = len(target_idx)
    num_l1 = (num_targets + branch_k - 1) // branch_k
    l1_start_idx = n
    root_idx = n + num_l1

    num_meters = num_l1 + 1
    meter_kind = torch.full((num_meters,), 10, dtype=d["node_kind_id"].dtype)
    meter_base = torch.full((num_meters,), 29, dtype=d["node_base_id"].dtype)
    meter_base[-1] = 30
    meter_drive = torch.zeros(num_meters, dtype=torch.float32)
    meter_pin = torch.full((num_meters,), 19, dtype=d["node_pin_id"].dtype)

    k_id = torch.cat([d["node_kind_id"].cpu(), meter_kind])
    b_id = torch.cat([d["node_base_id"].cpu(), meter_base])
    d_val = torch.cat([d["node_drive_val"].cpu(), meter_drive])
    p_id = torch.cat([d["node_pin_id"].cpu(), meter_pin])

    # 构造规约边：Target -> L1 (类型 14), L1 -> Root (类型 15)
    l1_edges = torch.stack([target_idx, l1_start_idx + torch.arange(num_targets) // branch_k], dim=0)
    root_edges = torch.stack([torch.arange(l1_start_idx, l1_start_idx + num_l1), torch.full((num_l1,), root_idx, dtype=torch.long)], dim=0)

    ei = torch.cat([d["edge_index"].cpu(), l1_edges, root_edges], dim=1)
    eti = torch.cat([d["edge_type_id"].cpu(), torch.full((num_targets,), 14, dtype=torch.uint8), torch.full((num_l1,), 15, dtype=torch.uint8)], dim=0)

    total_nodes = n + num_meters
    out_deg, in_deg = torch.zeros(total_nodes, 1), torch.zeros(total_nodes, 1)
    out_deg.scatter_add_(0, ei[0].unsqueeze(-1), torch.ones_like(ei[0].unsqueeze(-1), dtype=torch.float32))
    in_deg.scatter_add_(0, ei[1].unsqueeze(-1), torch.ones_like(ei[1].unsqueeze(-1), dtype=torch.float32))

    return {
        "node_kind_id": k_id.to(device), "node_base_id": b_id.to(device), "node_drive_val": d_val.to(device),
        "node_pin_id": p_id.to(device), "node_log_out_deg": torch.log1p(out_deg).squeeze(-1).to(device),
        "node_log_in_deg": torch.log1p(in_deg).squeeze(-1).to(device), "edge_index": ei.to(device),
        "edge_type_id": eti.to(device), "root_meter_idx": root_idx, "max_delay": d["max_delay"]
    }


def load_p2_critical_timing_dataset(d2_dir: Path, split="train", limit=None, device="cpu"):
    d2_files = sorted(list(d2_dir.glob("*.pt")), key=lambda x: x.stem)
    c1, c2 = int(len(d2_files) * 0.8), int(len(d2_files) * 0.9)
    ranges = {"train": (0, c1), "val": (c1, c2), "test": (c2, len(d2_files))}
    cur_d2 = d2_files[ranges[split][0]:ranges[split][1]][:limit] if limit else d2_files[ranges[split][0]:ranges[split][1]]

    def load_one(f_path):
        d = torch.load(f_path, map_location="cpu", weights_only=False)
        at = d["arrival_time"]
        mask = d["target_mask"]
        valid_at = at[mask]
        if len(valid_at) == 0: return None
        # 截断极值异常离群点 (最大放宽至 3.0 ns，约覆盖 99.5% 正常芯片)
        max_at = min(valid_at.max().item(), 3.0)
        d["max_delay"] = float(max_at) # 单位: ns
        return inject_timing_hierarchical_meter(d, branch_k=32, device=device)

    print(f"🚀 32 线程并发载入关键路径时序样本 [{split}]...", flush=True)
    with ThreadPoolExecutor(max_workers=32) as pool:
        graphs = [g for g in pool.map(load_one, cur_d2) if g is not None]
    return graphs


@torch.no_grad()
def evaluate_critical_timing(model, graphs, max_scale):
    model.eval()
    p_all, t_all = [], []
    for g in graphs:
        pred_norm = model(g, g["root_meter_idx"])
        pred_ns = pred_norm * max_scale
        p_all.append(pred_ns)
        t_all.append(torch.tensor(g["max_delay"], device=pred_norm.device))
    model.train()

    preds, targets = torch.stack(p_all), torch.stack(t_all)
    # 转为皮秒评估
    preds_ps, targets_ps = preds * 1000.0, targets * 1000.0
    mae_ps = (preds_ps - targets_ps).abs().mean().item()
    mape = ((preds - targets).abs() / targets.clamp_min(1e-4)).mean().item() * 100.0
    r = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().clamp_min(1e-6).item()
    return {"mae_ps": mae_ps, "mape": mape, "r": r, "r2": 1.0 - ss_res / ss_tot}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d2_dir = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_04_d2_causal_rc_1500_1000_v1/D2")

    train_graphs = load_p2_critical_timing_dataset(d2_dir, split="train", device=device)
    val_graphs = load_p2_critical_timing_dataset(d2_dir, split="val", device=device)

    delays = torch.tensor([g["max_delay"] for g in train_graphs], device=device)
    max_scale = 3.0 # 统一归一化标尺 3.0 ns
    print(f"🌟 关键路径分层电表就绪: 训练集={len(train_graphs)}, 验证集={len(val_graphs)}, 延迟均值={delays.mean().item()*1000:.1f}ps")

    model = HierarchicalMeterCriticalTimingDynamicsModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        g = random.choice(train_graphs)
        pred_linear = model(g, g["root_meter_idx"])
        loss = F.smooth_l1_loss(pred_linear, torch.tensor(g["max_delay"], device=device) / max_scale, beta=0.1)
        (loss / 4).backward()
        if step % 4 == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

        if step % 100 == 0 or step == args.steps:
            m = evaluate_critical_timing(model, val_graphs, max_scale)
            print(f"step={step:4d} loss={loss.item():.4e} mae={m['mae_ps']:.1f}ps mape={m['mape']:.2f}% r={m['r']:.4f} R2={m['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
