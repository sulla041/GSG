"""P2 任务：基于意涵“32-边平衡分层电表网络”的整图总面积 (Total Area) 训练流水线。
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

from code.v3_04.hierarchical_meter_dynamics import HierarchicalMeterAreaDynamicsModel


def inject_hierarchical_meter_network(d, branch_k=32, device="cpu"):
    """为图结构动态构建 32-边平衡分层金字塔电表网络。"""
    n = len(d["node_kind_id"])
    cell_idx = (d["node_kind_id"].cpu() == 0).nonzero().squeeze(-1)
    if len(cell_idx) == 0: cell_idx = torch.arange(n)
    
    num_cells = len(cell_idx)
    # 1. 构建 L1 电表：每 branch_k 个 Cell 汇聚到一个 L1 电表
    num_l1 = (num_cells + branch_k - 1) // branch_k
    l1_start_idx = n
    root_idx = n + num_l1 # Root 电表为最后一个节点
    
    # 构造额外节点属性 (追加 num_l1 个 L1 节点 + 1 个 Root 节点)
    num_meters = num_l1 + 1
    meter_kind = torch.full((num_meters,), 10, dtype=d["node_kind_id"].dtype) # 10: METER_NODE
    meter_base = torch.full((num_meters,), 29, dtype=d["node_base_id"].dtype) # 29: METER_L1
    meter_base[-1] = 30 # 30: METER_ROOT
    meter_drive = torch.zeros(num_meters, dtype=torch.float32)
    meter_pin = torch.full((num_meters,), 19, dtype=d["node_pin_id"].dtype)
    
    k_id = torch.cat([d["node_kind_id"].cpu(), meter_kind])
    b_id = torch.cat([d["node_base_id"].cpu(), meter_base])
    d_val = torch.cat([d["node_drive_val"].cpu(), meter_drive])
    p_id = torch.cat([d["node_pin_id"].cpu(), meter_pin])
    
    # 2. 构造连接边：Cell -> L1 (类型 14), L1 -> Root (类型 15)
    l1_edges = torch.stack([cell_idx, l1_start_idx + torch.arange(num_cells) // branch_k], dim=0)
    root_edges = torch.stack([torch.arange(l1_start_idx, l1_start_idx + num_l1), torch.full((num_l1,), root_idx, dtype=torch.long)], dim=0)
    
    ei = torch.cat([d["edge_index"].cpu(), l1_edges, root_edges], dim=1)
    eti = torch.cat([d["edge_type_id"].cpu(), torch.full((num_cells,), 14, dtype=torch.uint8), torch.full((num_l1,), 15, dtype=torch.uint8)], dim=0)

    total_nodes = n + num_meters
    out_deg, in_deg = torch.zeros(total_nodes, 1), torch.zeros(total_nodes, 1)
    out_deg.scatter_add_(0, ei[0].unsqueeze(-1), torch.ones_like(ei[0].unsqueeze(-1), dtype=torch.float32))
    in_deg.scatter_add_(0, ei[1].unsqueeze(-1), torch.ones_like(ei[1].unsqueeze(-1), dtype=torch.float32))

    res = {
        "node_kind_id": k_id.to(device), "node_base_id": b_id.to(device), "node_drive_val": d_val.to(device),
        "node_pin_id": p_id.to(device), "node_log_out_deg": torch.log1p(out_deg).squeeze(-1).to(device),
        "node_log_in_deg": torch.log1p(in_deg).squeeze(-1).to(device), "edge_index": ei.to(device),
        "edge_type_id": eti.to(device), "root_meter_idx": root_idx
    }
    if "area_target" in d: res["area_target"] = d["area_target"]
    if "power_target" in d: res["power_target"] = d["power_target"]
    return res


def load_p2_hierarchical_dataset(d2_dir: Path, p2_dir: Path, split="train", limit=None, device="cpu"):
    p2_files = sorted(list(p2_dir.glob("*.pt")), key=lambda x: x.stem)
    c1, c2 = int(len(p2_files) * 0.8), int(len(p2_files) * 0.9)
    ranges = {"train": (0, c1), "val": (c1, c2), "test": (c2, len(p2_files))}
    cur_p2 = p2_files[ranges[split][0]:ranges[split][1]][:limit] if limit else p2_files[ranges[split][0]:ranges[split][1]]

    def load_one(p2_path):
        d2_path = d2_dir / f"{p2_path.stem}.pt"
        if not d2_path.exists(): return None
        p2_data = torch.load(p2_path, map_location="cpu", weights_only=False)
        gt, mask = p2_data.get("graph_targets"), p2_data.get("graph_target_mask")
        if gt is None or mask is None or not mask[0]: return None
        d2_data = torch.load(d2_path, map_location="cpu", weights_only=False)
        d2_data["area_target"] = float(gt[0].item() * 1e-6) # 换算为 um^2
        return inject_hierarchical_meter_network(d2_data, branch_k=32, device=device)

    print(f"🚀 32 线程并发载入 P2 32-边分层电表网络样本 [{split}]...", flush=True)
    with ThreadPoolExecutor(max_workers=32) as pool:
        graphs = [g for g in pool.map(load_one, cur_p2) if g is not None]
    return graphs


@torch.no_grad()
def evaluate_hierarchical_linear(model, graphs, max_area):
    model.eval()
    p_all, t_all = [], []
    for g in graphs:
        pred_norm = model(g, g["root_meter_idx"])
        pred_um2 = pred_norm * max_area
        p_all.append(pred_um2)
        t_all.append(torch.tensor(g["area_target"], device=pred_norm.device))
    model.train()
    
    preds, targets = torch.stack(p_all), torch.stack(t_all)
    mae = (preds - targets).abs().mean().item()
    mape = ((preds - targets).abs() / targets.clamp_min(1.0)).mean().item() * 100.0
    r = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().clamp_min(1e-4).item()
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

    train_graphs = load_p2_hierarchical_dataset(d2_dir, p2_dir, split="train", device=device)
    val_graphs = load_p2_hierarchical_dataset(d2_dir, p2_dir, split="val", device=device)

    areas = torch.tensor([g["area_target"] for g in train_graphs], device=device)
    max_area = areas.max()
    print(f"🌟 32-边平衡分层电表网络就绪: 训练集={len(train_graphs)}, 验证集={len(val_graphs)}, max_area={max_area.item():.1f}um2")

    model = HierarchicalMeterAreaDynamicsModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        g = random.choice(train_graphs)
        pred_linear = model(g, g["root_meter_idx"])
        loss = F.smooth_l1_loss(pred_linear, torch.tensor(g["area_target"], device=device) / max_area, beta=0.1)
        (loss / 4).backward()
        if step % 4 == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

        if step % 100 == 0 or step == args.steps:
            m = evaluate_hierarchical_linear(model, val_graphs, max_area)
            print(f"step={step:4d} loss={loss.item():.4e} mae={m['mae']:.2f}um2 mape={m['mape']:.2f}% r={m['r']:.4f} R2={m['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
