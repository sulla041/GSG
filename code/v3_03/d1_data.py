"""V3.03 D1 时序预测数据加载器：加载预张量化的 D1 图并动态组装 Batch。"""

import json
from pathlib import Path
import torch

D1_TENSOR_DIR = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_03_task_tensors_v2/D1")
INDEX_PATH = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_03_task_tensors_v2/meta_index.json")


def load_d1_tensors(tensor_dir=D1_TENSOR_DIR, index_path=INDEX_PATH, max_nodes=1024,
                    split="train", seed=20261103, limit=None):
    """从 D1 缓存目录按 design_id 稳定切分加载样本图。"""
    meta = json.loads(index_path.read_text())
    items = [x for x in meta["results"] if x.get("task") == "D1" and x.get("status") == "ready"]
    selected = [x for x in items if x["input_nodes"] <= max_nodes]
    if not selected:
        raise ValueError(f"No D1 tensors match max_nodes={max_nodes}")

    selected.sort(key=lambda x: x["design_id"])
    c1, c2 = int(len(selected) * 0.8), int(len(selected) * 0.9)
    ranges = {"train": (0, c1), "val": (c1, c2), "test": (c2, len(selected))}
    lo, hi = ranges[split]
    cur_items = selected[lo:hi]
    if limit is not None:
        cur_items = cur_items[:limit]

    print(f"载入 D1 [{split}] 数据集 (共 {len(cur_items)}/{len(selected)} 设计, max_nodes={max_nodes})...", flush=True)
    graphs = []
    for item in cur_items:
        pt_path = Path(item["file"])
        if not pt_path.exists():
            pt_path = tensor_dir / f"{item['design_id']}.pt"
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        graphs.append(data)

    node_w = graphs[0]["input_nodes"].size(1) if graphs else meta.get("node_width", 76)
    rel_w = graphs[0]["input_edge_attr"].size(1) if graphs else meta.get("relation_width", 30)

    data_meta = {
        "node_width": node_w, "relation_width": rel_w,
        "max_nodes": max_nodes, "split_count": len(graphs),
        "split": split, "seed": seed, "total_ready": len(selected),
        "topology_fixed": any("topology_audit" in g for g in graphs)
    }
    print(f"[{split}] 加载完成！样本数: {len(graphs)} (node_w={node_w}, rel_w={rel_w})\n", flush=True)
    return graphs, data_meta


def compute_d1_log_stats(graphs, timing_channels=(10, 11), eps=1e-12):
    """计算训练集有效时序目标的对数空间均值与标准差。"""
    means, stds = [], []
    for ch in timing_channels:
        vals = [g["node_targets"][g["node_target_mask"][:, ch], ch]
                for g in graphs if g["node_target_mask"][:, ch].any()]
        if not vals:
            means.append(0.0); stds.append(1.0); continue
        log_v = torch.log(torch.cat(vals).clamp_min(eps))
        means.append(log_v.mean().item())
        stds.append(log_v.std().clamp_min(1e-4).item())
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)


def make_d1_batch(graphs, device, timing_channels=(10, 11)):
    """组装变长图为 batch，支持动态密集化关系矩阵与时序通道切片。"""
    batch = len(graphs)
    size = max(g["input_nodes"].size(0) for g in graphs)
    node_w = graphs[0]["input_nodes"].size(1)
    rel_w = graphs[0]["input_edge_attr"].size(1)
    n_ch = len(timing_channels)

    nodes = torch.zeros(batch, size, node_w, device=device)
    relations = torch.zeros(batch, size, size, rel_w, device=device)
    targets = torch.zeros(batch, size, n_ch, device=device)
    target_mask = torch.zeros(batch, size, n_ch, dtype=torch.bool, device=device)
    node_mask = torch.zeros(batch, size, dtype=torch.bool, device=device)

    for i, g in enumerate(graphs):
        n = g["input_nodes"].size(0)
        nodes[i, :n] = g["input_nodes"].to(device)
        src, dst = g["input_edge_index"]
        relations[i, src, dst] = g["input_edge_attr"].to(device)
        for c_idx, ch in enumerate(timing_channels):
            targets[i, :n, c_idx] = g["node_targets"][:, ch].to(device)
            target_mask[i, :n, c_idx] = g["node_target_mask"][:, ch].to(device)
        node_mask[i, :n] = True

    return nodes, relations, targets, target_mask, node_mask
