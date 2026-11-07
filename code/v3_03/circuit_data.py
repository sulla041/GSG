"""V3.03 电路张量极速加载器：带实时进度回显与高效内存/路径管理。"""

import json
import sys
from pathlib import Path
import torch

TENSOR_DIR = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_03_lossless_tensors")
META_INDEX = TENSOR_DIR / "meta_index.json"
DIE_SIZES_FILE = TENSOR_DIR / "die_sizes.json"
COORD_SCALE = 400000.0


def load_circuit_tensors(tensor_dir=TENSOR_DIR, max_nodes=2048, split="train", seed=20261101, limit=None):
    """从无损张量化目录按索引加载样本，并在终端实时打印清晰进度。"""
    meta_path = tensor_dir / "meta_index.json"
    index_data = json.loads(meta_path.read_text())
    die_sizes_data = json.loads((tensor_dir / "die_sizes.json").read_text()) if (tensor_dir / "die_sizes.json").exists() else {}
    items = index_data["items"]
    selected_items = [x for x in items if x["nodes"] <= max_nodes]
    if not selected_items:
        raise ValueError(f"No tensors match max_nodes={max_nodes}")

    selected_items.sort(key=lambda x: x["design_id"])
    cut1, cut2 = int(len(selected_items) * 0.8), int(len(selected_items) * 0.9)
    ranges = {"train": (0, cut1), "val": (cut1, cut2), "test": (cut2, len(selected_items))}
    lo, hi = ranges[split]
    current_items = selected_items[lo:hi]
    if limit is not None:
        current_items = current_items[:limit]

    total = len(current_items)
    print(f"开始加载 [{split}] 数据集张量 (共 {total} 个设计, max_nodes={max_nodes})...", flush=True)

    graphs = []
    log_interval = max(1, total // 10)
    for idx, item in enumerate(current_items, start=1):
        pt_path = tensor_dir / f"{item['design_id']}.pt"
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        # 挂载芯片物理尺寸 [W_die, H_die]
        ds = die_sizes_data.get(item["design_id"], [0.1, 0.1])
        data["die_size"] = torch.tensor(ds, dtype=torch.float32)
        graphs.append(data)
        if idx == 1 or idx == total or idx % log_interval == 0:
            pct = (idx / total) * 100.0
            print(f"  [进度: {idx:3d}/{total:3d}] ({pct:5.1f}%) 已载入: {item['design_id']} (nodes={item['nodes']})", flush=True)

    node_w = graphs[0]["nodes"].size(1) if graphs else index_data.get("node_width", 64)
    rel_w = graphs[0]["relations"].size(-1) if graphs else index_data.get("relation_width", 12)

    meta = {
        "node_width": node_w, "relation_width": rel_w, "coord_scale": COORD_SCALE,
        "max_nodes": max_nodes, "total_selected": len(selected_items),
        "split_count": len(graphs), "split": split, "seed": seed,
    }
    print(f"[{split}] 数据集全量加载完成！常驻内存图数量: {len(graphs)} (node_w={node_w}, rel_w={rel_w})\n", flush=True)
    return graphs, meta


def make_batch(graphs, device):
    """组装变长图为 batch，支持动态填充与 die_size 打包。"""
    size = max(g["nodes"].size(0) for g in graphs)
    batch = len(graphs)
    node_w = graphs[0]["nodes"].size(1)
    rel_w = graphs[0]["relations"].size(-1)

    nodes = torch.zeros(batch, size, node_w, device=device)
    relations = torch.zeros(batch, size, size, rel_w, device=device)
    targets = torch.zeros(batch, size, 2, device=device)
    target_mask = torch.zeros(batch, size, dtype=torch.bool, device=device)
    node_mask = torch.zeros(batch, size, dtype=torch.bool, device=device)
    die_sizes = torch.zeros(batch, 2, device=device)

    for i, g in enumerate(graphs):
        n = g["nodes"].size(0)
        nodes[i, :n] = g["nodes"].to(device)
        relations[i, :n, :n] = g["relations"].to(device)
        targets[i, :n] = g["targets"].to(device)
        target_mask[i, :n] = g["target_mask"].to(device)
        node_mask[i, :n] = True
        if "die_size" in g:
            die_sizes[i] = g["die_size"].to(device)

    return nodes, relations, targets, target_mask, node_mask, die_sizes
