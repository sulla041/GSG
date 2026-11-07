"""DeepGate2 纯真有向边数据加载器：仅提取 3 维 Gate One-hot，严格杜绝逻辑深度作弊。"""

from pathlib import Path
import numpy as np
import torch

DATA_DIR = Path("/data/yiyh/dac2027/dataset/raw/deepgate2/train")


def load_deepgate2_circuits(data_dir=DATA_DIR, max_nodes=512, limit=500, split="train", seed=20261103):
    """加载 DeepGate2 电路图，提取纯净的 3 维 Gate One-hot 和有向拓扑边。"""
    graphs_npz = data_dir / "graphs.npz"
    labels_npz = data_dir / "labels.npz"

    circuits = np.load(graphs_npz, allow_pickle=True)["circuits"].item()
    labels = np.load(labels_npz, allow_pickle=True)["labels"].item()

    names = sorted(list(circuits.keys()))
    rng = np.random.RandomState(seed)
    rng.shuffle(names)

    selected = []
    for name in names:
        g = circuits[name]
        n = g["x"].shape[0]
        if n <= max_nodes and name in labels:
            selected.append(name)
        if limit and len(selected) >= limit:
            break

    c1, c2 = int(len(selected) * 0.8), int(len(selected) * 0.9)
    ranges = {"train": (0, c1), "val": (c1, c2), "test": (c2, len(selected))}
    lo, hi = ranges[split]
    cur_names = selected[lo:hi]

    samples = []
    for name in cur_names:
        g = circuits[name]
        l = labels[name]
        gate_types = torch.tensor(g["x"][:, 1], dtype=torch.long)
        # 严格仅使用 3 维 One-hot (0: PI, 1: AND, 2: NOT)，彻底杜绝逻辑深度作弊
        node_feat = torch.nn.functional.one_hot(gate_types, num_classes=3).float()
        ei = torch.tensor(g["edge_index"], dtype=torch.long).t().contiguous() # [2, E]
        prob = torch.tensor(l["prob"], dtype=torch.float32)

        samples.append({
            "name": name,
            "nodes": node_feat,        # [N, 3]
            "edge_index": ei,          # [2, E]
            "prob": prob,              # [N]
            "num_nodes": node_feat.size(0),
            "num_edges": ei.size(1),
        })

    print(f"DeepGate2 [{split}] 载入完成！电路样本数: {len(samples)} (max_nodes={max_nodes})", flush=True)
    return samples
