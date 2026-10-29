"""CLRS 训练图适配：一个节点类型，null/edge/self 三类输入关系。"""

from pathlib import Path

import torch

DATA_ROOT = Path("/data/yiyh/dac2027/dataset/clrs30_bfs_hops_v1")
RELATION_TYPES = ["r_null", "edge", "self"]


def load_train(root=DATA_ROOT):
    """仅读取 train artifact，校验源数据约定；保留原文件。"""
    data = torch.load(Path(root) / "train.pt", map_location="cpu", weights_only=False)
    sentinel = int(data["metadata"]["unreachable_hop"])
    graphs = data["graphs"]
    if not graphs:
        raise ValueError("empty training split")
    for graph in graphs:
        x, r, y = graph["node_features"], graph["relation_vectors"], graph["hop_labels"]
        n = x.size(0)
        if x.shape != (n, 1) or r.shape != (n, n, 2) or y.shape != (n, n):
            raise ValueError("expected CLRS one-node/two-relation source schema")
        if graph["node_mask"].shape != (n,) or n == 0:
            raise ValueError("invalid node mask")
        if (y < 0).any() or (y > sentinel).any():
            raise ValueError("invalid hop labels")
    return graphs, sentinel


def make_batch(graphs, device):
    """在 CUDA 组装变长图，self 仅覆盖有效对角，padding 为零。"""
    if torch.device(device).type != "cuda":
        raise RuntimeError("V3 experiments require CUDA")
    n, b = max(g["node_features"].size(0) for g in graphs), len(graphs)
    nodes = torch.zeros(b, n, 1, device=device)
    relations = torch.zeros(b, n, n, 3, device=device)
    targets = torch.zeros(b, n, n, dtype=torch.long, device=device)
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    for index, graph in enumerate(graphs):
        size = graph["node_features"].size(0)
        nodes[index, :size] = graph["node_features"].to(device)
        relations[index, :size, :size, :2] = graph["relation_vectors"].to(device)
        targets[index, :size, :size] = graph["hop_labels"].to(device)
        mask[index, :size] = graph["node_mask"].to(device)
    eye = torch.eye(n, device=device, dtype=torch.bool)[None]
    relations[..., :2].masked_fill_(eye[..., None], 0)
    relations[..., 2] = eye & mask[:, :, None]
    pairs = mask[:, :, None] & mask[:, None, :]
    relations.masked_fill_(~pairs[..., None], 0)
    nodes.masked_fill_(~mask[..., None], 0)
    return nodes, relations, targets, mask, pairs
