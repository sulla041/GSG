"""V3.03 CC-Tensor v3 紧凑因果张量极速数据加载器：
支持原生 1500-design / 4096 节点扩展数据集，显存毫秒级直载。
"""

import json
from pathlib import Path
import torch

DEFAULT_TENSOR_DIR = Path("/data/yiyh/dac2027/dataset/circuit/circuitnet3_v3_03_compact_tensors_1500_4096_v2")


def load_compact_d1_tensors(root_dir=DEFAULT_TENSOR_DIR, max_nodes=4096,
                            split="train", seed=20261103, limit=None, device="cpu"):
    """极速加载紧凑因果 D1 张量，支持直接全图常驻显存。"""
    root_dir = Path(root_dir)
    meta_json = json.loads((root_dir / "meta_index.json").read_text())
    items = [x for x in meta_json["results"] if x.get("task") == "D1" and x.get("status") == "ready"]
    selected = [x for x in items if x["nodes"] <= max_nodes]
    if not selected:
        raise ValueError(f"No compact D1 tensors match max_nodes={max_nodes} in {root_dir}")

    selected.sort(key=lambda x: x["design_id"])
    c1, c2 = int(len(selected) * 0.8), int(len(selected) * 0.9)
    ranges = {"train": (0, c1), "val": (c1, c2), "test": (c2, len(selected))}
    lo, hi = ranges[split]
    cur_items = selected[lo:hi]
    if limit is not None:
        cur_items = cur_items[:limit]

    print(f"🚀 载入 CC-Tensor v3 D1 [{split}] 数据集: 共 {len(cur_items)}/{len(selected)} 设计 (max_nodes={max_nodes})...", flush=True)
    graphs = []
    for item in cur_items:
        pt_path = Path(item["file"])
        if not pt_path.exists():
            pt_path = root_dir / "D1" / f"{item['design_id']}.pt"
        data = torch.load(pt_path, map_location=device, weights_only=False)
        graphs.append(data)

    print(f"[{split}] 紧凑张量加载完成！共 {len(graphs)} 个设计，100% 就绪 (常驻设备: {device})\n", flush=True)
    return graphs


def compute_compact_d1_log_stats(graphs, eps=1e-12):
    """计算紧凑因果数据集的全局对数时序统计均值与方差。"""
    vals = [g["arrival_time"][g["target_mask"]] for g in graphs if g["target_mask"].any()]
    if not vals:
        return torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)
    log_v = torch.log(torch.cat(vals).clamp_min(eps))
    return log_v.mean(), log_v.std().clamp_min(1e-4)
