"""CC-Tensor v3 的离散节点、稀疏边和监督目标编码器。"""

from __future__ import annotations

import math
from collections import deque

import torch

MASTER_NAMES = (
    "AND2_X1", "AND3_X1", "AND4_X1", "AOI211_X1", "AOI21_X1",
    "AOI221_X1", "AOI222_X1", "AOI22_X1", "DFF_X1", "INV_X1",
    "MUX2_X1", "NAND2_X1", "NAND3_X1", "NAND4_X1", "NOR2_X1",
    "NOR3_X1", "NOR4_X1", "OAI211_X1", "OAI21_X1", "OAI221_X1",
    "OAI222_X1", "OAI22_X1", "OAI33_X1", "OR2_X1", "OR3_X1",
    "OR4_X1", "XNOR2_X1", "XOR2_X1",
)
PIN_NAMES = ("A", "A1", "A2", "A3", "A4", "B", "B1", "B2", "B3",
             "C1", "C2", "CK", "D", "Q", "QN", "S", "Z", "ZN")
KIND_NAMES = ("cell", "cell_input_pin", "cell_output_pin", "net",
              "port_input", "port_output", "physical_cell",
              "physical_input_pin", "physical_output_pin", "other")


def _category_ids(meta: dict) -> dict[str, int]:
    return {str(k): int(v) for k, v in meta["category_ids"].items()}


def _argmax_category(nodes: torch.Tensor, meta: dict, prefix: str) -> list[int]:
    ids = _category_ids(meta)
    choices = [(name, idx) for name, idx in ids.items() if name.startswith(prefix)]
    out = []
    for row in nodes[:, : len(ids)]:
        out.append(max(choices, key=lambda item: float(row[item[1]]))[0] if choices else "")
    return out


def _master_drive(name: str) -> float:
    try:
        strength = float(name.rsplit("_X", 1)[1])
        return math.log2(max(strength, 1.0)) / 5.0
    except (IndexError, ValueError):
        return 0.0


def _degrees(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros(n, dtype=torch.float32)
    inc = torch.zeros(n, dtype=torch.float32)
    if edge_index.numel():
        out.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        inc.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    return torch.log1p(out), torch.log1p(inc)


def _depth_and_loop(edge_index: torch.Tensor, source: torch.Tensor, n: int):
    outgoing = [[] for _ in range(n)]
    indegree = [0] * n
    for src, dst in edge_index.t().tolist():
        outgoing[src].append(dst)
        indegree[dst] += 1
    queue = deque(i for i, value in enumerate(indegree) if value == 0)
    seen = 0
    while queue:
        node = queue.popleft(); seen += 1
        for dst in outgoing[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0: queue.append(dst)
    depth = [-1] * n
    for node in source.nonzero().flatten().tolist(): depth[node] = 0
    for _ in range(n):
        changed = False
        for src, dst in edge_index.t().tolist():
            if depth[src] >= 0 and depth[dst] < depth[src] + 1:
                depth[dst] = depth[src] + 1; changed = True
        if not changed: break
    return max((x for x in depth if x >= 0), default=0), seen != n


def encode_graph(graph: dict, meta: dict) -> dict:
    """将旧宽张量解码为紧凑字段；该过程不改变边方向和监督值。"""
    nodes = graph["nodes"]
    n = int(nodes.size(0))
    categories = _argmax_category(nodes, meta, "kind:")
    masters = _argmax_category(nodes, meta, "master:")
    pins = _argmax_category(nodes, meta, "pin:")
    kind_map = {name: i for i, name in enumerate(KIND_NAMES)}
    master_map = {name: i + 1 for i, name in enumerate(MASTER_NAMES)}
    pin_map = {name: i + 1 for i, name in enumerate(PIN_NAMES)}
    edge_index = graph["edge_index"].to(torch.int64).contiguous()
    out_deg, in_deg = _degrees(edge_index, n)
    source = graph.get("source_anchor_mask")
    if source is None:
        source = torch.tensor([x == "kind:port_input" for x in categories], dtype=torch.bool)
    kind_id = torch.tensor([kind_map.get(x.removeprefix("kind:"), 9) for x in categories], dtype=torch.uint8)
    base_id = torch.tensor([master_map.get(x.removeprefix("master:"), 0) for x in masters], dtype=torch.uint8)
    pin_id = torch.tensor([pin_map.get(x.removeprefix("pin:"), 0) for x in pins], dtype=torch.uint8)
    drive = torch.tensor([_master_drive(x.removeprefix("master:")) for x in masters], dtype=torch.float32)
    max_depth, has_loop = _depth_and_loop(edge_index, source, n)
    return {"node_kind_id": kind_id, "node_base_id": base_id, "node_drive_val": drive,
            "node_pin_id": pin_id, "node_log_out_deg": out_deg, "node_log_in_deg": in_deg,
            "source_anchor_mask": source.to(torch.bool), "edge_index": edge_index,
            "edge_type_id": graph["edge_type_id"].to(torch.uint8),
            "max_depth": max_depth, "has_loop": has_loop}


def compact_sample(data: dict, meta: dict, relation_names: list[str]) -> dict:
    """组装 pair 级样本，并保留原始映射、目标和有效掩码。"""
    inp = {"nodes": data["input_nodes"], "edge_index": data["input_edge_index"],
           "source_anchor_mask": data.get("source_anchor_mask"),
           "edge_type_id": torch.argmax(data["input_edge_attr"][:, :len(meta["relation_ids"])], 1)}
    out = {"nodes": data["output_nodes"], "edge_index": data["output_edge_index"],
           "edge_type_id": torch.argmax(data["output_edge_attr"][:, :len(meta["relation_ids"])], 1)}
    source_meta = {"category_ids": meta["category_ids"]}
    compact_in = encode_graph(inp, source_meta)
    compact_out = encode_graph(out, source_meta)
    timing_col = 10 if data["node_targets"].size(1) > 10 else 0
    arrival = data["node_targets"][:, timing_col].clone().float()
    arrival_mask = data["node_target_mask"][:, timing_col].clone().bool()
    compact_in["arrival_time"] = arrival
    compact_in["target_mask"] = arrival_mask
    compact_in["node_targets"] = data["node_targets"]
    compact_in["node_target_mask"] = data["node_target_mask"]
    compact_in["output"] = compact_out
    compact_in["input_to_output"] = data["input_to_output"].to(torch.int64)
    compact_in["graph_targets"] = data["graph_targets"]
    compact_in["graph_target_mask"] = data["graph_target_mask"]
    compact_in["relation_names"] = relation_names
    compact_in["meta"] = {"design_id": data["design_id"], "task": data["task"],
                           "track": data.get("track"), "num_nodes": int(inp["nodes"].size(0)),
                           "num_edges": int(inp["edge_index"].size(1)),
                           "max_depth": compact_in.pop("max_depth"),
                           "has_loop": compact_in.pop("has_loop"),
                           "num_targets": int(arrival_mask.sum()),
                           "arrival_stats": {"min": float(arrival[arrival_mask].min()) if arrival_mask.any() else 0.0,
                                             "max": float(arrival[arrival_mask].max()) if arrival_mask.any() else 0.0}}
    return compact_in
