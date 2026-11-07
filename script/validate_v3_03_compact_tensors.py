#!/usr/bin/env python3
"""逐样本验收 CC-Tensor v3 与旧 pair 张量的结构和监督对齐。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def check_one(row: dict, root: Path) -> dict:
    data = torch.load(row["file"], map_location="cpu", weights_only=False)
    task = data["meta"]["task"]
    old_name = "circuitnet3_v3_03_task_tensors_v2" if task == "D1" else "circuitnet3_v3_03_task_tensors_v1"
    old = torch.load(root / old_name / task / f"{row['design_id']}.pt", map_location="cpu", weights_only=False)
    checks = ((data["edge_index"], old["input_edge_index"]),
              (data["input_to_output"], old["input_to_output"]),
              (data["node_targets"], old["node_targets"]),
              (data["node_target_mask"], old["node_target_mask"]),
              (data["graph_targets"], old["graph_targets"]),
              (data["graph_target_mask"], old["graph_target_mask"]))
    if not all(torch.equal(a, b) for a, b in checks):
        raise AssertionError(f"监督或边表不一致: {row['design_id']}")
    if "logic_level" in data or (task == "D1" and data["meta"]["has_loop"]):
        raise AssertionError(f"发现非法逻辑层级或环: {row['design_id']}")
    return {"task": task, "nodes": data["meta"]["num_nodes"],
            "edges": data["meta"]["num_edges"], "targets": data["meta"]["num_targets"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/yiyh/dac2027/dataset/circuit"))
    parser.add_argument("--index", type=Path, default=None)
    args = parser.parse_args()
    output = args.root / "circuitnet3_v3_03_compact_tensors"
    index = args.index or output / "meta_index.json"
    rows = json.loads(index.read_text())["results"]
    reports = [check_one(row, args.root) for row in rows if row.get("status") == "ready"]
    counts = Counter(row["task"] for row in reports)
    d1 = [row for row in reports if row["task"] == "D1"]
    print(json.dumps({"checked": len(reports), "tasks": dict(counts),
                      "d1_checked": len(d1), "output": str(output)}, sort_keys=True))
    assert len(reports) == 1499


if __name__ == "__main__":
    main()
