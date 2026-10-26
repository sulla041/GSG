"""方案1数据集的图构造、端点对齐、聚合和覆盖率 smoke。"""

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """使用最小寄存器到寄存器网表验证数据契约。"""
    from dataset.virtual_timing_graph import build_pin_graph
    from dataset.virtual_timing_labels import build_virtual_labels

    with tempfile.TemporaryDirectory(prefix="gsg_virtual_timing_") as folder:
        root = Path(folder)
        netlist = root / "netlist.json"
        liberty = root / "test.lib"
        paths = root / "timing_paths.json"
        netlist.write_text(json.dumps({"modules": {"top": {
            "ports": {"clk": {"direction": "input", "bits": [2]}},
            "cells": {"r0": {"type": "DFF", "connections": {"Q": [3], "D": [4], "CLK": [2]}},
                      "r1": {"type": "DFF", "connections": {"Q": [5], "D": [6], "CLK": [2]}},
                      "g0": {"type": "BUF", "connections": {"A": [3], "Y": [6]}}},
            "netnames": {"clk": {"bits": [2]}, "n0": {"bits": [3]},
                         "n1": {"bits": [6]}, "q1": {"bits": [5]}, "d0": {"bits": [4]}}}}}))
        liberty.write_text("""
cell (DFF) { ff (IQ) { } pin (Q) { direction : output; } pin (D) { direction : input; }
pin (CLK) { direction : input; clock : true; } }
cell (BUF) { pin (A) { direction : input; } pin (Y) { direction : output; } }
""")
        paths.write_text(json.dumps({"checks": [{
            "path_type": "max", "path_group": "clk", "startpoint": "r0/Q",
            "endpoint": "r1/D", "source_path": [{"pin": "r0/Q", "arrival": 1.0},
                         {"pin": "g0/A", "arrival": 1.0}, {"pin": "g0/Y", "arrival": 3.0},
                         {"pin": "r1/D", "arrival": 3.0}],
            "data_arrival_time": 3.0, "required_time": 2.5, "slack": -0.5}]}))
        graph = build_pin_graph(netlist, liberty)
        labels = build_virtual_labels(graph, paths, topk=1)
        assert "r0/Q" in graph["aliases"] and "r1/D" in graph["aliases"]
        assert any(x["relation_type"] == "net_arc" for x in graph["relations"])
        labeled = [x for x in labels["relations"] if x["has_label"]]
        assert len(labeled) == 1
        assert labeled[0]["violation"] and labeled[0]["contains_topk"]
        assert labels["metadata"]["pair_coverage"] == 1.0
    print("virtual_timing_dataset_smoke_pass nodes=9 labeled_pairs=1")


if __name__ == "__main__":
    main()
