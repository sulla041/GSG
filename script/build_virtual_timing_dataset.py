"""从已有固定网表和 OpenSTA 路径标签构造方案1原始数据集。"""

import argparse
import json
import time
from pathlib import Path

from dataset.virtual_timing_graph import build_pin_graph
from dataset.virtual_timing_labels import build_virtual_labels


def _build_one(netlist: Path, timing: Path, output: Path, liberty: Path,
               topk: int, min_labeled_pairs: int) -> dict:
    """生成一个样本，保留输入图、目标和审计 metadata。"""
    graph = build_pin_graph(netlist, liberty)
    labels = build_virtual_labels(graph, timing, topk)
    labeled = labels["metadata"]["labeled_pair_count"]
    if labeled < min_labeled_pairs:
        return {"status": "excluded", "reason": "insufficient_labeled_pairs",
                "labeled_pair_count": labeled, "node_count": len(graph["nodes"])}
    output.mkdir(parents=True, exist_ok=True)
    (output / "input").mkdir(exist_ok=True)
    (output / "target").mkdir(exist_ok=True)
    (output / "input/timing_graph.json").write_text(json.dumps(graph, indent=2))
    (output / "target/nodes.json").write_text(json.dumps(labels["nodes"], indent=2))
    (output / "target/relations.json").write_text(json.dumps(labels["relations"], indent=2))
    (output / "target/paths.json").write_text(json.dumps(labels["paths"], indent=2))
    manifest = {"netlist": str(netlist), "timing_paths": str(timing),
                "graph_metadata": graph["metadata"], "target_metadata": labels["metadata"],
                "node_count": len(graph["nodes"]), "relation_count": len(graph["relations"])}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    """按同名设计目录批量生成样本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist-root", type=Path, required=True)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--liberty", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--min-labeled-pairs", type=int, default=1)
    args = parser.parse_args()
    timing_files = sorted(args.timing_root.glob("*/timing_paths.json"))
    if args.limit:
        timing_files = timing_files[:args.limit]
    next_log, accepted, excluded = time.monotonic() + 5.0, 0, 0
    for index, timing in enumerate(timing_files, 1):
        netlist = args.netlist_root / timing.parent.name / "netlist.json"
        if not netlist.is_file():
            continue
        result = _build_one(netlist, timing, args.output / timing.parent.name,
                            args.liberty, args.topk, args.min_labeled_pairs)
        if result["status"] == "excluded":
            excluded += 1
        else:
            accepted += 1
        if time.monotonic() >= next_log:
            print(f"progress designs={index}/{len(timing_files)} accepted={accepted}", flush=True)
            next_log = time.monotonic() + 5.0
    manifest = {"task": "virtual_timing_reg2reg_setup_v1", "designs": accepted,
                "excluded": excluded, "source_timing_root": str(args.timing_root),
                "topk": args.topk, "min_labeled_pairs": args.min_labeled_pairs}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"virtual_timing_dataset designs={accepted} output={args.output}")


if __name__ == "__main__":
    main()
