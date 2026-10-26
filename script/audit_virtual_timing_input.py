"""审计 virtual timing 任务的输入字段是否足以支持 delay 预测。"""

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    """输出输入字段、标签字段和明确缺口，不读取 target 作为模型输入。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/yiyh/graphmath/dac2027/processed/verilog_rtl_corpus"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/dac2027_virtual_timing_v1"))
    parser.add_argument("--split", type=Path, default=Path("dataset/virtual_timing_split.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/dac2027_virtual_timing_input_audit.json"))
    args = parser.parse_args()
    manifest = json.loads(args.split.read_text())
    design_id = manifest["splits"]["train"][0]
    graph = torch.load(args.root / "graphs_timing_arc_v1" / f"{design_id}.pt",
                       map_location="cpu", weights_only=False)
    label = torch.load(args.labels / design_id / "labels.pt",
                       map_location="cpu", weights_only=False)
    report = {
        "task": "virtual_timing_v1",
        "sample_design": design_id,
        "split": manifest["split_counts"],
        "input_node_fields": ["node_kind", "node_direction", "node_cell_family",
                               "node_drive_strength", "node_is_sequential", "node_pin_cap"],
        "input_relation_fields": ["r_null", "CELL_ARC", "NET_ARC",
                                   "cell_arc_in_out_degree", "net_arc_in_out_degree"],
        "label_fields": label["metadata"],
        "available_shapes": {key: list(value.shape) for key, value in graph.items()
                             if hasattr(value, "shape")},
        "allowed_native_process_features_if_aligned": [
            "cell_or_pin_library_attributes",
            "edge_arc_type_and_related_pin",
            "arc_input_slew_and_output_load_from_input_conditions",
            "PVT_corner_and_library_condition_id",
        ],
        "forbidden_derived_graph_statistics": [
            "degree_or_fanout_aggregates_added_only_for_this_task",
            "path_delay_or_slack_derived_input",
            "global_graph_pooling_statistics_broadcast_to_nodes",
        ],
        "policy": "use native aligned X/R fields; do not add secondary graph statistics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"virtual_timing_input_audit design={design_id} output={args.output}")


if __name__ == "__main__":
    main()
