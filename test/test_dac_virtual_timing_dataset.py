"""验证 virtual timing label 能回填为 V1-C full-node 输入。"""

import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))


def main() -> None:
    """用最小 DAC-like artifact 检查 scatter、padding 和 V1-C 前向。"""
    from dataset.dac_virtual_timing_dataset import collate_virtual_timing, load_virtual_timing
    from v1_c.model import V1CModel

    with tempfile.TemporaryDirectory(prefix="gsg_vt_dataset_") as folder:
        root, labels = Path(folder), Path(folder) / "labels"
        design_id = "toy"
        (root / "graphs_timing_arc_v1").mkdir(parents=True)
        (labels / design_id).mkdir(parents=True)
        torch.save({"n": 4, "node_names": ["r0/Q", "r0/D", "r1/Q", "r1/D"],
                    "node_kind": torch.tensor([0, 0, 0, 0]),
                    "node_direction": torch.tensor([1, 0, 1, 0]),
                    "node_cell_family": torch.tensor([0, 0, 0, 0]),
                    "node_drive_strength": torch.ones(4),
                    "node_is_sequential": torch.ones(4), "node_pin_cap": torch.ones(4),
                    "edge_index": torch.tensor([[0, 1], [1, 2]]),
                    "edge_kind": torch.tensor([0, 1]), "arc_schema": {"CELL_ARC": 0, "NET_ARC": 1}},
                   root / "graphs_timing_arc_v1" / f"{design_id}.pt")
        torch.save({"endpoint_node_index": torch.tensor([0, 1, 2, 3]),
                    "node_target": torch.ones(4, 3), "node_valid": torch.ones(4, 3, dtype=torch.bool),
                    "relation_target": torch.ones(4, 4, 3),
                    "relation_valid": torch.eye(4, dtype=torch.bool),
                    "metadata": {"task": "virtual_timing_v1"}}, labels / design_id / "labels.pt")
        import dataset.dac_virtual_timing_dataset as module
        module.load_timing = lambda *args, **kwargs: {"nodes": torch.ones(4, 29),
            "relations": torch.zeros(4, 4, 3), "node_mask": torch.ones(4, dtype=torch.bool),
            "relation_class": torch.zeros(4, 4, dtype=torch.long), "relation_target": torch.zeros(4, 4, 3),
            "relation_valid": torch.zeros(4, 4, dtype=torch.bool)}
        sample = load_virtual_timing(root, labels, design_id)
        batch = collate_virtual_timing([sample, sample])
        model = V1CModel(29, 3, 8, 6, 16, layers=1)
        output = model(batch["nodes"], batch["relations"], batch["node_mask"])
        assert output[0].shape == (2, 4, 8)
        assert batch["relation_query_mask"].sum().item() == 8
    print("dac_virtual_timing_dataset_smoke_pass node_width=29 relation_width=3")


if __name__ == "__main__":
    main()
