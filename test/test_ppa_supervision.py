"""验证 PPA 标签映射、padding、mask 和 V1-C 监督反向传播。"""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))


def main() -> None:
    """在一个真实 pair 上检查四类关系和三类 timing 标签。"""
    from dataset.ppa_supervised import load_ppa_pair
    from v1_c.model import V1CModel
    from v1_c.ppa_loss import ppa_loss
    from v1_c.ppa_metrics import ppa_metrics

    pair = ROOT / "dataset/paired_graphs/timingpredict__spm"
    liberty = ROOT / "dataset/raw/timingpredict_extracted/netlists/techlib/sky130_fd_sc_hd__ss_100C_1v60.lib"
    sample = load_ppa_pair(pair, liberty, max_nodes=4096)
    assert sample["nodes"].size(0) == sample["node_target"].size(0)
    assert sample["node_target"].size(-1) == 10
    assert sample["relation_target"].size(-1) == 3
    assert sample["relation_class"].max().item() <= 3
    assert sample["relation_valid"].sum().item() > 0
    model = V1CModel(sample["nodes"].size(-1), 3, 8, 6, 16, layers=1)
    output = model(sample["nodes"][None], sample["relations"][None],
                   sample["node_mask"][None], return_ppa=True)
    node_pred, class_pred, relation_pred = output[-3:]
    loss, metrics = ppa_loss(node_pred, sample["node_target"][None],
                             sample["node_valid"][None], class_pred, relation_pred,
                             sample["relation_class"][None],
                             sample["relation_class_mask"][None],
                             sample["relation_target"][None],
                             sample["relation_valid"][None])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    scores = ppa_metrics(node_pred, sample["node_target"][None],
                         sample["node_valid"][None], class_pred, relation_pred,
                         sample["relation_class"][None],
                         sample["relation_class_mask"][None],
                         sample["relation_dense_mask"][None],
                         sample["relation_target"][None],
                         sample["relation_valid"][None])
    assert set(scores) == {"relation_acc", "dense_relation_acc", "non_null_relation_acc",
                           "edge_precision", "edge_recall", "edge_f1", "critical_acc",
                           "node_within_1std", "timing_within_1std"}
    assert all(torch.isfinite(value) for value in scores.values())
    print(f"ppa_supervision_smoke_pass nodes={sample['nodes'].size(0)} loss={loss.item():.3e}")


if __name__ == "__main__":
    main()
