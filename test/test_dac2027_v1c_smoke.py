"""验证 DAC 两种图视图可以接入 V1-C 的前向与反向路径。"""

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DAC_ROOT = Path("/home/yiyh/graphmath/dac2027/processed/verilog_rtl_corpus")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))


def _small_train_id() -> str:
    """使用冻结 split 中已知的小型训练设计，限制 smoke 显存。"""
    split = json.loads((DAC_ROOT / "splits/seed42_mapped_circuit_v2.json").read_text())
    return split["train"][1]


def _run_mapped(model, device: torch.device, design_id: str) -> float:
    """验证 mapped-cell 五维节点标签和三类显式关系。"""
    from dataset.dac2027 import load_mapped

    sample = load_mapped(DAC_ROOT, design_id)
    nodes = sample["nodes"][None].to(device)
    relations = sample["relations"][None].to(device)
    mask = sample["node_mask"][None].to(device)
    node_pred, relation_logits, _ = model(nodes, relations, mask,
                                          return_ppa=True, ppa_only=True)
    target = sample["node_target"][None].to(device)
    valid = sample["node_valid"][None].to(device)
    node_loss = (F.smooth_l1_loss(node_pred[..., :5], target, reduction="none")
                 * valid).sum() / valid.sum().clamp_min(1)
    relation_target = sample["relations"].argmax(-1)[None].to(device)
    relation_loss = F.cross_entropy(relation_logits[..., :3].permute(0, 3, 1, 2),
                                    relation_target)
    loss = node_loss + relation_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert node_pred.shape[-1] == 10
    assert relation_logits.shape[-1] == 4
    return float(loss.detach().cpu())


def _run_timing(model, device: torch.device, design_id: str) -> float:
    """验证 pin timing 的三类关系和有效 CELL_ARC delay 标签。"""
    from dataset.dac2027 import load_timing

    sample = load_timing(DAC_ROOT, design_id)
    nodes = sample["nodes"][None].to(device)
    relations = sample["relations"][None].to(device)
    mask = sample["node_mask"][None].to(device)
    _, relation_logits, relation_pred = model(nodes, relations, mask,
                                               return_ppa=True, ppa_only=True)
    class_target = sample["relations"].argmax(-1)[None].to(device)
    class_loss = F.cross_entropy(relation_logits[..., :3].permute(0, 3, 1, 2),
                                 class_target)
    valid = sample["relation_valid"][None].to(device)
    target = sample["relation_target"][None].to(device)
    delay_loss = (F.smooth_l1_loss(relation_pred, target, reduction="none")
                  * valid.unsqueeze(-1)).sum() / valid.sum().clamp_min(1)
    loss = class_loss + delay_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert relation_pred.shape[-1] == 3
    assert valid.any()
    return float(loss.detach().cpu())


def main() -> None:
    """运行两个 DAC 视图的短时真实数据 smoke。"""
    from v1_c.model import V1CModel

    design_id = _small_train_id()
    mapped = __import__("dataset.dac2027", fromlist=["load_mapped"]).load_mapped(
        DAC_ROOT, design_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V1CModel(mapped["nodes"].size(-1), mapped["relations"].size(-1),
                     point_width=8, pair_width=6, hidden_width=16, layers=1).to(device)
    mapped_loss = _run_mapped(model, device, design_id)
    model.zero_grad(set_to_none=True)
    timing_loss = _run_timing(model, device, design_id)
    print(f"dac2027_v1c_smoke_pass device={device} design={design_id} "
          f"mapped_loss={mapped_loss:.3e} timing_loss={timing_loss:.3e}")


if __name__ == "__main__":
    main()
