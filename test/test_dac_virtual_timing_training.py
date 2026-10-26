"""Virtual timing loss、mask 和 V1-C 训练步 smoke。"""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))


def main() -> None:
    """验证真实 virtual label 能完成一次带监督反向。"""
    from dataset.dac_virtual_timing_dataset import DACVirtualTimingDataset, collate_virtual_timing
    from v1_c.model import V1CModel
    from v1_c.virtual_timing import _loss, target_stats

    root = Path("/home/yiyh/graphmath/dac2027/processed/verilog_rtl_corpus")
    labels = ROOT / "dataset/dac2027_virtual_timing_v1"
    data = DACVirtualTimingDataset(root, labels, "train", "native_process", 2048)
    assert len(data) > 0
    batch = collate_virtual_timing([data[0]])
    mean, std = target_stats(data, 0, True)
    model = V1CModel(batch["nodes"].size(-1), batch["relations"].size(-1), 8, 6, 16, layers=1)
    prediction = model(batch["nodes"], batch["relations"], batch["node_mask"],
                       virtual_timing_only=True)
    loss = _loss(prediction, batch["relation_target"], batch["relation_valid"], 0, mean, std, True)
    loss.backward()
    assert torch.isfinite(loss) and int(batch["relation_valid"].sum()) > 0
    assert model.encs.mlp[0].weight.grad is not None
    assert model.virtual_timing.readout[0].weight.grad is not None
    assert model.encs.mlp[0].weight.grad.abs().sum() > 0
    assert model.virtual_timing.readout[0].weight.grad.abs().sum() > 0
    print(f"dac_virtual_timing_training_smoke_pass designs={len(data)} valid={int(batch['relation_valid'].sum())}")


if __name__ == "__main__":
    main()
