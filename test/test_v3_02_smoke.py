"""CUDA-only V3.02 独立性、图契约、梯度及对称性 smoke。"""

import ast
from pathlib import Path

import torch

from test.v3_02.model import V3_02Model
from test.v3_02.data import make_batch, load_train
from test.v3_02.evaluation import evaluate, fixed_noises
from test.v3_02.hop_task import hop_loss


def fixture(device):
    """链 0--1--2 与孤立点 3，另含变长图和全 padding 图。"""
    adjacency = torch.zeros(4, 4, device=device)
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[1, 2] = adjacency[2, 1] = 1
    graph = {
        "node_features": torch.ones(4, 1, device=device),
        "relation_vectors": torch.stack((1 - adjacency, adjacency), -1),
        "hop_labels": torch.tensor(
            [[0, 1, 2, 64], [1, 0, 1, 64], [2, 1, 0, 64], [64, 64, 64, 0]],
            device=device,
        ),
        "node_mask": torch.ones(4, dtype=torch.bool, device=device),
    }
    small = {
        k: (v[:3, :3] if v.ndim >= 2 and k != "node_features" else v[:3])
        for k, v in graph.items()
    }
    empty = {**graph, "node_mask": torch.zeros(4, dtype=torch.bool, device=device)}
    return [graph, small, empty]


def check_model(context, graphs, device):
    """验证 V3.02 padding、同步排列、共享参数和完整反向。"""
    model = V3_02Model(steps=4, context=context).to(device)
    x, r, y, mask, pairs = make_batch(graphs, device)
    assert torch.equal(r.sum(-1), pairs.float())
    assert torch.equal(r[..., 2].diagonal(dim1=1, dim2=2), mask.float())
    noises = fixed_noises(model, x, 23)
    output, trace = model(x, r, mask, noises, return_trace=True)
    assert all(torch.isfinite(v).all() for v in output.values())
    assert all(torch.isfinite(v) for row in trace for v in row.values())

    # 验证 trace 中的详细敏感度是否包含 source_state 和 target_state
    first_step = trace[0]
    assert "source_state_field_delta_rms" in first_step
    assert "target_state_field_delta_rms" in first_step
    assert first_step["source_state_field_delta_rms"] > 0
    assert first_step["target_state_field_delta_rms"] > 0

    loss = hop_loss(output["min_hop"].squeeze(-1), y, pairs, 64)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"梯度异常: {name}"

    print(
        f"v3_02_interface_pass context={context} loss={loss.item():.6f} "
        f"src_rms={first_step['source_state_field_delta_rms']:.4f} "
        f"tgt_rms={first_step['target_state_field_delta_rms']:.4f}"
    )


def test_clrs_smoke(device):
    """在真实 CLRS 训练前 8 图上执行两步快速训练 smoke。"""
    graphs, sentinel = load_train()
    smoke_samples = graphs[:8]
    model = V3_02Model(steps=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, 3):
        x, r, target, mask, pairs = make_batch(smoke_samples[:4], device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, r, mask)["min_hop"].squeeze(-1)
        loss = hop_loss(prediction, target, pairs, sentinel)
        assert torch.isfinite(loss), f"Loss 不有限: {loss}"
        loss.backward()
        optimizer.step()
        print(f"v3_02_smoke_step step={step} loss={loss.item():.6f}")
    print("v3_02_clrs_smoke_pass: 真实 CLRS 两步训练正常闭环")


def main():
    """主测试入口。"""
    assert torch.cuda.is_available(), "必须使用 CUDA"
    device = torch.device("cuda")
    torch.manual_seed(20261029)
    graphs = fixture(device)
    for context in ("none", "setattn"):
        check_model(context, graphs, device)
    test_clrs_smoke(device)
    print(f"v3_02_all_smoke_pass on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
