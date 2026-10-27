"""V2-EX 独立包的 CUDA、反向、固定点接口和真实图 smoke。"""

import ast
from pathlib import Path

import torch

from v2_ex.task import reachable_loss, reach_metrics

ROOT = Path(__file__).resolve().parents[1]


def local_import_check() -> None:
    """确认 v2_ex 只依赖 torch 和自身相对模块。"""
    forbidden = {"v0", "v1", "v1_c", "v2"}
    for path in (ROOT / "test/v2_ex").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module not in forbidden, (path, node.module)


def check_outputs(model, nodes, relations, mask, noises) -> None:
    """检查最终监督、共享动力学、噪声退出和梯度。"""
    output, trace, points = model(nodes, relations, mask, noises, True)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    target = torch.ones(2, 5, 5, dtype=torch.bool, device=nodes.device)
    loss = reachable_loss(output["reachable"], target, pair_mask)
    loss.backward()
    assert len(trace) == model.steps
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.equal(points[1, 4:], torch.zeros_like(points[1, 4:]))
    assert torch.isfinite(loss)
    scales = torch.stack([item["noise_scale"] for item in trace])
    assert torch.all(scales[1:] <= scales[:-1] + 1e-7)
    assert sum(parameter.grad is not None for parameter in model.parameters()) > 0
    print({"random_loss": loss.item(), "metrics": reach_metrics(
        output["reachable"], target, pair_mask)})


def real_graph_check(model, device) -> None:
    """用 ADC_CTRL 的五类关系执行最终 reachable-only 反向。"""
    path = ROOT / ("dataset/v2_dense_multi_hop_directed/train/"
                   "open_verilog__ADC_CTRL__a6a73eece6e776c5_0a4065ef.pt")
    raw = torch.load(path, map_location=device, weights_only=False)
    nodes = raw["node_features"][None].to(device)
    relations = raw["relation_vectors"][None].to(device)
    target = raw["reachable"][None].to(device)
    mask = torch.ones(1, nodes.size(1), dtype=torch.bool, device=device)
    output, trace, _ = model(nodes, relations, mask, return_trace=True)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    loss = reachable_loss(output["reachable"], target, pair_mask)
    loss.backward()
    assert relations.size(-1) == 5
    assert torch.isfinite(loss)
    assert len(trace) == model.steps
    print(f"real_v2_ex_smoke nodes={nodes.size(1)} loss={loss.item():.6f}")


def permutation_check(model, nodes, relations, mask):
    """同步排列输入和噪声，验证最终点集和有序关系输出等变。"""
    permutation = torch.tensor([2, 0, 4, 1, 3], device=nodes.device)
    noises = tuple(torch.randn(1, 5, 32, device=nodes.device) for _ in range(model.steps))
    with torch.no_grad():
        original = model(nodes[:1], relations[:1], mask[:1], noises, True)
        perm_nodes = nodes[:1, permutation]
        perm_relations = relations[:1, permutation][:, :, permutation]
        shuffled = model(perm_nodes, perm_relations, mask[:1],
                         tuple(noise[:, permutation] for noise in noises), True)
    torch.testing.assert_close(original[2][:, permutation], shuffled[2], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(original[0]["reachable"][:, permutation][:, :, permutation],
                               shuffled[0]["reachable"], atol=3e-5, rtol=3e-5)


def main() -> None:
    """执行 V2-EX 隔离验证。"""
    local_import_check()
    from v2_ex import V2ExModel

    torch.manual_seed(20261027)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V2ExModel(steps=4, step_size=0.05).to(device)
    nodes = torch.randn(2, 5, 4, device=device)
    relations = torch.randn(2, 5, 5, 5, device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
                        dtype=torch.bool, device=device)
    noises = tuple(torch.randn(2, 5, 32, device=device) for _ in range(model.steps))
    check_outputs(model, nodes, relations, mask, noises)
    permutation_check(model, nodes, relations, mask)
    real_graph_check(V2ExModel(steps=4, step_size=0.05).to(device), device)
    print(f"v2_ex_isolation_smoke_pass device={device}")


if __name__ == "__main__":
    main()
