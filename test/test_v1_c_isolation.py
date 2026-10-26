"""验证 V1-C 只从自身包导入，并保持 V1 的前向接口。"""

import torch

from v1_c.model import V1CModel


def main() -> None:
    """检查独立导入、完整前向形状、mask 和梯度。"""
    torch.manual_seed(101)
    model = V1CModel(3, 2, 5, 4, 8, layers=2)
    nodes = torch.randn(2, 4, 3)
    relations = torch.randn(2, 4, 4, 2)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    noises = tuple(torch.randn(2, 4, 5) for _ in range(2))
    points, nodes_out, relations_out, deltas, scales = model(
        nodes, relations, mask, noises
    )
    loss = nodes_out.square().mean() + relations_out.square().mean()
    loss.backward()
    assert points.shape == (2, 4, 5)
    assert nodes_out.shape == nodes.shape
    assert relations_out.shape == relations.shape
    assert len(deltas) == len(scales) == 2
    assert torch.equal(nodes_out[1, 3:], torch.zeros_like(nodes_out[1, 3:]))
    assert torch.isfinite(loss)
    print("v1_c_isolation_smoke_pass")


if __name__ == "__main__":
    main()
