"""保存模型、优化器及随机数状态，支持后续复现退化轨迹。"""

import random

import torch


def save_checkpoint(path, model, optimizer, config, step, result):
    """CPU tensor 序列化不运行 CPU 模型；CUDA 实验设备保持不变。"""
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "step": step, "result": result,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, path)
