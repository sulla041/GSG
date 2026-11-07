"""独立 V3 CUDA 跳数训练入口，仅评估训练集。"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .data import DATA_ROOT, RELATION_TYPES, load_train, make_batch
from .evaluation import TrainMonitor, evaluate, probe_trace
from .hop_task import format_mape, hop_loss, positive_hop_mape
from .model import V3_02Model


def arguments():
    """登记首版实验选择，避免将聚合或 H_i 选项隐含为最终设计。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dynamics-steps", type=int, default=32)
    parser.add_argument("--state-width", type=int, default=32)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--context", choices=("none", "setattn"), default="none")
    parser.add_argument("--step-size", type=float, default=.05)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--eval-seed", type=int, default=20261029)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.train_steps, args.batch_size, args.dynamics_steps, args.eval_every,
           args.state_width, args.hidden_width) < 1:
        parser.error("step counts, batch size and widths must be positive")
    if min(args.learning_rate, args.grad_clip) <= 0 or args.weight_decay < 0:
        parser.error("invalid optimizer settings")
    return args


def main():
    """CUDA smoke 使用真实训练集前八图、两次更新，不加载 val/test。"""
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("output directory must be empty to preserve prior experiments")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    graphs, sentinel = load_train(args.data_root)
    if args.smoke:
        graphs, args.train_steps, args.eval_every = graphs[:8], 2, 1
    model_config = {"node_width": 1, "relation_width": 3, "state_width": args.state_width,
        "hidden_width": args.hidden_width, "steps": args.dynamics_steps,
        "context": args.context,
        "noise_temperature": 1., "noise_max": .01, "noise_stop": 1e-6}
    model = V3_02Model(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    config = {**vars(args), "output": str(args.output), "data_root": str(args.data_root),
        "model": model_config, "node_types": 1, "relation_types": RELATION_TYPES,
        "graph_count": len(graphs), "evaluation_split": "train", "sentinel": sentinel,
        "loss": "reachable_pair_smooth_l1_beta1", "aggregation": "direct_sum",
        "primary_metric": "mape_percent", "mape_mask": "valid & finite_reachable & hop_gt_0",
        "zero_hop_policy": "included_in_loss_and_mae; reported_separately_from_mape",
        "basis": ["source_state", "target_state", "difference", "product", "directed_projection", "l2_distance"],
        "noise": "adaptive_monotone_seeded_evaluation", "device": str(device),
        "gpu": torch.cuda.get_device_name(), "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": "AdamW", "sampling": "uniform_with_replacement",
        "selection": "minimum_full_train_loss_at_evaluation_steps", "schema_version": "v3_02_smoke_v1"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    monitor = TrainMonitor(model, optimizer, graphs, sentinel, config, args.output)
    started, next_log = time.monotonic(), time.monotonic() + 5
    print(f"v3_start device={device} gpu={config['gpu']} graphs={len(graphs)} "
          f"context={args.context} self_relation=true", flush=True)
    model.train()
    for step in range(1, args.train_steps + 1):
        selected = random.choices(graphs, k=min(args.batch_size, len(graphs)))
        x, r, target, mask, pairs = make_batch(selected, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, r, mask)["min_hop"].squeeze(-1)
        loss = hop_loss(prediction, target, pairs, sentinel)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                              error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step == args.train_steps or step % args.eval_every == 0:
            monitor.log(step, loss, prediction, target, pairs, grad)
            next_log = time.monotonic() + 5
        elif time.monotonic() >= next_log:
            mape = positive_hop_mape(prediction.detach(), target, pairs, sentinel)
            print(f"step={step} batch_loss={loss.item():.6f} batch_mape={format_mape(mape)}", flush=True)
            next_log = time.monotonic() + 5
    save_checkpoint(args.output / "checkpoint.pt", model, optimizer, config,
                    args.train_steps, monitor.history[-1])
    best = torch.load(args.output / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    model.eval()
    best_eval = evaluate(model, graphs, sentinel, args.eval_seed)
    result = {"best_step": best["step"], "best_training": best_eval,
              "final_training": monitor.history[-1],
              "best_probe": probe_trace(model, graphs[:8], args.eval_seed)}
    torch.cuda.synchronize()
    result["elapsed_seconds"] = time.monotonic() - started
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"v3_complete best_step={best['step']} train_loss={best_eval['loss']:.6f} "
          f"train_mape={format_mape(best_eval['metrics']['mape_percent'])} "
          f"train_mae={best_eval['metrics']['mae']:.4f} "
          f"results={args.output / 'results.json'}", flush=True)
