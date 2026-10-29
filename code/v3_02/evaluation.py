"""固定种子、原生非零噪声的训练集评估与最佳权重监控。"""

import json

import torch

from .checkpoint import save_checkpoint
from .data import make_batch
from .hop_task import format_mape, hop_loss, hop_metrics, reachable_mask


def fixed_noises(model, nodes, seed):
    """局部生成器保证复现且不推进训练 RNG；跨深度共用噪声前缀。"""
    rng = torch.Generator(device=nodes.device).manual_seed(seed)
    shape = (*nodes.shape[:2], model.state_width)
    return tuple(torch.randn(shape, device=nodes.device, generator=rng)
                 for _ in range(model.steps))


@torch.inference_mode()
def evaluate(model, graphs, sentinel, seed, batch_size=32):
    """完整训练集按有效 pair 汇总，兼容变长图并恢复模型模式。"""
    device = next(model.parameters()).device
    mode, predictions, targets = model.training, [], []
    model.eval()
    try:
        for start in range(0, len(graphs), batch_size):
            x, r, y, mask, pairs = make_batch(graphs[start:start + batch_size], device)
            p = model(x, r, mask, fixed_noises(model, x, seed + start))["min_hop"].squeeze(-1)
            valid = reachable_mask(y, pairs, sentinel)
            predictions.append(p[valid])
            targets.append(y[valid])
        prediction, target = torch.cat(predictions), torch.cat(targets)
        mask = torch.ones_like(target, dtype=torch.bool)
        return {"loss": hop_loss(prediction, target, mask, sentinel).item(),
                "metrics": hop_metrics(prediction, target, mask, sentinel)}
    finally:
        model.train(mode)


@torch.inference_mode()
def probe_trace(model, graphs, seed):
    """仅对固定探针图汇总每步指标；记录样本量以区别完整训练指标。"""
    x, r, _, mask, _ = make_batch(graphs, next(model.parameters()).device)
    output, trace = model(x, r, mask, fixed_noises(model, x, seed), return_trace=True)
    prediction = output["min_hop"].squeeze(-1)
    valid = mask[:, :, None] & mask[:, None, :]
    routed = prediction.masked_select(valid)
    readout = {"rout_output_rms": routed.square().mean().sqrt().item(),
               "rout_output_abs_mean": routed.abs().mean().item(),
               "rout_output_min": routed.min().item(), "rout_output_max": routed.max().item()}
    return {"graphs": len(graphs), "seed": seed, "readout": readout,
            "steps": [{k: v.item() for k, v in row.items()} for row in trace]}


@torch.inference_mode()
def trajectory_diagnostics(model, graphs, sentinel, seed, batch_size=32):
    """逐层解码 rout，比较状态运动与关系读出的稳定性。"""
    device, batches = next(model.parameters()).device, []
    for start in range(0, len(graphs), batch_size):
        x, r, y, mask, pairs = make_batch(graphs[start:start + batch_size], device)
        final, trace, states = model.embed(x, r, mask, fixed_noises(model, x, seed + start),
                                           True, True)
        predictions = [model.decode(state, mask)["relations"].squeeze(-1)
                       for state in states]
        batches.append((y, mask, pairs, predictions, states, trace))
    rows = []
    for step in range(model.steps):
        values = {"mape": [], "rout_delta_rms": [], "state_delta_rms": [],
                  "rout_state_sensitivity": [], "prediction_final_corr": [],
                  "velocity_cosine": [], "pair_distance_delta_rms": []}
        for y, mask, pairs, predictions, states, trace in batches:
            valid = reachable_mask(y, pairs, sentinel)
            current = predictions[step + 1]
            values["mape"].append(hop_metrics(current, y, pairs, sentinel)["mape_percent"])
            previous, current_state = states[step], states[step + 1]
            state_delta = current_state - previous
            rout_delta = current - predictions[step]
            if step == 0:
                velocity_cosine = torch.tensor(0., device=device)
            else:
                old_velocity = states[step] - states[step - 1]
                velocity_cosine = (state_delta * old_velocity).sum(-1)
                velocity_cosine = velocity_cosine / (state_delta.norm(dim=-1) *
                    old_velocity.norm(dim=-1)).clamp_min(1e-12)
                velocity_cosine = velocity_cosine.masked_select(mask).mean()
            state_rms = state_delta.square().sum(-1).masked_select(mask).mean().sqrt()
            rout_rms = rout_delta[valid].square().mean().sqrt()
            values["rout_delta_rms"].append(rout_rms)
            values["state_delta_rms"].append(state_rms)
            values["rout_state_sensitivity"].append(rout_rms / state_rms.clamp_min(1e-12))
            values["velocity_cosine"].append(velocity_cosine)
            final_prediction = predictions[-1][valid]
            current_prediction = current[valid]
            a, b = current_prediction - current_prediction.mean(), final_prediction - final_prediction.mean()
            values["prediction_final_corr"].append((a * b).mean() / (a.square().mean().sqrt() * b.square().mean().sqrt()).clamp_min(1e-12))
            pair_state = current_state[:, :, None] - current_state[:, None, :]
            pair_previous = previous[:, :, None] - previous[:, None, :]
            pair_delta = (pair_state.norm(dim=-1) - pair_previous.norm(dim=-1)).masked_select(pairs)
            values["pair_distance_delta_rms"].append(pair_delta.square().mean().sqrt())
        rows.append({"step": step + 1,
                     **{key: float(torch.stack([torch.as_tensor(item, device=device)
                                                 for item in value]).mean())
                         for key, value in values.items()}})
    return rows


class TrainMonitor:
    """以最低完整训练 loss 选权重，保留历史和真实优化状态。"""

    def __init__(self, model, optimizer, graphs, sentinel, config, output):
        self.model, self.optimizer, self.graphs = model, optimizer, graphs
        self.sentinel, self.config, self.output = sentinel, config, output
        self.best_loss, self.best_step, self.history = float("inf"), None, []

    def log(self, step, loss, prediction, target, pairs, grad_norm):
        """batch 输出在优化前，完整训练评估在当前优化步之后。"""
        batch = hop_metrics(prediction.detach(), target, pairs, self.sentinel)
        full = evaluate(self.model, self.graphs, self.sentinel, self.config["eval_seed"])
        if not torch.isfinite(torch.tensor(full["loss"])):
            raise FloatingPointError("non-finite training evaluation loss")
        row = {"step": step, "train_batch_loss": loss.item(),
               "train_batch_mape_percent": batch["mape_percent"],
               "train_batch_mae": batch["mae"], "train_batch_zero_hop_mae": batch["zero_hop_mae"],
               "train_batch_accuracy": batch["rounded_exact_accuracy"],
               "pre_clip_grad_norm": float(grad_norm), "train_full": full}
        self.history.append(row)
        if full["loss"] < self.best_loss:
            self.best_loss, self.best_step = full["loss"], step
            save_checkpoint(self.output / "best_checkpoint.pt", self.model,
                            self.optimizer, self.config, step, row)
        (self.output / "history.json").write_text(json.dumps(self.history, indent=2) + "\n")
        print(f"step={step} train_batch_loss={loss.item():.6f} "
              f"train_batch_mape={format_mape(batch['mape_percent'])} "
              f"train_full_loss={full['loss']:.6f} "
              f"train_full_mape={format_mape(full['metrics']['mape_percent'])} "
              f"train_full_mae={full['metrics']['mae']:.4f} "
              f"zero_hop_mae={full['metrics']['zero_hop_mae']} "
              f"best_step={self.best_step}", flush=True)
