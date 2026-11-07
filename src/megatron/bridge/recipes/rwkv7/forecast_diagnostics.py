# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OHLCV diagnostics for replicated RWKV7 with native AdamW master weights."""

import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

import torch

from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.bridge.training.callbacks import Callback, CallbackContext


def tensor_statistics(tensor: torch.Tensor) -> torch.Tensor:
    """Return additive squared norm/count and a maximum, retaining FP32 precision."""
    value = tensor.detach().float()
    return torch.stack((value.square().sum(), value.new_tensor(value.numel()), value.abs().max()))


def adamw_update_direction(
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    betas: tuple[float, float],
    eps: float,
    bias_correction: bool,
) -> torch.Tensor:
    """Compute u_t from Adam moments, excluding learning rate and weight decay."""
    beta1, beta2 = betas
    correction1 = 1 - beta1**step if bias_correction else 1.0
    correction2 = 1 - beta2**step if bias_correction else 1.0
    return (first_moment / correction1) / ((second_moment / correction2).sqrt() + eps)


def _statistic_groups(name: str) -> list[str]:
    groups = ["global", name.rsplit(".", 1)[0]]
    if name.startswith("hf.layers."):
        groups.append(".".join(name.split(".")[:3]))
        if name.endswith((".k_a", ".k_k", ".w0")):
            groups.append(name)
    return groups


class ForecastDiagnosticsCallback(Callback):
    """Log unmodified AdamW updates, gradients and activations by module and layer.

    Supports the forecast recipe's TP=PP=CP=1, BF16 model, FP32 master-weight
    AdamW path. DP optimizer shards are summed once; absmax uses MAX. Activation
    statistics cover all window positions, including masked tails. Master-weight
    snapshots avoid treating BF16 weight-copy rounding as an optimizer update.
    """

    def __init__(self, output: Path, *, interval: int, spike_snapshot: bool = False) -> None:
        self.output, self.interval = output, interval
        self.parameters: list[tuple[str, torch.Tensor, dict[str, Any]]] = []
        self.names: list[str] = []
        self.before: list[torch.Tensor] = []
        self.hyperparameters: list[tuple[float, float]] = []
        self.handles: list[Any] = []
        self.activations: dict[str, list[torch.Tensor]] = {}
        self.grad_history: deque[float] = deque(maxlen=100)
        self.loss_history: deque[float] = deque(maxlen=100)
        self.expected_squares: list[torch.Tensor] = []
        self.state_owners: list[Any] = []
        self.capture = False
        self.spike_snapshot = spike_snapshot
        self.snapshot_written = False
        self.model: torch.nn.Module | None = None
        self.before_model: dict[str, torch.Tensor] = {}

    def on_train_start(self, context: CallbackContext) -> None:
        """Resolve local optimizer shards and install observational forward hooks."""
        model = unwrap_model(context.model)[0]
        self.model = model
        self.snapshot_written = (self.output / "spike-before.pt").exists()
        optimizer = context.optimizer
        optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        groups = {id(p): group for opt in optimizers for group in opt.param_groups for p in group["params"]}
        owners = {id(p): opt.optimizer for opt in optimizers for group in opt.param_groups for p in group["params"]}
        names = set()
        for name, parameter in model.named_parameters():
            names.update(_statistic_groups(name))
            master = parameter.main_param
            if master is not None:
                self.parameters.append((name, master, groups[id(master)]))
                self.expected_squares.append(master.detach().square().mean())
                self.state_owners.append(owners[id(master)])
        self.names = sorted(names)
        for name, module in model.named_modules():
            if name in {"input_projection", "output_projection", "hf.norm"} or (
                name.startswith("hf.layers.") and (name.count(".") == 2 or name.endswith(("linear_attn", "mlp")))
            ):
                self.handles.append(module.register_forward_hook(self._activation_hook(name)))

    def _activation_hook(self, name: str) -> Any:
        def hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
            if self.capture and module.training:
                value = output[0] if isinstance(output, tuple) else output
                self.activations.setdefault(name, []).append(tensor_statistics(value))

        return hook

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Snapshot sharded FP32 master weights before any optimizer update."""
        self.capture = True
        self.activations.clear()
        self.before = [p.detach().clone() for _, p, _ in self.parameters]
        self.hyperparameters = [(group["lr"], group["weight_decay"]) for _, _, group in self.parameters]
        if self.spike_snapshot and not self.snapshot_written:
            self.before_model = {name: p.detach().clone() for name, p in self.model.named_parameters()}

    @torch.no_grad()
    def on_train_step_end(self, context: CallbackContext) -> None:
        """Reduce exact shard statistics; record intervals and exceptional steps."""
        self.capture = False
        step = context.state.train_state.step + 1
        grad_norm = float(context.grad_norm)
        loss = float(context.loss_dict["ohlcv huber"])
        grad_ratio = grad_norm / max(statistics.median(self.grad_history), 1e-30) if self.grad_history else 1.0
        loss_ratio = loss / max(statistics.median(self.loss_history), 1e-30) if self.loss_history else 1.0
        spike = len(self.grad_history) == 100 and (grad_ratio > 5 or loss_ratio > 1.2)
        if spike and self.spike_snapshot and not self.snapshot_written:
            if context.state.tensorboard_logger is not None:
                torch.save(
                    {
                        "version": 1,
                        "run": json.loads((self.output / "run.json").read_text()),
                        "step": step - 1,
                        "diagnostic_step": step,
                        "consumed_samples_before_step": context.state.train_state.consumed_train_samples,
                        "state_dict": {name: value.cpu() for name, value in self.before_model.items()},
                    },
                    self.output / "spike-before.pt",
                )
            self.snapshot_written = True
        self.before_model.clear()
        self.grad_history.append(grad_norm)
        self.loss_history.append(loss)
        for i, ((_, p, _), (lr, wd)) in enumerate(zip(self.parameters, self.hyperparameters)):
            # Dynamic mean-field reference from kexue.fm/archives/11404;
            # this is a diagnostic approximation, not a clipping threshold.
            self.expected_squares[i] = (1 - lr * wd) ** 2 * self.expected_squares[i] + lr**2
        if step % self.interval and not spike and step != 1:
            self.before.clear()
            return
        clip = context.state.cfg.optimizer.clip_grad
        coefficient = min(1.0, clip / (grad_norm + 1e-6)) if clip > 0 else 1.0
        device = self.before[0].device
        # Per module: count, parameter SS, raw gradient SS, actual delta SS,
        # Adam direction SS, old parameter SS, mean-field parameter SS.
        sums = torch.zeros((len(self.names), 7), device=device, dtype=torch.float64)
        maxima = torch.zeros((len(self.names), 4), device=device)
        index = {name: i for i, name in enumerate(self.names)}
        for i, ((name, p, group), old) in enumerate(zip(self.parameters, self.before)):
            raw_grad = p.grad.float() / coefficient
            delta = p.detach() - old
            state = self.state_owners[i].state[p]
            update = adamw_update_direction(
                state["exp_avg"],
                state["exp_avg_sq"],
                step=group["step"],
                betas=group["betas"],
                eps=group["eps"],
                bias_correction=group["bias_correction"],
            )
            values = torch.stack(
                (
                    p.new_tensor(p.numel()),
                    p.square().sum(),
                    raw_grad.square().sum(),
                    delta.square().sum(),
                    update.square().sum(),
                    old.square().sum(),
                    self.expected_squares[i] * p.numel(),
                )
            ).double()
            peaks = torch.stack((p.abs().max(), raw_grad.abs().max(), delta.abs().max(), update.abs().max()))
            for key in _statistic_groups(name):
                row = index[key]
                sums[row] += values
                maxima[row] = torch.maximum(maxima[row], peaks)
        distributed = torch.distributed.is_initialized()
        if distributed:
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
            if not context.state.cfg.optimizer.use_distributed_optimizer:
                sums /= torch.distributed.get_world_size()
        metrics = {"grad_spike_ratio": grad_ratio, "loss_spike_ratio": loss_ratio, "clip_coefficient": coefficient}
        for name, row, peaks in zip(self.names, sums.cpu().tolist(), maxima.cpu().tolist()):
            count, param, grad, delta, update, old, expected = row
            for label, square, peak in zip(("param", "grad", "delta", "update"), (param, grad, delta, update), peaks):
                metrics[f"{name}/{label}_rms"] = math.sqrt(square / count)
                metrics[f"{name}/{label}_absmax"] = peak
                metrics[f"{name}/{label}_norm"] = math.sqrt(square)
            metrics[f"{name}/grad_norm_after_clip"] = math.sqrt(grad) * coefficient
            metrics[f"{name}/param_rms_mean_field"] = math.sqrt(expected / count)
            if old > 0:
                metrics[f"{name}/relative_update"] = math.sqrt(delta / old)
        for name, values in self.activations.items():
            values = torch.stack(values)
            total = values[:, :2].sum(0)
            peak = values[:, 2].max()
            if distributed:
                torch.distributed.all_reduce(total)
                torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            metrics[f"{name}/hidden_rms"] = (total[0] / total[1]).sqrt().item()
            metrics[f"{name}/hidden_absmax"] = peak.item()
        writer = context.state.tensorboard_logger
        if writer is not None:
            for name, value in metrics.items():
                writer.add_scalar(f"forecast/{name}", value, step)
            with (self.output / "diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "spike": spike, "loss": loss, **metrics}) + "\n")
        self.before.clear()

    def on_train_end(self, context: CallbackContext) -> None:
        """Release activation hooks after training."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
