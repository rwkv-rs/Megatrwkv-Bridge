# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Bridge data and training configuration for next-bar OHLCV prediction."""

from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict

import torch
from transformers import RwkvConfig

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.datasets.ohlcv import OHLCVWindowDataset
from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import RWKV7ForecastConfig
from megatron.bridge.recipes.rwkv7.rwkv7 import rwkv7_pretrain_config
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import validate_local_loss
from megatron.bridge.training.state import GlobalState


class ForecastRun(TypedDict):
    """Training semantics and source identity persisted with task weights."""

    hf_config: dict[str, Any]
    manifest_sha256: str
    steps: int
    global_batch_size: int
    micro_batch_size: int
    learning_rate: float
    warmup_steps: int
    save_interval: int
    eval_interval: int
    eval_iters: int
    seed: int
    world_size: int


class ForecastCheckpoint(TypedDict):
    """Standalone inference artifact; native checkpoints own training resume."""

    version: int
    run: ForecastRun
    step: int
    state_dict: dict[str, torch.Tensor]


@dataclass(kw_only=True)
class OHLCVDatasetConfig(DatasetProvider):
    """Configure immutable local windows through Bridge's dataset interface."""

    root: str
    seq_length: int
    seed: int
    dataloader_type: Literal["single"] = "single"

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> tuple[OHLCVWindowDataset | None, OHLCVWindowDataset | None, OHLCVWindowDataset | None]:
        """Build requested splits with the sample lengths expected by Bridge."""
        datasets: list[OHLCVWindowDataset | None] = []
        for split, count in zip(
            ("train", "validation", "test"),
            (context.train_samples, context.valid_samples, context.test_samples),
        ):
            datasets.append(
                OHLCVWindowDataset(
                    Path(self.root),
                    split=split,
                    seq_length=self.seq_length,
                    target_length=count,
                    seed=self.seed if split == "train" else None,
                )
                if count > 0
                else None
            )
        return datasets[0], datasets[1], datasets[2]


def rwkv7_ohlcv_config(
    hf_config: RwkvConfig,
    *,
    data: Path,
    output: Path,
    steps: int,
    micro_batch_size: int,
    global_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    save_interval: int,
    eval_interval: int,
    eval_iters: int,
    seed: int,
) -> ConfigContainer:
    """Build an early, unoptimized continuous RWKV7 training experiment.

    Throughput has not been measured. Uses teacher forcing with FP32 Huber
    loss over five encoded channels, BF16 computation and FP32 WKV state.
    Native DDP/checkpoint/optimizer/scheduler/data-progress handling is reused.
    """
    cfg = rwkv7_pretrain_config(hf_config)
    base = cfg.model
    cfg.model = RWKV7ForecastConfig(**{field.name: getattr(base, field.name) for field in fields(base) if field.init})
    cfg.dataset = OHLCVDatasetConfig(root=str(data), seq_length=hf_config.context_length, seed=seed)
    cfg.train.train_iters = steps
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.global_batch_size = global_batch_size
    cfg.optimizer.lr = learning_rate
    cfg.optimizer.min_lr = learning_rate * 0.1
    cfg.scheduler.lr_warmup_iters = warmup_steps
    cfg.scheduler.lr_decay_iters = steps
    cfg.validation.eval_interval = eval_interval
    cfg.validation.eval_iters = eval_iters
    cfg.checkpoint.save = str(output / "checkpoints")
    cfg.checkpoint.save_interval = save_interval
    cfg.logger.tensorboard_dir = str(output / "tb_logs")
    cfg.rng.seed = seed
    return cfg


def forecast_loss(
    output: torch.Tensor, *, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return summed per-bar loss and counts for native DP normalization."""
    loss = output.float().sum()
    validate_local_loss(loss, check_for_nan_in_loss=check_for_nan_in_loss, check_for_spiky_loss=check_for_spiky_loss)
    count = torch.tensor(output.numel(), device=output.device, dtype=torch.int)
    return loss, count, {"ohlcv huber": torch.stack((loss.detach(), count.float()))}


def forecast_forward_step(
    state: GlobalState, data_iterator: Iterator[dict[str, torch.Tensor]], model: torch.nn.Module
) -> tuple[torch.Tensor, partial[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]]:
    """Run already aligned continuous windows through the wrapped model."""
    batch = next(data_iterator)
    features = batch["features"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    output = model(features=features, labels=labels)
    return output, partial(
        forecast_loss,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )


class ForecastExportCallback(Callback):
    """Export replicated forecast weights for single-process inference."""

    def __init__(self, output: Path, *, run: ForecastRun) -> None:
        self.output, self.run = output, run

    def on_train_end(self, context: CallbackContext) -> None:
        """Save inference weights after the native training checkpoint lifecycle."""
        if torch.distributed.get_rank() != 0:
            return
        model = unwrap_model(context.model)[0]
        torch.save(
            {
                "version": 1,
                "run": self.run,
                "step": context.state.train_state.step,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            },
            self.output / "forecast.pt",
        )
