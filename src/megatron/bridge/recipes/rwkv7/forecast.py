# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Bridge data and training configuration for next-bar OHLCV prediction."""

import json
import logging
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


logger = logging.getLogger(__name__)


class ForecastRun(TypedDict):
    """Training semantics and source identity persisted with task weights."""

    hf_config: dict[str, Any]
    manifest_sha256: str
    steps: int
    epochs: int | None
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
    epochs: int | None = None
    evaluation_samples: int | None = None

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
                    seed=self.seed if split == "train" else self.seed + 1,
                    stride=self.seq_length if self.epochs is not None else 1,
                    keep_tail=self.epochs is not None,
                    epochs=self.epochs if split == "train" else None,
                    cycle_length=self.evaluation_samples if split != "train" else None,
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
    epochs: int | None = None,
) -> ConfigContainer:
    """Build continuous RWKV7 training with fixed, distributed holdout windows.

    Uses teacher forcing with FP32 Huber
    loss over five encoded channels, BF16 computation and FP32 WKV state.
    Native DDP/checkpoint/optimizer/scheduler/data-progress handling is reused.
    """
    cfg = rwkv7_pretrain_config(hf_config)
    base = cfg.model
    cfg.model = RWKV7ForecastConfig(**{field.name: getattr(base, field.name) for field in fields(base) if field.init})
    cfg.dataset = OHLCVDatasetConfig(
        root=str(data),
        seq_length=hf_config.context_length,
        seed=seed,
        epochs=epochs,
        evaluation_samples=global_batch_size * eval_iters if eval_iters else None,
    )
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False
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
    output: torch.Tensor,
    *,
    check_for_nan_in_loss: bool,
    check_for_spiky_loss: bool,
    loss_mask: torch.Tensor | None = None,
    channel_losses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return summed per-bar loss and counts for native DP normalization."""
    mask = torch.ones_like(output) if loss_mask is None else loss_mask
    loss = (output.float() * mask).sum()
    validate_local_loss(loss, check_for_nan_in_loss=check_for_nan_in_loss, check_for_spiky_loss=check_for_spiky_loss)
    count = mask.sum().detach().to(torch.int)
    report = {"ohlcv huber": torch.stack((loss.detach(), count.float()))}
    if channel_losses is not None:
        for i, name in enumerate(("open", "close", "upper_wick", "lower_wick", "volume")):
            report[f"ohlcv/{name}"] = torch.stack(((channel_losses[..., i] * mask).sum(), count.float()))
    return loss, count, report


def forecast_forward_step(
    state: GlobalState, data_iterator: Iterator[dict[str, torch.Tensor]], model: torch.nn.Module
) -> tuple[torch.Tensor, partial[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]]:
    """Run already aligned continuous windows through the wrapped model."""
    batch = next(data_iterator)
    features = batch["features"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    output = model(features=features, labels=labels, return_channel_losses=True)
    return output.mean(-1), partial(
        forecast_loss,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        loss_mask=batch["loss_mask"].cuda(non_blocking=True),
        channel_losses=output.detach(),
    )


class ForecastExportCallback(Callback):
    """Export replicated forecast weights for single-process inference."""

    def __init__(self, output: Path, *, run: ForecastRun) -> None:
        self.output, self.run = output, run

    def on_train_end(self, context: CallbackContext) -> None:
        """Save inference weights after the native training checkpoint lifecycle."""
        memory = torch.tensor(
            [torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30],
            device="cuda",
            dtype=torch.float64,
        )
        all_memory = [torch.empty_like(memory) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(all_memory, memory)
        if torch.distributed.get_rank() != 0:
            return
        peaks = [
            {"rank": rank, "allocated_gib": values[0], "reserved_gib": values[1]}
            for rank, tensor in enumerate(all_memory)
            for values in [tensor.cpu().tolist()]
        ]
        (self.output / "runtime.json").write_text(
            json.dumps(
                {
                    "step": context.state.train_state.step,
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "peak_memory": peaks,
                },
                indent=2,
            )
            + "\n"
        )
        logger.info("Forecast peak memory across ranks (GiB): %s", peaks)
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
