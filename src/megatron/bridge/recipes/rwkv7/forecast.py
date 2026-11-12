# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Bridge data and training configuration for next-bar OHLCV prediction."""

import json
import logging
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Literal, NotRequired, TypedDict

import torch
from megatron.core.optimizer import ParamKey, ParamWithNamePredicate
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from transformers import RwkvConfig

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset, OHLCVWindowDataset
from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import RWKV7ForecastConfig, forecast_pinball
from megatron.bridge.recipes.rwkv7.rwkv7 import rwkv7_pretrain_config
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    ConfigContainer,
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)
from megatron.bridge.training.losses import validate_local_loss
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)


@dataclass
class RWKV7ForecastOptimizerOverrides(OptimizerConfigOverrideProvider):
    """Match RWKV-LM's matrix-only decay and doubled decay-logit learning rate."""

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride]:
        """Exclude TimeMix vectors and low-rank factors from weight decay."""
        overrides = super().build_config_overrides(context) or {}
        no_decay = ParamWithNamePredicate(
            name="rwkv7_forecast_non_matrix_weight",
            fn=lambda parameter, name: not (name.endswith(".weight") and parameter.squeeze().ndim >= 2),
        )
        overrides[ParamKey(with_name_predicate=no_decay)] = ParamGroupOverride(wd_mult=0.0)
        overrides[ParamKey(name="*.linear_attn.w0")] = ParamGroupOverride(
            lr_mult=2.0,
            max_lr=2 * context.optimizer_config.lr,
            min_lr=2 * context.optimizer_config.min_lr,
        )
        return overrides


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
    forecast_references: NotRequired[dict[str, str]]
    sample_budget: NotRequired[dict[str, int]]
    covariates: NotRequired[str]
    covariates_sha256: NotRequired[str]
    validation_indices: NotRequired[str]
    validation_indices_sha256: NotRequired[str]
    pvw_cache: NotRequired[str]
    pvw_cache_sha256: NotRequired[str]
    data_workers: NotRequired[int]


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
    forecast_task: str = "next_bar"
    forecast_horizon: int = 1
    covariates: str | None = None
    validation_indices: str | None = None
    pvw_cache: str | None = None
    feature_columns: list[int] | None = None

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> tuple[OHLCVWindowDataset | None, OHLCVWindowDataset | None, OHLCVWindowDataset | None]:
        """Build requested splits with the sample lengths expected by Bridge."""
        datasets: list[OHLCVWindowDataset | None] = []
        direct = self.forecast_task == "direct_quantile"
        dataset_class = (
            partial(
                OHLCVQuantileWindowDataset,
                pvw_cache=Path(self.pvw_cache) if self.pvw_cache else None,
                feature_columns=self.feature_columns,
            )
            if direct
            else OHLCVWindowDataset
        )
        for split, count in zip(
            ("train", "validation", "test"),
            (context.train_samples, context.valid_samples, context.test_samples),
        ):
            if split == "test" and self.forecast_task == "direct_quantile":
                # Test evaluation is explicit, after validation-based model selection.
                datasets.append(None)
                continue
            datasets.append(
                dataset_class(
                    Path(self.root),
                    horizon=self.forecast_horizon,
                    split=split,
                    seq_length=self.seq_length,
                    target_length=count,
                    seed=self.seed if split == "train" else self.seed + 1,
                    stride=self.seq_length if self.epochs is not None and not direct else 1,
                    keep_tail=self.epochs is not None and not direct,
                    epochs=self.epochs if split == "train" else None,
                    cycle_length=self.evaluation_samples if split != "train" else None,
                    covariates=Path(self.covariates) if self.covariates else None,
                    window_indices=Path(self.validation_indices)
                    if split == "validation" and self.validation_indices
                    else None,
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
    covariates: Path | None = None,
    validation_indices: Path | None = None,
    pvw_cache: Path | None = None,
    data_workers: int | None = None,
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
        forecast_task=getattr(hf_config, "forecast_task", "next_bar"),
        forecast_horizon=getattr(hf_config, "forecast_horizon", 1),
        evaluation_samples=global_batch_size * eval_iters if eval_iters else None,
        num_workers=data_workers if data_workers is not None else (4 if covariates else 2),
        covariates=str(covariates) if covariates else None,
        validation_indices=str(validation_indices) if validation_indices else None,
        pvw_cache=str(pvw_cache) if pvw_cache else None,
        feature_columns=getattr(hf_config, "forecast_feature_columns", None),
    )
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False
    cfg.train.train_iters = steps
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.global_batch_size = global_batch_size
    cfg.optimizer.lr = learning_rate
    cfg.optimizer.min_lr = learning_rate * 0.1
    if getattr(hf_config, "forecast_optimizer", "megatron") == "rwkv_lm":
        cfg.optimizer_config_override_provider = RWKV7ForecastOptimizerOverrides()
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
    quantile_levels: tuple[float, ...] = (),
    price_error: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return summed per-bar loss and counts for native DP normalization."""
    mask = torch.ones_like(output) if loss_mask is None else loss_mask
    loss = (output.float() * mask).sum()
    validate_local_loss(loss, check_for_nan_in_loss=check_for_nan_in_loss, check_for_spiky_loss=check_for_spiky_loss)
    count = mask.sum().detach().to(torch.int)
    loss_name = "forecast pinball" if quantile_levels else "ohlcv huber"
    report = {loss_name: torch.stack((loss.detach(), count.float()))}
    if quantile_levels:
        assert channel_losses is not None
        for i, level in enumerate(quantile_levels):
            report[f"forecast/q{level:.1f}"] = torch.stack(((channel_losses[..., i] * mask).sum(), count.float()))
        for horizon in (1, 5, 15, 30):
            if horizon <= output.shape[1]:
                report[f"forecast/h{horizon}"] = torch.stack(
                    ((output[:, horizon - 1] * mask[:, horizon - 1]).sum().detach(), mask[:, horizon - 1].sum())
                )
    elif channel_losses is not None:
        for i, name in enumerate(("open", "close", "upper_wick", "lower_wick", "volume")):
            report[f"ohlcv/{name}"] = torch.stack(((channel_losses[..., i] * mask).sum(), count.float()))
    if price_error is not None:
        report["forecast/median_price_mae"] = torch.stack(((price_error * mask).sum(), count.float()))
    return loss, count, report


def forecast_forward_step(
    state: GlobalState, data_iterator: Iterator[dict[str, torch.Tensor]], model: torch.nn.Module
) -> tuple[torch.Tensor, partial[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]]:
    """Run already aligned continuous windows through the wrapped model."""
    batch = next(data_iterator)
    features = batch["features"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    levels = tuple(state.cfg.model.hf_config.get("forecast_quantiles", ()))
    price_error = None
    if levels and not model.training:
        # Select models on return-unit scores, independent of training-loss scaling.
        predictions = model(features=features)
        output = forecast_pinball(predictions, labels, levels)
        median = predictions.sort(dim=-1).values[..., levels.index(0.5)]
        price_error = 100 * (torch.exp(median / 100) - torch.exp(labels / 100)).abs()
    else:
        output = model(features=features, labels=labels, return_channel_losses=True)
    return output.mean(-1), partial(
        forecast_loss,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        loss_mask=batch["loss_mask"].cuda(non_blocking=True),
        channel_losses=output.detach(),
        quantile_levels=levels,
        price_error=price_error,
    )


class ForecastExportCallback(Callback):
    """Export replicated forecast weights for single-process inference."""

    def __init__(self, output: Path, *, run: ForecastRun) -> None:
        self.output, self.run = output, run
        self.best_loss = float("inf")
        self.best_median_loss = float("inf")
        selection = output / "best-validation.json"
        if selection.exists():
            self.best_loss = json.loads(selection.read_text())["pinball"]
        median_selection = output / "best-median-validation.json"
        if median_selection.exists():
            record = json.loads(median_selection.read_text())
            self.best_median_loss = record.get("median_price_mae", record["median_return_mae"])

    def on_eval_end(self, context: CallbackContext) -> None:
        """Export the earliest minimum on the fixed validation windows."""
        if self.run["hf_config"].get("forecast_task") != "direct_quantile":
            return
        assert context.total_loss_dict is not None
        score = float(context.total_loss_dict["forecast pinball"])
        median_score = 2 * float(context.total_loss_dict["forecast/q0.5"])
        record = {"step": context.state.train_state.step, "pinball": score, "median_return_mae": median_score}
        if "forecast/median_price_mae" in context.total_loss_dict:
            median_score = float(context.total_loss_dict["forecast/median_price_mae"])
            record["median_price_mae"] = median_score
        if torch.distributed.get_rank() == 0:
            with (self.output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if score < self.best_loss:
            self.best_loss = score
            if torch.distributed.get_rank() == 0:
                self._export(context, "best-forecast.pt")
                (self.output / "best-validation.json").write_text(json.dumps(record, indent=2) + "\n")
        if median_score < self.best_median_loss:
            self.best_median_loss = median_score
            if torch.distributed.get_rank() == 0:
                self._export(context, "best-median-forecast.pt")
                (self.output / "best-median-validation.json").write_text(json.dumps(record, indent=2) + "\n")

    def _export(self, context: CallbackContext, name: str) -> None:
        model = unwrap_model(context.model)[0]
        temporary = self.output / f"{name}.pending"
        torch.save(
            {
                "version": 1,
                "run": self.run,
                "step": context.state.train_state.step,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            },
            temporary,
        )
        temporary.replace(self.output / name)

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
        self._export(context, "forecast.pt")
