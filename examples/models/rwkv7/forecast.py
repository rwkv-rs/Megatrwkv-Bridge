# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare local A-share bars, train RWKV7, and autoregressively forecast OHLCV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import RWKV7ForecastModel
    from megatron.bridge.recipes.rwkv7.forecast import ForecastCheckpoint


logger = logging.getLogger(__name__)


def prepare(args: argparse.Namespace) -> None:
    """Convert a local canonical snapshot into immutable chronological splits."""
    from megatron.bridge.data.datasets.ohlcv import prepare_ohlcv

    manifest = prepare_ohlcv(
        args.source,
        args.output,
        train_end=args.train_end,
        validation_end=args.validation_end,
        revision=args.revision,
        workers=args.workers,
    )
    for split in ("train", "validation", "test"):
        counts = manifest["splits"][split]
        logger.info("%s: %d segments, %d encoded bars", split, counts["segments"], counts["bars"])


def train(args: argparse.Namespace) -> None:
    """Train with Bridge's optimizer, scheduler, DDP and distributed checkpoints."""
    from transformers import RwkvConfig

    from megatron.bridge.data.datasets.ohlcv import OHLCVWindowDataset, manifest_digest
    from megatron.bridge.recipes.rwkv7.forecast import (
        ForecastExportCallback,
        ForecastRun,
        forecast_forward_step,
        rwkv7_ohlcv_config,
    )
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists

    hf = RwkvConfig.from_json_file(args.config)
    hf.context_length = args.context
    # Forecasting starts from scratch and discards the language embedding/head.
    hf.vocab_size = 1
    hf.tie_word_embeddings = False
    if args.calibration_windows:
        import numpy as np

        calibration = OHLCVWindowDataset(
            args.data,
            split="train",
            seq_length=args.context,
            stride=args.context,
            keep_tail=True,
            seed=args.seed,
        )
        total, count = np.zeros(5, dtype=np.float64), 0
        for i in range(args.calibration_windows):
            sample = calibration[i * calibration.num_windows // args.calibration_windows]
            mask = sample["loss_mask"].astype(bool)
            total += sample["labels"][mask].sum(axis=0, dtype=np.float64)
            count += int(mask.sum())
        hf.forecast_initialization = "calibrated"
        hf.forecast_target_mean = (total / count).tolist()
        hf.forecast_head_gain = args.head_gain
        logger.info("Training-only head calibration (%d bars): %s", count, hf.forecast_target_mean)
    if args.head_fp32:
        hf.forecast_head_fp32 = True
    if args.rwkv_optimizer:
        hf.forecast_optimizer = "rwkv_lm"
    # Count actual windows before resolving an epoch-based optimizer-step budget.
    for split in ("train", "validation", "test") if args.eval_iters else ("train",):
        dataset = OHLCVWindowDataset(
            args.data,
            split=split,
            seq_length=args.context,
            stride=args.context if args.epochs is not None else 1,
            keep_tail=args.epochs is not None,
        )
        logger.info("%s: %d available windows", split, dataset.num_windows)
        if split == "train" and args.epochs is not None:
            args.steps = math.ceil(dataset.num_windows * args.epochs / args.global_batch_size)
            logger.info("%d epoch(s) resolve to %d optimizer steps", args.epochs, args.steps)
    if args.warmup_steps >= args.steps:
        raise ValueError("warmup_steps must be less than the resolved training steps.")
    run: ForecastRun = {
        "hf_config": json.loads(hf.to_json_string()),
        "manifest_sha256": manifest_digest(args.data),
        "steps": args.steps,
        "epochs": args.epochs,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "save_interval": args.save_interval,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "seed": args.seed,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    if args.resume and not checkpoint_exists(str(args.resume)):
        raise FileNotFoundError(f"No native training checkpoint found in {args.resume}.")
    if int(os.environ.get("RANK", "0")) == 0:
        if args.resume:
            if args.resume.resolve() != (args.output / "checkpoints").resolve():
                raise ValueError("--resume must point to this run's OUTPUT/checkpoints directory.")
            if json.loads((args.output / "run.json").read_text()) != run:
                raise ValueError("Resume requires the same architecture, dataset, schedule, batch layout and seed.")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    cfg = rwkv7_ohlcv_config(
        hf,
        data=args.data,
        output=args.output,
        steps=args.steps,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
        epochs=args.epochs,
    )
    cfg.checkpoint.load = str(args.resume) if args.resume else None
    cfg.train.exit_interval = args.exit_interval
    callbacks = [ForecastExportCallback(args.output, run=run)]
    if args.diagnostics_interval:
        from megatron.bridge.recipes.rwkv7.forecast_diagnostics import ForecastDiagnosticsCallback

        callbacks.append(ForecastDiagnosticsCallback(args.output, interval=args.diagnostics_interval))
    pretrain(cfg, forecast_forward_step, callbacks=callbacks)


def load_forecaster(checkpoint: Path) -> tuple[RWKV7ForecastModel, ForecastCheckpoint]:
    """Reload exported forecast weights on one GPU in a fresh process."""
    import torch
    from transformers import RwkvConfig

    from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import RWKV7ForecastConfig, RWKV7ForecastModel
    from megatron.bridge.models.rwkv7.rwkv7_bridge import RWKV7Bridge

    payload: ForecastCheckpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["version"] != 1:
        raise ValueError("Unsupported forecast checkpoint version.")
    base = RWKV7Bridge().hf_config_to_model_config(RwkvConfig(**payload["run"]["hf_config"]))
    cfg = RWKV7ForecastConfig(
        transformer=base.transformer,
        hf_config=base.hf_config,
        vocab_size=1,
        seq_length=base.seq_length,
        should_pad_vocab=False,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    )
    with torch.device("cuda"):
        model = RWKV7ForecastModel(cfg, None)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def predict(args: argparse.Namespace) -> None:
    """Generate subsequent OHLCV bars from the end of one local instrument."""
    import torch

    from megatron.bridge.data.datasets.ohlcv import (
        OHLCV_COLUMNS,
        contiguous_segments,
        decode_bars,
        encode_bars,
        load_instrument,
    )

    model, payload = load_forecaster(args.checkpoint)
    exchange, symbol, timestamps, bars = load_instrument(args.instrument)
    trained_context = payload["run"]["hf_config"]["context_length"]
    segments = contiguous_segments(timestamps, bars)
    if not segments or segments[-1][1] != len(bars) or segments[-1][1] - segments[-1][0] < 2:
        raise ValueError("Instrument must end with at least two valid contiguous minute bars.")
    context = min(trained_context, segments[-1][1] - segments[-1][0] - 1)
    features = torch.from_numpy(encode_bars(bars[-context - 1 :])).unsqueeze(0).cuda()
    encoded = model.generate(features, horizon=args.horizon)[0].cpu().numpy()
    prediction = decode_bars(encoded, previous_close=float(bars[-1, 3]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("exchange", "symbol", "context_end", "context_bars", "step", *OHLCV_COLUMNS))
        for step, values in enumerate(prediction, start=1):
            writer.writerow((exchange, symbol, str(timestamps[-1]), context, step, *values.tolist()))
    logger.info(
        "Saved %d predicted bars to %s (context %d, checkpoint step %s)",
        args.horizon,
        args.output,
        context,
        payload["step"],
    )


def evaluate(args: argparse.Namespace) -> None:
    """Compare free-running multi-bar forecasts with a last-bar baseline."""
    import numpy as np
    import torch

    from megatron.bridge.data.datasets.ohlcv import (
        OHLCV_COLUMNS,
        PRICE_SCALE,
        OHLCVWindowDataset,
        decode_bars,
        manifest_digest,
    )

    model, payload = load_forecaster(args.checkpoint)
    if manifest_digest(args.data) != payload["run"]["manifest_sha256"]:
        raise ValueError("Evaluation must use the training run's prepared dataset manifest.")
    context = payload["run"]["hf_config"]["context_length"]
    dataset = OHLCVWindowDataset(
        args.data, split=args.split, seq_length=context, horizon=args.horizon, stride=args.horizon
    )
    count = min(args.max_windows, len(dataset))
    errors = np.zeros((args.horizon, 5), dtype=np.float64)
    squared = np.zeros_like(errors)
    baseline_errors = np.zeros_like(errors)
    baseline_squared = np.zeros_like(errors)
    # Spread a bounded evaluation across the full chronological window inventory.
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    for index in indices:
        _, start, arrays = dataset.window(int(index))
        end = start + context
        anchor = float(arrays["close"][end - 1])
        history = np.array(arrays["features"][start:end])
        encoded = model.generate(torch.from_numpy(history).unsqueeze(0).cuda(), horizon=args.horizon)
        predicted = decode_bars(encoded[0].cpu().numpy(), previous_close=anchor)
        actual = decode_bars(arrays["features"][end : end + args.horizon], previous_close=anchor)
        # Persistence: repeat the last observed complete OHLCV bar.
        previous_last = anchor * np.exp(-float(history[-1, 1]) / PRICE_SCALE)
        last = decode_bars(history[-1:], previous_close=previous_last)
        delta, baseline_delta = predicted - actual, last - actual
        errors += np.abs(delta)
        squared += delta**2
        baseline_errors += np.abs(baseline_delta)
        baseline_squared += baseline_delta**2
    report = {
        "split": args.split,
        "windows": count,
        "horizon": args.horizon,
        "checkpoint_step": payload["step"],
        "manifest_sha256": payload["run"]["manifest_sha256"],
        "metrics": {
            name: {
                "mae_by_step": (errors[:, column] / count).tolist(),
                "rmse_by_step": np.sqrt(squared[:, column] / count).tolist(),
                "persistence_mae_by_step": (baseline_errors[:, column] / count).tolist(),
                "persistence_rmse_by_step": np.sqrt(baseline_squared[:, column] / count).tolist(),
            }
            for column, name in enumerate(OHLCV_COLUMNS)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.info("Evaluated %d free-running windows; saved %s", count, args.output)


def main() -> None:
    """Expose preparation, native training, prediction and evaluation commands."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare a local canonical Parquet snapshot")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--train-end", required=True, help="Exclusive local date, YYYY-MM-DD")
    prep.add_argument("--validation-end", required=True, help="Exclusive local date, YYYY-MM-DD")
    prep.add_argument("--revision", required=True, help="Downloaded Hugging Face dataset commit SHA")
    prep.add_argument("--workers", type=int, default=1, help="Parallel per-instrument preprocessing workers")
    prep.set_defaults(handler=prepare)
    training = commands.add_parser("train", help="Train from scratch with native Bridge checkpoints")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--context", type=int, default=64)
    budget = training.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int, help="Optimizer steps; defaults to 10000 without --epochs")
    budget.add_argument(
        "--epochs", type=int, help="Complete passes with nonoverlapping targets, retaining tail windows"
    )
    training.add_argument("--global-batch-size", type=int, default=32)
    training.add_argument("--micro-batch-size", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--warmup-steps", type=int, default=100)
    training.add_argument("--save-interval", type=int, default=1000)
    training.add_argument("--eval-interval", type=int, default=1000)
    training.add_argument("--eval-iters", type=int, default=10)
    training.add_argument("--seed", type=int, default=1234)
    training.add_argument("--resume", type=Path)
    training.add_argument("--exit-interval", type=int, help="Native bounded exit, retaining the full LR schedule")
    training.add_argument(
        "--calibration-windows", type=int, default=0, help="Training windows for head initialization"
    )
    training.add_argument("--head-gain", type=float, default=0.01, help="Orthogonal gain of the calibrated head")
    training.add_argument("--head-fp32", action="store_true", help="Compute the continuous output projection in FP32")
    training.add_argument(
        "--rwkv-optimizer", action="store_true", help="Use RWKV-LM matrix-only decay and 2x w0 learning rate"
    )
    training.add_argument("--diagnostics-interval", type=int, default=0, help="Module/activation metrics interval")
    training.set_defaults(handler=train)
    prediction = commands.add_parser("predict", help="Autoregressively generate future OHLCV bars")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--instrument", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--horizon", type=int, required=True)
    prediction.set_defaults(handler=predict)
    evaluation = commands.add_parser("evaluate", help="Free-running holdout evaluation against persistence")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--horizon", type=int, required=True)
    evaluation.add_argument("--max-windows", type=int, default=1000)
    evaluation.set_defaults(handler=evaluate)
    args = parser.parse_args()
    if args.command == "train":
        if args.steps is None and args.epochs is None:
            args.steps = 10000
        if (
            any(
                value <= 0
                for value in (
                    args.context,
                    args.global_batch_size,
                    args.micro_batch_size,
                    args.learning_rate,
                    args.save_interval,
                    args.eval_interval,
                )
            )
            or args.eval_iters < 0
            or args.warmup_steps < 0
            or args.calibration_windows < 0
            or args.head_gain <= 0
            or args.diagnostics_interval < 0
            or (args.steps is not None and (args.steps <= 0 or args.warmup_steps >= args.steps))
            or (args.epochs is not None and args.epochs <= 0)
        ):
            parser.error("Positive training settings and 0 <= warmup_steps < steps are required.")
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if args.global_batch_size % (world * args.micro_batch_size):
            parser.error("global batch size must be divisible by world size times micro batch size")
        if args.exit_interval is not None and args.exit_interval <= 0:
            parser.error("--exit-interval must be positive")
    if args.command in ("predict", "evaluate") and args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.command == "evaluate" and args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
