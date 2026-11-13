# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare local A-share bars, train RWKV7, and autoregressively forecast OHLCV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

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


def training_feature_columns(covariates: Path | None, pvw_cache: Path | None) -> list[int] | None:
    """Return the covariate-cache columns that enter the model, or None without covariates."""
    if covariates is None:
        return None
    from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES, MARKET_FEATURE_NAMES

    columns = list(range(len(FORECAST_FEATURE_NAMES)))
    if pvw_cache is not None:
        columns = json.loads((pvw_cache / "manifest.json").read_text())["columns"]
    # Market aggregates are shared by every instrument at a minute; models
    # memorized training-period market moves from them and lost the gain
    # out of sample, so they never enter the model.
    return [i for i in columns if FORECAST_FEATURE_NAMES[i] not in MARKET_FEATURE_NAMES]


def train(args: argparse.Namespace) -> None:
    """Train with Bridge's optimizer, scheduler, DDP and distributed checkpoints."""
    from transformers import RwkvConfig

    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset, OHLCVWindowDataset, manifest_digest
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
    direct = args.task == "direct_quantile"
    pvw_cache = args.pvw_cache
    if args.forecast_architecture is not None:
        hf.forecast_architecture = args.forecast_architecture
    if direct:
        hf.forecast_task = "direct_quantile"
        hf.forecast_horizon = args.forecast_horizon
        hf.forecast_quantiles = [i / 10 for i in range(1, 10)]
    if args.covariates:
        from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES

        if pvw_cache:
            metadata = json.loads((pvw_cache / "manifest.json").read_text())
            if (
                metadata["validation_indices_sha256"]
                != hashlib.sha256(args.validation_indices.read_bytes()).hexdigest()
            ):
                raise ValueError("Validation indices do not match the PVW cache.")
            hf.forecast_pvw = {key: metadata[key] for key in ("modes", "alpha", "columns")}
        hf.forecast_feature_columns = training_feature_columns(args.covariates, pvw_cache)
        hf.forecast_feature_names = [FORECAST_FEATURE_NAMES[i] for i in hf.forecast_feature_columns]
        if pvw_cache:
            hf.forecast_feature_names.append(metadata["feature_names"][-1])
        hf.forecast_input_size = len(hf.forecast_feature_names)
    dataset_class = (
        partial(
            OHLCVQuantileWindowDataset,
            covariates=args.covariates,
            pvw_cache=pvw_cache,
            feature_columns=getattr(hf, "forecast_feature_columns", None),
        )
        if direct
        else OHLCVWindowDataset
    )
    reference_features = None
    if args.calibration_windows:
        import numpy as np

        calibration = dataset_class(
            args.data,
            split="train",
            seq_length=args.context,
            horizon=args.forecast_horizon if direct else 1,
            stride=1 if direct else args.context,
            keep_tail=not direct,
            seed=args.seed,
        )
        indices = (
            np.arange(args.calibration_windows, dtype=np.int64) * calibration.num_windows // args.calibration_windows
        )
        if pvw_cache:
            reference_features = np.empty((len(indices), 23), dtype=np.float32)
            for positions, _, starts, arrays, _ in calibration.iter_raw_windows(calibration.raw_indices(indices)):
                reference_features[positions] = arrays["features"][starts + args.context - 1]
        samples = calibration.__getitems__(indices.tolist()) if pvw_cache else [calibration[int(i)] for i in indices]
        hf.forecast_initialization = "calibrated"
        hf.forecast_head_gain = args.head_gain
        if direct:
            import torch

            from megatron.bridge.data.datasets.ohlcv import FEATURE_NAMES
            from megatron.bridge.data.datasets.ohlcv_covariates import RETURN_FEATURE_NAMES
            from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import (
                forecast_scaled_inputs,
                forecast_window_rms,
                forecast_window_scale,
            )

            windows = torch.from_numpy(np.stack([sample["features"] for sample in samples]))
            targets = np.stack([sample["labels"] for sample in samples])
            calibration_features = windows[:, -1].numpy()
            # Quantiles are learned relative to each window's own volatility,
            # so intervals follow volatility regimes absent from training.
            rms = forecast_window_rms(windows, floor=0)
            # The floor bounds scaled errors after flat, untraded histories.
            hf.forecast_volatility_floor = float(torch.quantile(rms[rms > 0], 0.05))
            scale = forecast_window_scale(windows, floor=hf.forecast_volatility_floor, horizon=args.forecast_horizon)
            scaled_targets = targets / scale.numpy()
            hf.forecast_target_quantiles = np.quantile(scaled_targets, hf.forecast_quantiles, axis=0).T.tolist()
            names = getattr(hf, "forecast_feature_names", list(FEATURE_NAMES))
            hf.forecast_scaled_features = [i for i, name in enumerate(names) if name in RETURN_FEATURE_NAMES]
            mask = torch.tensor([i in hf.forecast_scaled_features for i in range(len(names))])
            model_inputs = forecast_scaled_inputs(windows, mask, floor=hf.forecast_volatility_floor)
            values = model_inputs.reshape(-1, model_inputs.shape[-1]).double().numpy()
        else:
            values = np.concatenate([sample["features"][sample["loss_mask"].astype(bool)] for sample in samples])
            labels = np.concatenate([sample["labels"][sample["loss_mask"].astype(bool)] for sample in samples])
            hf.forecast_target_mean = labels.astype(np.float64).mean(axis=0).tolist()
        logger.info("Training-only head calibration: %d input bars", len(values))
        if args.normalize_inputs:
            values = values.astype(np.float64)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std[std == 0] = 1
            hf.forecast_input_mean = mean.tolist()
            hf.forecast_input_std = std.tolist()
            logger.info("Training-only input mean=%s std=%s", hf.forecast_input_mean, hf.forecast_input_std)
    if args.head_fp32:
        hf.forecast_head_fp32 = True
    if args.rwkv_optimizer:
        hf.forecast_optimizer = "rwkv_lm"
    # Count actual windows before resolving an epoch-based optimizer-step budget.
    epoch_windows = None
    for split in ("train", "validation", "test") if args.eval_iters else ("train",):
        dataset = dataset_class(
            args.data,
            split=split,
            seq_length=args.context,
            horizon=args.forecast_horizon if direct else 1,
            stride=args.context if args.epochs is not None and not direct else 1,
            keep_tail=args.epochs is not None and not direct,
        )
        logger.info("%s: %d available windows", split, dataset.num_windows)
        if split == "train" and args.epochs is not None:
            epoch_windows = dataset.num_windows * args.epochs
            args.steps = math.ceil(epoch_windows / args.global_batch_size)
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
    if args.covariates is not None:
        run["covariates"] = str(args.covariates)
        run["covariates_sha256"] = hashlib.sha256((args.covariates / "manifest.json").read_bytes()).hexdigest()
    if pvw_cache is not None:
        run["pvw_cache"] = str(pvw_cache)
        run["pvw_cache_sha256"] = manifest_digest(pvw_cache)
    if args.packed_train is not None:
        run["packed_train"] = str(args.packed_train)
        run["packed_train_sha256"] = manifest_digest(args.packed_train)
    if args.early_stopping_patience:
        run["early_stopping_patience"] = args.early_stopping_patience
    if args.data_workers is not None:
        run["data_workers"] = args.data_workers
    if args.validation_indices is not None:
        run["validation_indices"] = str(args.validation_indices)
        run["validation_indices_sha256"] = hashlib.sha256(args.validation_indices.read_bytes()).hexdigest()
    if direct:
        samples = epoch_windows if epoch_windows is not None else args.steps * args.global_batch_size
        run["sample_budget"] = {
            "windows": samples,
            "history_bars": samples * args.context,
            "future_positions": samples * args.forecast_horizon,
        }
        run["forecast_references"] = {
            "chronos_bolt": "1224fafa91e1ecd6674f5204066b159e40b5e4a1",
            "tft": "0eebfd403ba438bfd14cb70b9b24a7745c997083",
            "timesfm": "e51928e27119cb17bebc005be2696b75e0a9e688",
            "qlib": "54355232463878d2eebb91fe0ee5fa7fa1f5976c",
            "master": "de8f58557096abde4216a701b35fc4368158d111",
            "kronos": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
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
            if direct and args.calibration_windows:
                np.savez(
                    args.output / "calibration.npz",
                    features=np.asarray(calibration_features),
                    targets=np.asarray(targets),
                    **({"reference_features": reference_features} if reference_features is not None else {}),
                )
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
        covariates=args.covariates,
        validation_indices=args.validation_indices,
        pvw_cache=pvw_cache,
        packed_train=args.packed_train,
        data_workers=args.data_workers,
    )
    cfg.checkpoint.load = str(args.resume) if args.resume else None
    cfg.train.exit_interval = args.exit_interval
    callback = ForecastExportCallback(args.output, run=run, patience=args.early_stopping_patience or None)
    pretrain(cfg, forecast_forward_step, callbacks=[callback])


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
    import numpy as np
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
    hf_config = payload["run"]["hf_config"]
    if model.direct_quantile and model.input_size > 5:
        from hf_forecast.processing_rwkv7_forecast import prepare_features

        from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES, MARKET_FEATURE_NAMES

        start, _ = segments[-1]
        if context != trained_context:
            raise ValueError("Covariate prediction requires the complete trained context.")
        pvw = hf_config.get("forecast_pvw")
        columns = hf_config.get(
            "forecast_feature_columns", pvw["columns"] if pvw else list(range(len(FORECAST_FEATURE_NAMES)))
        )
        if any(FORECAST_FEATURE_NAMES[i] in MARKET_FEATURE_NAMES for i in columns):
            if args.market is None:
                raise ValueError("This checkpoint requires --market with three same-minute market columns.")
            market = np.load(args.market, allow_pickle=False)
            if market.shape != (len(bars), 3):
                raise ValueError("Market rows must align with the instrument bars.")
        else:
            # The feature builder always emits market columns; this checkpoint discards them.
            market = np.zeros((len(bars), 3), dtype=np.float32)
        values, _ = prepare_features(bars[start:], timestamps[start:], market[start:], context_length=context)
        features = torch.from_numpy(values[..., columns]).unsqueeze(0).cuda()
        if pvw:
            from megatron.bridge.data.datasets.ohlcv_vmd import denoise_history

            close = torch.from_numpy(bars[-context:, 3].copy()).unsqueeze(0).cuda()
            filtered = denoise_history(close, modes=pvw["modes"], alpha=pvw["alpha"])
            features = torch.cat((features, filtered.unsqueeze(-1)), dim=-1)
    else:
        features = torch.from_numpy(encode_bars(bars[-context - 1 :])).unsqueeze(0).cuda()
    if model.direct_quantile:
        if args.horizon > model.horizon:
            raise ValueError("Requested horizon exceeds the trained direct forecast horizon.")
        raw = model.predict_quantiles(features, fix_quantile_crossing=False)[0, : args.horizon].cpu().numpy()
        ordered = np.sort(raw, axis=-1)
        prices = bars[-1, 3] * np.exp(ordered.astype(np.float64) / 100)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "exchange",
                    "symbol",
                    "context_end",
                    "context_bars",
                    "step",
                    "quantile",
                    "raw_return",
                    "return",
                    "close",
                )
            )
            for h in range(args.horizon):
                for q, level in enumerate(model.quantile_levels):
                    writer.writerow(
                        (
                            exchange,
                            symbol,
                            str(timestamps[-1]),
                            context,
                            h + 1,
                            level,
                            raw[h, q],
                            ordered[h, q],
                            prices[h, q],
                        )
                    )
        if args.barrier_pct is not None:
            _report_barriers(model, features, args, exchange, symbol, timestamps[-1], float(bars[-1, 3]))
        logger.info("Saved direct quantile forecast to %s", args.output)
        return
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


def _report_barriers(
    model: "RWKV7ForecastModel",
    features: "torch.Tensor",
    args: argparse.Namespace,
    exchange: str,
    symbol: str,
    origin: object,
    anchor: float,
) -> None:
    """Write touch probabilities, first-touch minutes and running extremes."""
    import json as _json

    import torch

    from megatron.bridge.data.datasets.ohlcv import PRICE_SCALE

    generator = torch.Generator(device=features.device).manual_seed(1234)
    forecast = model.predict_paths(features, samples=args.path_samples, generator=generator)
    # Barriers are percentage moves; labels are PRICE_SCALE * log(close / anchor).
    upper = PRICE_SCALE * math.log1p(args.barrier_pct / 100)
    lower = PRICE_SCALE * math.log1p(-args.barrier_pct / 100)
    up_minute, up_probability = forecast.first_touch_minute(upper)
    down_minute, down_probability = forecast.first_touch_minute(lower)
    levels = list(model.quantile_levels)
    report = {
        "exchange": exchange,
        "symbol": symbol,
        "context_end": str(origin),
        "anchor_close": anchor,
        "barrier_pct": args.barrier_pct,
        "path_samples": args.path_samples,
        "quantile_levels": levels,
        "touch_upper_probability": float(up_probability[0]),
        "touch_lower_probability": float(down_probability[0]),
        "upper_before_lower_probability": float(forecast.barrier_race(upper, lower)[0]),
        "median_first_touch_minute_upper": float(up_minute[0]),
        "median_first_touch_minute_lower": float(down_minute[0]),
        "running_max_pct_by_quantile": (torch.expm1(forecast.running_max[0, -1] / PRICE_SCALE) * 100).tolist(),
        "running_min_pct_by_quantile": (torch.expm1(forecast.running_min[0, -1] / PRICE_SCALE) * 100).tolist(),
    }
    destination = args.output.with_name(args.output.stem + "-paths.json")
    destination.write_text(_json.dumps(report, indent=2, allow_nan=False) + "\n")
    numpy_destination = args.output.with_name(args.output.stem + "-paths.npy")
    import numpy as _np

    _np.save(numpy_destination, forecast.paths[0].cpu().numpy())
    logger.info(
        "Touch +%.2f%%: %.1f%% (median minute %.0f) | touch -%.2f%%: %.1f%% | upper first: %.1f%%; saved %s",
        args.barrier_pct,
        100 * report["touch_upper_probability"],
        report["median_first_touch_minute_upper"],
        args.barrier_pct,
        100 * report["touch_lower_probability"],
        100 * report["upper_before_lower_probability"],
        destination,
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
    if model.direct_quantile:
        from forecast_quantile import evaluate_quantiles

        evaluate_quantiles(args, model, payload)
        return
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
        "--epochs", type=int, help="Complete stride-one direct quantile epochs or nonoverlapping next-bar epochs"
    )
    training.add_argument("--global-batch-size", type=int, default=32)
    training.add_argument("--micro-batch-size", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--warmup-steps", type=int, default=100)
    training.add_argument("--save-interval", type=int, default=250)
    training.add_argument("--eval-interval", type=int, default=250)
    training.add_argument("--eval-iters", type=int, default=10)
    training.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1000,
        help="Stop direct quantile training when validation pinball has not improved for this many steps (0 disables)",
    )
    training.add_argument("--seed", type=int, default=1234)
    training.add_argument("--resume", type=Path)
    training.add_argument("--exit-interval", type=int, help="Native bounded exit, retaining the full LR schedule")
    training.add_argument(
        "--calibration-windows",
        type=int,
        default=0,
        help="Training windows for head initialization; direct_quantile also fits its volatility floor",
    )
    training.add_argument("--head-gain", type=float, default=0.01, help="Orthogonal gain of the calibrated head")
    training.add_argument("--head-fp32", action="store_true", help="Compute the continuous output projection in FP32")
    training.add_argument(
        "--normalize-inputs", action="store_true", help="Standardize encoded inputs using training-only calibration"
    )
    training.add_argument(
        "--rwkv-optimizer", action="store_true", help="Use RWKV-LM matrix-only decay and 2x w0 learning rate"
    )
    training.add_argument("--task", choices=("next_bar", "direct_quantile"), default="next_bar")
    training.add_argument("--forecast-horizon", type=int, default=30)
    training.add_argument("--covariates", type=Path, help="Prepared past-only market and temporal feature cache")
    training.add_argument("--validation-indices", type=Path, help="Frozen stride-one validation index NPY")
    training.add_argument("--pvw-cache", type=Path, help="Frozen per-origin PVW feature cache")
    training.add_argument("--packed-train", type=Path, help="Indexed .bin/.idx training windows from the pack command")
    training.add_argument("--forecast-architecture", choices=("cnn_rwkv_attention",))
    training.add_argument("--data-workers", type=int, help="Persistent DataLoader workers per rank")
    training.set_defaults(handler=train)
    prediction = commands.add_parser("predict", help="Autoregressively generate future OHLCV bars")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--instrument", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--horizon", type=int, required=True)
    prediction.add_argument("--market", type=Path, help="Same-minute market statistics NPY aligned to instrument rows")
    prediction.add_argument(
        "--barrier-pct",
        type=float,
        help="Report touch probability, first-touch minute and running extremes for +/- this percent",
    )
    prediction.add_argument("--path-samples", type=int, default=512, help="Paths drawn for the barrier report")
    prediction.set_defaults(handler=predict)
    evaluation = commands.add_parser("evaluate", help="Free-running holdout evaluation against persistence")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluation.add_argument("--horizon", type=int, required=True)
    evaluation.add_argument("--max-windows", type=int, default=1000)
    evaluation.add_argument("--windows", type=Path, help="Frozen window identities from an earlier evaluation")
    evaluation.add_argument("--baseline-paths", type=Path, help="Earlier autoregressive prediction NPZ")
    evaluation.add_argument("--direct-baseline-paths", type=Path, help="Earlier direct quantile prediction NPZ")
    evaluation.add_argument("--covariates", type=Path, help="Feature cache matching the checkpoint")
    evaluation.add_argument("--metrics-only", action="store_true", help="Skip large prediction arrays, CSV and plots")
    evaluation.add_argument("--pvw-cache", type=Path, help="PVW cache matching the checkpoint")
    evaluation.set_defaults(handler=evaluate)
    pvw = commands.add_parser("prepare-pvw", help="Fit and cache PVW features by forecast origin")
    for argument in ("data", "covariates", "validation-indices", "test-windows", "output"):
        pvw.add_argument(f"--{argument}", type=Path, required=True)
    pvw.add_argument("--batch-size", type=int, required=True)

    def cache_pvw(args: argparse.Namespace) -> None:
        from forecast_pvw import prepare_pvw

        prepare_pvw(args)

    pvw.set_defaults(handler=cache_pvw)
    packing = commands.add_parser("pack", help="Pack direct-quantile training windows into .bin/.idx")
    packing.add_argument("--data", type=Path, required=True)
    packing.add_argument("--covariates", type=Path, help="Covariate cache used for training")
    packing.add_argument("--pvw-cache", type=Path, help="PVW cache used for training")
    packing.add_argument("--output", type=Path, required=True)
    packing.add_argument("--context", type=int, default=64)
    packing.add_argument("--forecast-horizon", type=int, default=30)
    packing.add_argument("--workers", type=int, required=True)

    def pack_windows(args: argparse.Namespace) -> None:
        from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset
        from megatron.bridge.data.datasets.ohlcv_indexed import pack_quantile_windows

        pack_quantile_windows(
            OHLCVQuantileWindowDataset(
                args.data,
                split="train",
                seq_length=args.context,
                horizon=args.forecast_horizon,
                covariates=args.covariates,
                pvw_cache=args.pvw_cache,
                feature_columns=training_feature_columns(args.covariates, args.pvw_cache),
            ),
            args.output,
            workers=args.workers,
        )

    packing.set_defaults(handler=pack_windows)
    covariates = commands.add_parser("prepare-covariates", help="Cache past-only temporal/market covariates")
    covariates.add_argument("--data", type=Path, required=True)
    covariates.add_argument("--output", type=Path, required=True)
    covariates.add_argument("--workers", type=int, required=True)

    def cache_features(args: argparse.Namespace) -> None:
        from megatron.bridge.data.datasets.ohlcv_covariates import prepare_covariates

        prepare_covariates(args.data, args.output, workers=args.workers)

    covariates.set_defaults(handler=cache_features)
    selection = commands.add_parser("select-windows", help="Freeze month/time-stratified stride-one windows")
    selection.add_argument("--data", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--split", choices=("train", "validation", "test"), required=True)
    selection.add_argument("--count", type=int, required=True)
    selection.add_argument("--context", type=int, default=64)
    selection.add_argument("--horizon", type=int, default=30)
    selection.add_argument("--seed", type=int, default=1234)
    selection.add_argument("--exclude-dates", type=Path, help="Exclude complete dates from a frozen windows JSON")

    def freeze_windows(args: argparse.Namespace) -> None:
        from forecast_windows import select_windows

        select_windows(args)

    selection.set_defaults(handler=freeze_windows)
    args = parser.parse_args()
    if args.command in ("select-windows", "prepare-covariates"):
        if (args.command == "select-windows" and min(args.count, args.context, args.horizon) <= 0) or (
            args.command == "prepare-covariates" and args.workers <= 0
        ):
            parser.error("Counts, window lengths and worker count must be positive")
    if args.command == "prepare-pvw" and args.batch_size <= 0:
        parser.error("PVW batch size must be positive")
    if args.command == "pack" and (args.workers <= 0 or (args.pvw_cache and not args.covariates)):
        parser.error("Packing requires positive workers, and PVW requires its covariates")
    if args.command == "train":
        if args.pvw_cache is not None and (
            args.task != "direct_quantile"
            or args.context != 64
            or args.forecast_horizon != 30
            or not args.covariates
            or not args.validation_indices
        ):
            parser.error("PVW requires direct_quantile, 64/30 windows, covariates and frozen validation indices")
        if args.forecast_architecture is not None and args.task != "direct_quantile":
            parser.error("CNN with feature attention requires direct_quantile")
        if args.packed_train is not None and args.task != "direct_quantile":
            parser.error("--packed-train requires direct_quantile")
        if args.data_workers is not None and args.data_workers < 0:
            parser.error("Data worker count cannot be negative")
        if (args.covariates or args.validation_indices) and args.task != "direct_quantile":
            parser.error("Covariates and frozen validation indices require direct_quantile")
        if args.task == "direct_quantile" and (args.forecast_horizon <= 0 or args.calibration_windows <= 0):
            parser.error("direct_quantile requires a positive --forecast-horizon and --calibration-windows")
        if args.normalize_inputs and not args.calibration_windows:
            parser.error("--normalize-inputs requires --calibration-windows > 0")
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
            or (args.steps is not None and (args.steps <= 0 or args.warmup_steps >= args.steps))
            or (args.epochs is not None and args.epochs <= 0)
        ):
            parser.error("Positive training settings and 0 <= warmup_steps < steps are required.")
        if args.early_stopping_patience < 0:
            parser.error("--early-stopping-patience must be nonnegative")
        if args.task == "direct_quantile" and args.eval_iters and args.early_stopping_patience % args.eval_interval:
            parser.error("--early-stopping-patience must be a multiple of --eval-interval")
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if args.global_batch_size % (world * args.micro_batch_size):
            parser.error("global batch size must be divisible by world size times micro batch size")
        if args.exit_interval is not None and args.exit_interval <= 0:
            parser.error("--exit-interval must be positive")
    if args.command in ("predict", "evaluate") and args.horizon <= 0:
        parser.error("--horizon must be positive")
    if (
        args.command == "predict"
        and args.barrier_pct is not None
        and (args.barrier_pct <= 0 or args.path_samples <= 0)
    ):
        parser.error("--barrier-pct and --path-samples must be positive")
    if args.command == "evaluate" and args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
