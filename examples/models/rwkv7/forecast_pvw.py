# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit and materialize the past-only PVW channels for A-share forecasts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from megatron.bridge.data.datasets.ohlcv import FEATURE_NAMES, PRICE_SCALE, OHLCVQuantileWindowDataset, manifest_digest
from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES
from megatron.bridge.data.datasets.ohlcv_vmd import (
    CONTEXT,
    PEARSON_THRESHOLD,
    VMD_ITERATIONS,
    VMD_TOLERANCE,
    denoise_history,
    envelope_entropy,
    vmd_modes,
)


logger = logging.getLogger(__name__)
HORIZON = 30
SEED = 1234
POA_WINDOWS = 4096
XGB_WINDOWS = 1_048_576


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _atomic_array(path: Path, values: NDArray[Any]) -> None:
    temporary = path.with_suffix(".pending.npy")
    np.save(temporary, values, allow_pickle=False)
    temporary.replace(path)


def _sample_training(dataset: OHLCVQuantileWindowDataset) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(SEED)
    multiplier = int(rng.integers(1, len(dataset)))
    while math.gcd(multiplier, len(dataset)) != 1:
        multiplier = int(rng.integers(1, len(dataset)))
    pool_size = min(len(dataset), 4 * XGB_WINDOWS)
    pool = (multiplier * np.arange(pool_size) + int(rng.integers(len(dataset)))) % len(dataset)
    strata = np.empty(len(pool), dtype=np.int64)
    for positions, _, starts, arrays, _ in dataset.iter_raw_windows(pool):
        stamps = arrays["timestamps"][starts + CONTEXT - 1]
        months = stamps.astype("datetime64[M]").astype(np.int64)
        minutes = stamps.astype("datetime64[m]").astype(np.int64) % 1440
        strata[positions] = months * 1440 + minutes
    order = np.argsort(strata, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(strata[order])) + 1, len(order)]
    sizes = np.diff(boundaries)
    round_index = np.arange(len(order)) - np.repeat(boundaries[:-1], sizes)
    if len(pool) < XGB_WINDOWS:
        raise ValueError(f"PVW calibration requires at least {XGB_WINDOWS} distinct training windows.")
    # Shuffle the strata so the partial final round does not favor early months.
    shuffled_strata = rng.permutation(len(sizes))
    priority = np.empty(len(sizes), dtype=np.int64)
    priority[shuffled_strata] = np.arange(len(sizes))
    balanced = order[np.argsort(round_index * len(sizes) + np.repeat(priority, sizes))]
    return pool[balanced[:POA_WINDOWS]], pool[balanced[:XGB_WINDOWS]]


def _fit_features(dataset: OHLCVQuantileWindowDataset, indices: NDArray[np.int64]) -> dict[str, Any]:
    import xgboost as xgb

    names = FORECAST_FEATURE_NAMES[len(FEATURE_NAMES) :]
    inputs = np.empty((len(indices), len(names)), dtype=np.float32)
    targets = np.empty((len(indices), 4), dtype=np.float32)
    for positions, _, starts, arrays, _ in dataset.iter_raw_windows(indices):
        end = starts + CONTEXT - 1
        inputs[positions] = arrays["features"][end, len(FEATURE_NAMES) :]
        targets[positions] = PRICE_SCALE * np.log(
            arrays["close"][end[:, None] + np.array([1, 5, 15, 30])] / arrays["close"][end, None]
        )
    matrix = xgb.DMatrix(inputs, feature_names=list(names), nthread=16)
    scores = []
    for horizon, target in zip((1, 5, 15, 30), targets.T, strict=True):
        matrix.set_label(target)
        model = xgb.train(
            {
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "max_depth": 4,
                "eta": 0.05,
                "seed": SEED,
                "nthread": 16,
            },
            matrix,
            num_boost_round=256,
        )
        gains = model.get_score(importance_type="total_gain")
        values = np.array([gains.get(name, 0) for name in names])
        scores.append(values / max(values.sum(), 1e-30))
        logger.info("XGBoost feature ranking finished for horizon %d", horizon)
    mean_score = np.mean(scores, axis=0)
    ranked = np.argsort(-mean_score, kind="stable")[:11]
    columns = [*range(len(FEATURE_NAMES)), *sorted(int(i) + len(FEATURE_NAMES) for i in ranked if mean_score[i] > 0)]
    return {
        "columns": columns,
        "feature_names": [*(FORECAST_FEATURE_NAMES[i] for i in columns), "pvw_log_close"],
        "xgboost_version": xgb.__version__,
        "xgboost_importance": dict(zip(names, mean_score.tolist(), strict=True)),
    }


def _fit_poa(history: torch.Tensor) -> dict[str, Any]:
    # Courtship, approach and Lévy-flight updates from Zheng et al. (2022),
    # doi:10.3389/fenrg.2022.985523, equations 7–16, in normalized search coordinates.
    rng = np.random.default_rng(SEED)
    population = rng.random((12, 2))
    lower, width = np.array([2, 100]), np.array([8, 4900])
    scores: dict[tuple[int, float], float] = {}
    trace = []
    fitness = np.empty(len(population))
    best_score, best_modes, best_alpha = math.inf, 0, 0.0
    generations = 10
    for generation in range(generations + 1):
        for index, point in enumerate(lower + width * population):
            modes, alpha = int(np.rint(point[0])), float(point[1])
            key = (modes, alpha)
            if key not in scores:
                scores[key] = float(envelope_entropy(vmd_modes(history, modes=modes, alpha=alpha)))
            fitness[index] = scores[key]
            if fitness[index] < best_score:
                best_score, best_modes, best_alpha = float(fitness[index]), modes, alpha
        trace.append({"generation": generation, "modes": best_modes, "alpha": best_alpha, "entropy": best_score})
        logger.info("POA %s", trace[-1])
        if generation == generations:
            break
        population = population[np.argsort(fitness, kind="stable")]
        leaders = population[:5].copy()
        directions = rng.uniform(-1, 1, size=leaders.shape)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radius = 0.05 * (1 - generation / generations)
        population[:5] += (
            radius
            * np.array([1, 1.5, 2, 3, 5])[:, None]
            * directions
            * (rng.random(5) < [1, 0.9, 0.8, 0.6, 0.3])[:, None]
        )
        hens = population[5:9]
        males = leaders[rng.choice(5, size=len(hens), p=[0.4, 0.2, 0.2, 0.1, 0.1])]
        theta = 0.1 + 0.9 * generation / generations
        hens += 3 * theta * (males - hens)
        cubs = population[9:]
        guides = leaders[rng.integers(5, size=len(cubs))]
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        levy = rng.normal(0, sigma, size=cubs.shape) / np.maximum(np.abs(rng.normal(size=cubs.shape)), 1e-12) ** (
            1 / beta
        )
        cubs += 0.01 * levy * (leaders[0] - cubs) + rng.random((len(cubs), 1)) * (guides - cubs)
        np.clip(population, 0, 1, out=population)
        population[0] = (np.array([best_modes, best_alpha]) - lower) / width
    return {
        "modes": best_modes,
        "alpha": best_alpha,
        "poa_trace": trace,
        "poa_scores": [[modes, alpha, score] for (modes, alpha), score in scores.items()],
    }


def _history(dataset: OHLCVQuantileWindowDataset, indices: NDArray[np.int64]) -> NDArray[np.float64]:
    close = np.empty((len(indices), CONTEXT), dtype=np.float64)
    offsets = np.arange(CONTEXT)
    for positions, _, starts, arrays, _ in dataset.iter_raw_windows(indices):
        close[positions] = arrays["close"][starts[:, None] + offsets]
    return close


def _test_indices(dataset: OHLCVQuantileWindowDataset, windows: Path) -> NDArray[np.int64]:
    rows = json.loads(windows.read_text())
    indices = np.array([row["index"] for row in rows], dtype=np.int64)
    for index, row in zip(indices, rows, strict=True):
        if row.get("window_stride", HORIZON) != 1:
            raise ValueError("PVW test windows must use the stride-one inventory.")
        info, start, arrays = dataset.window(int(index))
        end = start + CONTEXT
        if (
            (row["exchange"], row["symbol"]) != (info["exchange"], info["symbol"])
            or row["context_end"] != str(arrays["timestamps"][end - 1])
            or row["timestamps"] != arrays["timestamps"][end : end + HORIZON].astype(str).tolist()
        ):
            raise ValueError("PVW test window does not match the prepared snapshot.")
    return indices


def _write_instrument(
    dataset: OHLCVQuantileWindowDataset,
    instrument: int,
    path: Path,
    *,
    batch_size: int,
    modes: int,
    alpha: float,
    device: torch.device,
) -> int:
    raw_path = dataset.root / dataset.instruments[instrument]["path"]
    segments = np.load(raw_path / "segments.npy", mmap_mode="r")
    segments = segments[segments[:, 2] == 0]
    counts = np.maximum(segments[:, 1] - CONTEXT - HORIZON + 1, 0)
    prefix = np.cumsum(counts)
    total = int(prefix[-1])
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape != (total, CONTEXT) or existing.dtype != np.float32:
            raise ValueError(f"Incompatible PVW shard: {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pending.npy")
    values = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(total, CONTEXT))
    close = np.load(raw_path / "close.npy", mmap_mode="r")
    offsets = np.arange(CONTEXT)
    for begin in range(0, total, batch_size):
        indices = np.arange(begin, min(begin + batch_size, total))
        segment = np.searchsorted(prefix, indices, side="right")
        prior = np.where(segment > 0, prefix[segment - 1], 0)
        starts = segments[segment, 0] + indices - prior
        batch = torch.from_numpy(close[starts[:, None] + offsets]).to(device)
        values[begin : begin + len(indices)] = denoise_history(batch, modes=modes, alpha=alpha).cpu().numpy()
    values.flush()
    del values
    temp.replace(path)
    return total


def prepare_pvw(args: argparse.Namespace) -> None:
    """Fit once on train windows and write immutable per-origin cache shards."""
    import hashlib

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group(backend="gloo", timeout=timedelta(days=2))
    datasets = {
        split: OHLCVQuantileWindowDataset(
            args.data, split=split, seq_length=CONTEXT, horizon=HORIZON, covariates=args.covariates
        )
        for split in ("train", "validation", "test")
    }
    dataset = datasets["train"]
    identity = {
        "version": 1,
        "source_manifest_sha256": manifest_digest(args.data),
        "covariates_sha256": manifest_digest(args.covariates),
        "validation_indices_sha256": hashlib.sha256(args.validation_indices.read_bytes()).hexdigest(),
        "test_windows_sha256": hashlib.sha256(args.test_windows.read_bytes()).hexdigest(),
        "context": CONTEXT,
        "horizon": HORIZON,
        "windows": {split: item.num_windows for split, item in datasets.items()},
        "seed": SEED,
        "vmd_iterations": VMD_ITERATIONS,
        "vmd_tolerance": VMD_TOLERANCE,
        "pearson_threshold": PEARSON_THRESHOLD,
        "wavelet": "haar-3-soft-mad-universal",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    contract = args.output / "settings.json"
    if rank == 0 and not contract.exists():
        pvw_indices, xgb_indices = _sample_training(dataset)
        _atomic_array(args.output / "poa-sample.npy", pvw_indices)
        _atomic_array(args.output / "xgboost-sample.npy", xgb_indices)
        close = torch.from_numpy(_history(dataset, pvw_indices)).to(device)
        history = (PRICE_SCALE * torch.log(close / close[:, :1])).float()
        if not history.abs().any():
            raise ValueError("The VMD calibration sample contains only constant closes.")
        params = _fit_poa(history)
        selected = _fit_features(dataset, xgb_indices)
        _atomic_json(contract, {**identity, **params, **selected})
    if world > 1:
        torch.distributed.barrier()
    settings = json.loads(contract.read_text())
    if any(settings[key] != value for key, value in identity.items()):
        raise ValueError("PVW resume requires identical data, cohorts and preprocessing semantics.")
    modes, alpha = settings["modes"], settings["alpha"]
    if rank == 0:
        cohorts = {
            "validation": np.load(args.validation_indices, allow_pickle=False),
            "test": _test_indices(datasets["test"], args.test_windows),
        }
        for split, indices in cohorts.items():
            indices = np.unique(indices)
            if not len(indices) or indices[0] < 0 or indices[-1] >= datasets[split].num_windows:
                raise ValueError(f"Invalid frozen {split} indices.")
            if (args.output / f"{split}-values.npy").exists():
                continue
            close = _history(datasets[split], indices)
            values = np.concatenate(
                [
                    denoise_history(torch.from_numpy(block).to(device), modes=modes, alpha=alpha).cpu().numpy()
                    for block in np.array_split(close, math.ceil(len(close) / args.batch_size))
                ]
            )
            _atomic_array(args.output / f"{split}-indices.npy", indices)
            _atomic_array(args.output / f"{split}-values.npy", values)
    start_time, progress_time, processed = time.monotonic(), time.monotonic(), 0
    sizes = np.diff(np.r_[0, dataset.cumulative])
    instruments = np.argsort(-sizes, kind="stable")[rank::world]
    for instrument in instruments:
        info = dataset.instruments[int(instrument)]
        path = args.output / "train" / info["path"] / "values.npy"
        processed += _write_instrument(
            dataset, int(instrument), path, batch_size=args.batch_size, modes=modes, alpha=alpha, device=device
        )
        if time.monotonic() - progress_time >= 30:
            logger.info(
                "PVW rank %d wrote %d windows at %.0f windows/s",
                rank,
                processed,
                processed / (time.monotonic() - start_time),
            )
            progress_time = time.monotonic()
    logger.info("PVW rank %d finished %d new windows in %.1fs", rank, processed, time.monotonic() - start_time)
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        _atomic_json(args.output / "manifest.json", settings)
        logger.info("PVW cache ready at %s", args.output)
    if world > 1:
        torch.distributed.destroy_process_group()
