# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Past-only OHLCV covariates and a contemporaneous, fixed-universe market index."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from megatron.bridge.data.datasets.ohlcv import FEATURE_NAMES, manifest_digest


logger = logging.getLogger(__name__)
# Same-minute aggregates shared by every instrument. Their history identifies
# the training-period timestamp, so forecast inputs exclude them.
MARKET_FEATURE_NAMES = ("market_return", "market_volatility", "market_up_fraction")
COVARIATE_NAMES = (
    "minute_sin",
    "minute_cos",
    "afternoon",
    "weekday_sin",
    "weekday_cos",
    *tuple(f"return_{n}" for n in (5, 15, 30, 60)),
    *tuple(f"volatility_{n}" for n in (5, 15, 30, 60)),
    "relative_volume_60",
    "up_fraction_60",
    *MARKET_FEATURE_NAMES,
)
FORECAST_FEATURE_NAMES = (*FEATURE_NAMES, *COVARIATE_NAMES)


def rolling_mean(values: NDArray[Any], starts: NDArray[Any], window: int) -> NDArray[np.float64]:
    """Compute trailing means without crossing an instrument's segment boundaries."""
    right = np.arange(1, len(values) + 1)
    left = np.maximum(right - window, starts)
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return np.asarray((prefix[right] - prefix[left]) / (right - left), dtype=np.float64)


def forecast_covariates(
    features: NDArray[Any], timestamps: NDArray[Any], segments: NDArray[Any], market: NDArray[Any]
) -> NDArray[np.float32]:
    """Encode only completed bars, resetting rolling statistics at every segment."""
    starts = np.repeat(segments[:, 0], segments[:, 1])
    minutes = timestamps.astype("datetime64[m]").astype(np.int64) % 1440
    weekdays = (timestamps.astype("datetime64[D]").astype(np.int64) + 3) % 7
    minute_phase = 2 * np.pi * minutes / 1440
    weekday_phase = 2 * np.pi * weekdays / 7
    returns = features[:, 1].astype(np.float64)
    volume = np.expm1(features[:, 4].astype(np.float64) * 10)
    columns = [
        np.sin(minute_phase),
        np.cos(minute_phase),
        (minutes >= 780).astype(float),
        np.sin(weekday_phase),
        np.cos(weekday_phase),
    ]
    right = np.arange(1, len(features) + 1)
    for n in (5, 15, 30, 60):
        columns.append(rolling_mean(returns, starts, n) * (right - np.maximum(right - n, starts)))
    for n in (5, 15, 30, 60):
        columns.append(np.sqrt(np.maximum(rolling_mean(returns**2, starts, n), 0)))
    columns.extend(
        (np.log1p(volume) - np.log1p(rolling_mean(volume, starts, 60)), rolling_mean(returns > 0, starts, 60))
    )
    return np.concatenate((features, np.column_stack(columns), market), axis=1).astype(np.float32)


def _prepare_instrument(path: str, *, root: Path, output: Path, first_minute: int) -> None:
    directory = root / path
    features = np.load(directory / "features.npy", mmap_mode="r")
    timestamps = np.load(directory / "timestamps.npy", mmap_mode="r")
    segments = np.load(directory / "segments.npy", mmap_mode="r")
    market = np.load(output / "market.npy", mmap_mode="r")
    indices = timestamps.astype("datetime64[m]").astype(np.int64) - first_minute
    values = forecast_covariates(features, timestamps, segments, market[indices])
    destination = output / path
    destination.mkdir(parents=True)
    np.save(destination / "features.npy", values)


def prepare_covariates(root: Path, output: Path, *, workers: int) -> None:
    """Persist immutable covariates alongside the original, unchanged prepared data.

    Market aggregates use the manifest's fixed instrument universe and completed
    bars at the same timestamp. No future returns or fitted statistics enter them.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    instruments = manifest["instruments"]
    first = min(np.datetime64(i["start"], "m").astype(np.int64) for i in instruments)
    last = max(np.datetime64(i["end"], "m").astype(np.int64) for i in instruments)
    output.mkdir(parents=True, exist_ok=False)
    totals = np.zeros((int(last - first + 1), 4), dtype=np.float64)
    for number, instrument in enumerate(instruments):
        directory = root / instrument["path"]
        stamps = np.load(directory / "timestamps.npy", mmap_mode="r")
        returns = np.load(directory / "features.npy", mmap_mode="r")[:, 1].astype(np.float64)
        indices = stamps.astype("datetime64[m]").astype(np.int64) - first
        # Each instrument has strictly increasing timestamps, so indices are unique.
        totals[indices, 0] += returns
        totals[indices, 1] += returns**2
        totals[indices, 2] += returns > 0
        totals[indices, 3] += 1
        if number % 250 == 0:
            logger.info("Market aggregation: %d/%d instruments", number, len(instruments))
    count = np.maximum(totals[:, 3], 1)
    mean = totals[:, 0] / count
    market = np.column_stack(
        (mean, np.sqrt(np.maximum(totals[:, 1] / count - mean**2, 0)), totals[:, 2] / count)
    ).astype(np.float32)
    np.save(output / "market.npy", market)
    del totals, market
    operation = partial(_prepare_instrument, root=root, output=output, first_minute=int(first))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for number, _ in enumerate(executor.map(operation, (i["path"] for i in instruments))):
            if number % 250 == 0:
                logger.info("Covariate preparation: %d/%d instruments", number, len(instruments))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_manifest_sha256": manifest_digest(root),
                "feature_names": FORECAST_FEATURE_NAMES,
                "first_market_minute": int(first),
                "universe": "all instruments in the source manifest; equal-weight completed-bar returns",
                "rolling_boundary": "source segment, including split/session/gap boundaries",
            },
            indent=2,
        )
        + "\n"
    )
