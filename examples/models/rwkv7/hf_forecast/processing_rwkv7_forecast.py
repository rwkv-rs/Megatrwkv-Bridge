# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy feature preparation matching the causal training data contract."""

import numpy as np
from numpy.typing import NDArray


FEATURE_NAMES = (
    "open_return",
    "close_return",
    "upper_wick",
    "lower_wick",
    "log_volume",
    "minute_sin",
    "minute_cos",
    "afternoon",
    "weekday_sin",
    "weekday_cos",
    "return_5",
    "return_15",
    "return_30",
    "return_60",
    "volatility_5",
    "volatility_15",
    "volatility_30",
    "volatility_60",
    "relative_volume_60",
    "up_fraction_60",
    "market_return",
    "market_volatility",
    "market_up_fraction",
)


def market_statistics(returns: NDArray[np.float64]) -> NDArray[np.float32]:
    """Aggregate same-minute scaled log returns shaped [time, stocks].

    Missing instrument bars are NaN. Use the fixed instrument universe and
    drop each instrument's first raw bar at every session/gap boundary.
    """
    if returns.ndim != 2 or np.isinf(returns).any() or np.any(np.all(np.isnan(returns), axis=1)):
        raise ValueError("Expected [time, stocks] returns with at least one finite value per minute.")
    count = np.sum(~np.isnan(returns), axis=1)
    mean = np.nansum(returns, axis=1) / count
    variance = np.nansum(returns**2, axis=1) / count - mean**2
    return np.column_stack((mean, np.sqrt(np.maximum(variance, 0)), np.sum(returns > 0, axis=1) / count)).astype(
        np.float32
    )


def prepare_features(
    bars: NDArray[np.float64],
    timestamps: NDArray[np.datetime64],
    market: NDArray[np.float32],
    *,
    context_length: int = 64,
) -> tuple[NDArray[np.float32], float]:
    """Encode one complete session prefix and take its last history window.

    Args:
        bars: Raw [N, 5] open/high/low/close/volume from the actual contiguous
            session start through the prediction origin, with N >= 65.
        timestamps: Shanghai local, timezone-naive minute timestamps, shape [N].
        market: Same-minute market return, population volatility and up fraction,
            shape [N, 3]. Return/volatility units are 100 * log(close/previous).
        context_length: Number of encoded bars consumed by the model.

    Returns:
        Unnormalized [context_length, 23] features and last raw closing price.
    """
    bars = np.asarray(bars, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype="datetime64[us]")
    market = np.asarray(market, dtype=np.float32)
    if bars.ndim != 2 or bars.shape[1] != 5 or len(bars) < context_length + 1:
        raise ValueError("Supply at least context_length + 1 raw OHLCV bars from the contiguous session start.")
    if timestamps.shape != (len(bars),) or market.shape != (len(bars), 3):
        raise ValueError("timestamps and market must align with every raw OHLCV bar.")
    minute = timestamps.astype("datetime64[m]").astype(np.int64) % 1440
    if (
        np.isnat(timestamps).any()
        or np.any(timestamps != timestamps.astype("datetime64[m]"))
        or np.any(np.diff(timestamps) != np.timedelta64(1, "m"))
        or not (np.all((minute >= 570) & (minute <= 690)) or np.all((minute >= 780) & (minute <= 900)))
    ):
        raise ValueError("Use one contiguous one-minute morning or afternoon session in Shanghai local time.")
    opening, high, low, close, volume = bars.T
    if (
        not np.isfinite(bars).all()
        or not np.isfinite(market).all()
        or np.any(bars[:, :4] <= 0)
        or np.any(volume < 0)
        or np.any(high < np.maximum(opening, close))
        or np.any(low > np.minimum(opening, close))
        or np.any(market[:, 1] < 0)
        or np.any((market[:, 2] < 0) | (market[:, 2] > 1))
    ):
        raise ValueError("Invalid OHLCV values or market statistics.")
    encoded = np.column_stack(
        (
            100 * np.log(opening[1:] / close[:-1]),
            100 * np.log(close[1:] / close[:-1]),
            100 * np.log(high[1:] / np.maximum(opening[1:], close[1:])),
            100 * np.log(np.minimum(opening[1:], close[1:]) / low[1:]),
            np.log1p(volume[1:]) / 10,
        )
    ).astype(np.float32)
    minutes = minute[1:]
    weekdays = (timestamps[1:].astype("datetime64[D]").astype(np.int64) + 3) % 7
    columns = [
        np.sin(2 * np.pi * minutes / 1440),
        np.cos(2 * np.pi * minutes / 1440),
        (minutes >= 780).astype(float),
        np.sin(2 * np.pi * weekdays / 7),
        np.cos(2 * np.pi * weekdays / 7),
    ]
    returns = encoded[:, 1].astype(np.float64)
    right = np.arange(1, len(encoded) + 1)

    def trailing_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
        left = np.maximum(right - window, 0)
        prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        return (prefix[right] - prefix[left]) / (right - left)

    for window in (5, 15, 30, 60):
        columns.append(trailing_mean(returns, window) * np.minimum(right, window))
    for window in (5, 15, 30, 60):
        columns.append(np.sqrt(np.maximum(trailing_mean(returns**2, window), 0)))
    # Match the original encode-to-FP32 boundary before reconstructing volume.
    reconstructed_volume = np.expm1(encoded[:, 4].astype(np.float64) * 10)
    columns.extend(
        (
            np.log1p(reconstructed_volume) - np.log1p(trailing_mean(reconstructed_volume, 60)),
            trailing_mean((returns > 0).astype(np.float64), 60),
        )
    )
    features = np.concatenate((encoded, np.column_stack(columns), market[1:]), axis=1).astype(np.float32)
    return features[-context_length:], float(close[-1])
