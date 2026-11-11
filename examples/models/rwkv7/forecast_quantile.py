# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate direct close-return quantiles on frozen historical forecast origins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset, manifest_digest


if TYPE_CHECKING:
    from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import RWKV7ForecastModel
    from megatron.bridge.recipes.rwkv7.forecast import ForecastCheckpoint

logger = logging.getLogger(__name__)


def quantile_metrics(target: NDArray[Any], prediction: NDArray[Any], levels: NDArray[Any]) -> dict[str, Any]:
    """Score marginal log-return quantiles, including proper interval scores."""
    error = target[..., None] - prediction
    pinball = np.maximum(levels * error, (levels - 1) * error) * 100  # scaled return -> bp
    result: dict[str, Any] = {
        "pinball_bps": float(pinball.mean()),
        "pinball_bps_by_horizon": pinball.mean((0, 2)).tolist(),
        "pinball_bps_by_quantile": pinball.mean((0, 1)).tolist(),
        "cdf_strict_by_horizon_quantile": (target[..., None] < prediction).mean(0).tolist(),
        "cdf_weak_by_horizon_quantile": (target[..., None] <= prediction).mean(0).tolist(),
        "discrete_calibration_violation_by_quantile": (
            np.maximum((target[..., None] < prediction).mean((0, 1)) - levels, 0)
            + np.maximum(levels - (target[..., None] <= prediction).mean((0, 1)), 0)
        ).tolist(),
        "calibration_by_horizon_quantile": (target[..., None] <= prediction).mean(0).tolist(),
        "calibration_error_by_quantile": ((target[..., None] <= prediction).mean((0, 1)) - levels).tolist(),
    }
    for lower, upper in ((0.1, 0.9), (0.2, 0.8)):
        lo = prediction[..., int(np.argmin(abs(levels - lower)))]
        hi = prediction[..., int(np.argmin(abs(levels - upper)))]
        alpha = 1 - (upper - lower)
        width = hi - lo
        score = width + 2 / alpha * (np.maximum(lo - target, 0) + np.maximum(target - hi, 0))
        result[f"interval_{round((upper - lower) * 100)}"] = {
            "coverage": float(((target >= lo) & (target <= hi)).mean()),
            "coverage_by_horizon": ((target >= lo) & (target <= hi)).mean(0).tolist(),
            "width_bps": float(width.mean() * 100),
            "width_bps_by_horizon": (width.mean(0) * 100).tolist(),
            "interval_score_bps": float(score.mean() * 100),
            "interval_score_bps_by_horizon": (score.mean(0) * 100).tolist(),
        }
    return result


def point_metrics(actual: NDArray[Any], predicted: NDArray[Any], anchors: NDArray[Any]) -> dict[str, Any]:
    """Report price and cumulative-return errors in original and shared units."""
    error = predicted - actual
    relative = error / anchors[:, None] * 100
    returns = np.log(predicted / actual) * 10000
    actual_returns = np.log(actual / anchors[:, None]) * 10000
    predicted_returns = np.log(predicted / anchors[:, None]) * 10000
    actual_centered = actual_returns - actual_returns.mean(0)
    predicted_centered = predicted_returns - predicted_returns.mean(0)
    denominator = np.sqrt((actual_centered**2).sum(0) * (predicted_centered**2).sum(0))
    correlations = [
        float((actual_centered[:, h] * predicted_centered[:, h]).sum() / value) if value > 0 else None
        for h, value in enumerate(denominator)
    ]
    return {
        "return_correlation_by_horizon": correlations,
        "predicted_return_mean_bps_by_horizon": predicted_returns.mean(0).tolist(),
        "actual_return_mean_bps_by_horizon": actual_returns.mean(0).tolist(),
        "mae_cny_by_horizon": np.abs(error).mean(0).tolist(),
        "rmse_cny_by_horizon": np.sqrt(np.square(error).mean(0)).tolist(),
        "bias_cny_by_horizon": error.mean(0).tolist(),
        "mae_pct_by_horizon": np.abs(relative).mean(0).tolist(),
        "rmse_pct_by_horizon": np.sqrt(np.square(relative).mean(0)).tolist(),
        "bias_pct_by_horizon": relative.mean(0).tolist(),
        "return_mae_bps_by_horizon": np.abs(returns).mean(0).tolist(),
        "return_rmse_bps_by_horizon": np.sqrt(np.square(returns).mean(0)).tolist(),
        "return_bias_bps_by_horizon": returns.mean(0).tolist(),
        "mean_path_mae_pct": float(np.abs(relative).mean()),
    }


def evaluate_quantiles(args: argparse.Namespace, model: RWKV7ForecastModel, payload: ForecastCheckpoint) -> None:
    """Save all predictions, baselines, grouped scores and deterministic figures."""
    if manifest_digest(args.data) != payload["run"]["manifest_sha256"]:
        raise ValueError("Evaluation must use the training dataset manifest.")
    if args.horizon != model.horizon:
        raise ValueError("Evaluation horizon must equal the trained forecast horizon.")
    context, horizon = payload["run"]["hf_config"]["context_length"], model.horizon
    windows = json.loads(args.windows.read_text()) if args.windows else []
    stride = int(windows[0].get("window_stride", horizon)) if windows else 1
    covariates = getattr(args, "covariates", None)
    expected_digest = payload["run"].get("covariates_sha256")
    if expected_digest:
        if covariates is None:
            covariates = Path(payload["run"]["covariates"])
        with (covariates / "manifest.json").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_digest:
                raise ValueError("Covariate identity differs from the checkpoint.")
    elif covariates is not None:
        raise ValueError("This checkpoint does not use covariates.")
    pvw_cache = getattr(args, "pvw_cache", None)
    if expected_pvw := payload["run"].get("pvw_cache_sha256"):
        pvw_cache = pvw_cache or Path(payload["run"]["pvw_cache"])
        if manifest_digest(pvw_cache) != expected_pvw:
            raise ValueError("PVW cache identity differs from the checkpoint.")
    elif pvw_cache is not None:
        raise ValueError("This checkpoint does not use a PVW cache.")
    dataset = OHLCVQuantileWindowDataset(
        args.data,
        split=args.split,
        seq_length=context,
        horizon=horizon,
        stride=stride,
        covariates=covariates,
        pvw_cache=pvw_cache,
    )
    if not args.windows:
        indices = np.linspace(0, len(dataset) - 1, min(args.max_windows, len(dataset)), dtype=np.int64)
        windows = [{"index": int(i), "window_stride": stride} for i in indices]
    # Freeze illustrative examples before predicting any test outcome.
    fixed = np.random.default_rng(payload["run"]["seed"]).choice(len(windows), min(8, len(windows)), replace=False)
    inputs_list, anchors, actual, histories, times, identities = [], [], [], [], [], []
    for row in windows:
        info, start, arrays = dataset.window(row["index"])
        end = start + context
        identity = {
            "exchange": info["exchange"],
            "symbol": info["symbol"],
            "context_end": str(arrays["timestamps"][end - 1]),
        }
        future_times = [str(t) for t in arrays["timestamps"][end : end + horizon]]
        if args.windows and (any(row[k] != v for k, v in identity.items()) or row["timestamps"] != future_times):
            raise ValueError("Frozen evaluation window no longer matches dataset identity and timestamps.")
        if not args.windows:
            row.update(identity)
            row["timestamps"] = future_times
        inputs_list.append(np.array(arrays["features"][start:end]))
        anchors.append(float(arrays["close"][end - 1]))
        actual.append(np.array(arrays["close"][end : end + horizon]))
        histories.append(np.array(arrays["close"][start:end]))
        times.append([str(t) for t in arrays["timestamps"][start : end + horizon]])
        identities.append(identity)
    anchor = np.asarray(anchors)
    truth = np.asarray(actual)
    if pvw_cache is not None:
        samples = dataset.__getitems__([row["index"] for row in windows])
        all_inputs = np.stack([sample["features"] for sample in samples])
    else:
        all_inputs = np.asarray(inputs_list)
    raw = []
    for begin in range(0, len(all_inputs), 256):
        inputs = torch.from_numpy(all_inputs[begin : begin + 256]).cuda()
        raw.append(model.predict_quantiles(inputs, fix_quantile_crossing=False).cpu().numpy())
    raw_array = np.concatenate(raw).astype(np.float64)
    ordered = np.sort(raw_array, axis=-1)
    levels = np.asarray(model.quantile_levels)
    target = 100 * np.log(truth / anchor[:, None])
    empirical = np.broadcast_to(np.asarray(payload["run"]["hf_config"]["forecast_target_quantiles"]), ordered.shape)
    prices = anchor[:, None, None] * np.exp(ordered / 100)
    median = prices[..., int(np.argmin(abs(levels - 0.5)))]
    persistence = np.broadcast_to(anchor[:, None], truth.shape)
    point = {
        "direct_quantile": point_metrics(truth, median, anchor),
        "persistence": point_metrics(truth, persistence, anchor),
    }
    if args.baseline_paths:
        previous = np.load(args.baseline_paths, allow_pickle=False)
        if previous["baseline"].shape[:2] != truth.shape:
            raise ValueError("Baseline forecast shape does not match the frozen windows.")
        np.testing.assert_allclose(previous["anchors"], anchor, rtol=0, atol=0)
        np.testing.assert_allclose(previous["actual"][..., 3], truth, rtol=1e-6, atol=1e-6)
        point["autoregressive"] = point_metrics(truth, previous["baseline"][..., 3], anchor)
    probabilistic_baselines = {}
    calibration_path = args.checkpoint.parent / "calibration.npz"
    if calibration_path.exists():
        calibration = np.load(calibration_path, allow_pickle=False)
        x_train, y_train = calibration["features"].astype(float), calibration["targets"].astype(float)
        # An interpretable conditional reference: historical volatility quartile x session.
        if covariates is not None:
            from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES

            vol_column = FORECAST_FEATURE_NAMES.index("volatility_60")
            session_column = FORECAST_FEATURE_NAMES.index("afternoon")
            boundaries = np.quantile(x_train[:, vol_column], [0.25, 0.5, 0.75])
            train_group = np.searchsorted(boundaries, x_train[:, vol_column]) + 4 * x_train[:, session_column].astype(
                int
            )
            test_group = np.searchsorted(boundaries, all_inputs[:, -1, vol_column]) + 4 * all_inputs[
                :, -1, session_column
            ].astype(int)
            group_quantiles = np.stack(
                [np.quantile(y_train[train_group == group], levels, axis=0).T for group in range(8)]
            )
            probabilistic_baselines["volatility_session_empirical"] = group_quantiles[test_group]
        mean, scale = x_train.mean(0), x_train.std(0)
        scale[scale == 0] = 1
        train_design = np.column_stack((np.ones(len(x_train)), (x_train - mean) / scale))
        test_design = np.column_stack((np.ones(len(all_inputs)), (all_inputs[:, -1] - mean) / scale))
        penalty = np.eye(train_design.shape[1]) * len(x_train)  # fixed ridge alpha=1, no test tuning
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(train_design.T @ train_design + penalty, train_design.T @ y_train)
        ridge_returns = test_design @ coefficients
        ridge_prices = anchor[:, None] * np.exp(ridge_returns / 100)
        point["ridge_alpha1"] = point_metrics(truth, ridge_prices, anchor)
    grouped = {}
    symbols = np.asarray([r["exchange"] + r["symbol"] for r in identities])
    dates = np.asarray([r["context_end"][:10] for r in identities])
    improvement = (np.abs(median - truth) - np.abs(persistence - truth)) / anchor[:, None] * 100
    for name, keys in (
        ("symbol", symbols),
        ("date", dates),
        ("month", np.asarray([date[:7] for date in dates])),
        ("origin_minute", np.asarray([r["context_end"][11:16] for r in identities])),
    ):
        grouped[name] = {
            key: {
                "windows": int((keys == key).sum()),
                "point": point_metrics(truth[keys == key], median[keys == key], anchor[keys == key]),
                "persistence_point": point_metrics(truth[keys == key], persistence[keys == key], anchor[keys == key]),
                "probabilistic": quantile_metrics(target[keys == key], ordered[keys == key], levels),
            }
            for key in np.unique(keys)
        }
    model_error = target[..., None] - ordered
    empirical_error = target[..., None] - empirical
    pinball_difference = (
        np.maximum(levels * model_error, (levels - 1) * model_error)
        - np.maximum(levels * empirical_error, (levels - 1) * empirical_error)
    ).mean(-1) * 100
    differences = np.column_stack(
        (improvement.mean(-1), improvement[:, -1], pinball_difference.mean(-1), pinball_difference[:, -1])
    )
    difference_columns = [
        "median_minus_persistence_path_mae_pct",
        "median_minus_persistence_h30_mae_pct",
        "model_minus_empirical_path_pinball_bp",
        "model_minus_empirical_h30_pinball_bp",
    ]
    for key, reference in probabilistic_baselines.items():
        reference_error = target[..., None] - reference
        reference_difference = (
            np.maximum(levels * model_error, (levels - 1) * model_error)
            - np.maximum(levels * reference_error, (levels - 1) * reference_error)
        ).mean(-1) * 100
        differences = np.column_stack((differences, reference_difference.mean(-1), reference_difference[:, -1]))
        difference_columns.extend((f"model_minus_{key}_path_pinball_bp", f"model_minus_{key}_h30_pinball_bp"))
    confidence_by_group = {}
    for grouping, keys in (("symbol", symbols), ("date", dates)):
        _, inverse = np.unique(keys, return_inverse=True)
        sums = np.zeros((inverse.max() + 1, differences.shape[1]))
        np.add.at(sums, inverse, differences)
        counts = np.bincount(inverse)
        rng = np.random.default_rng(payload["run"]["seed"])
        boot = []
        for _ in range(1000):
            selected = rng.integers(len(counts), size=len(counts))
            boot.append(sums[selected].sum(0) / counts[selected].sum())
        confidence_by_group[grouping] = np.quantile(boot, [0.025, 0.975], axis=0).tolist()
    confidence = np.asarray(confidence_by_group["symbol"])
    with args.checkpoint.open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "split": args.split,
        "windows": len(windows),
        "symbols": len(set(symbols)),
        "dates": len(set(dates)),
        "context_end_range": [min(r["context_end"] for r in identities), max(r["context_end"] for r in identities)],
        "horizon": horizon,
        "checkpoint_step": payload["step"],
        "manifest_sha256": payload["run"]["manifest_sha256"],
        "checkpoint_sha256": checkpoint_digest,
        "window_selection": "frozen window identities"
        if args.windows
        else "evenly spaced across stride-one inventory",
        "window_stride": stride,
        "origin_time_counts": {
            key: int(value)
            for key, value in zip(*np.unique([r["context_end"][11:16] for r in identities], return_counts=True))
        },
        "raw_crossing_fraction": float((np.diff(raw_array, axis=-1) < 0).any(-1).mean()),
        "point": point,
        "probabilistic": {
            "raw": {
                key: value
                for key, value in quantile_metrics(target, raw_array, levels).items()
                if not key.startswith("interval_")
            },
            "sorted": quantile_metrics(target, ordered, levels),
            "empirical": quantile_metrics(target, empirical, levels),
            **{key: quantile_metrics(target, values, levels) for key, values in probabilistic_baselines.items()},
        },
        "mae_difference_vs_persistence_pct_bootstrap_95ci_path_and_h30": confidence[:, :2].tolist(),
        "pinball_difference_vs_empirical_bps_bootstrap_95ci_path_and_h30": confidence[:, 2:4].tolist(),
        "bootstrap_difference_columns": difference_columns,
        "paired_difference_bootstrap_95ci_by_group": confidence_by_group,
        "grouped": grouped,
        "fixed_examples": fixed.tolist(),
    }
    destination = args.output.parent
    destination.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (destination / "windows.json").write_text(json.dumps(windows, indent=2) + "\n")
    if getattr(args, "metrics_only", False):
        logger.info("Evaluated %d windows at checkpoint %d", len(windows), payload["step"])
        return
    np.savez(
        destination / "paths.npz",
        raw=raw_array,
        quantiles=ordered,
        prices=prices,
        actual=truth,
        anchors=anchor,
        history=np.asarray(histories),
        timestamps=np.asarray(times),
        levels=levels,
    )
    with (destination / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "exchange",
                "symbol",
                "context_end",
                "timestamp",
                "horizon",
                "quantile",
                "actual_close",
                "predicted_close",
                "raw_return",
                "return",
            )
        )
        for n, identity in enumerate(identities):
            for h in range(horizon):
                for q, level in enumerate(levels):
                    writer.writerow(
                        (
                            *identity.values(),
                            times[n][context + h],
                            h + 1,
                            level,
                            truth[n, h],
                            prices[n, h, q],
                            raw_array[n, h, q],
                            ordered[n, h, q],
                        )
                    )
    render_quantiles(destination, report, np.asarray(histories), truth, prices, np.asarray(times), identities, levels)
    logger.info("Evaluated %d frozen direct-quantile windows at checkpoint %d", len(windows), payload["step"])


def render_quantiles(
    destination: Path,
    report: dict[str, Any],
    history: NDArray[Any],
    actual: NDArray[Any],
    prices: NDArray[Any],
    times: NDArray[Any],
    identities: list[dict[str, str]],
    levels: NDArray[Any],
) -> None:
    """Render price curves and proper-score diagnostics with the existing Pillow runtime."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=17)
    heading = ImageFont.load_default(size=23)
    colors = ("#e27722", "#64748b", "#2c7fb8", "#238b45")
    median = prices[..., int(np.argmin(abs(levels - 0.5)))]
    anchors = history[:, -1]
    errors = (np.abs(median - actual) / anchors[:, None]).mean(-1)
    ranked = [int(np.argmin(abs(errors - np.quantile(errors, p)))) for p in (0.1, 0.5, 0.9)]

    def panel(
        draw: Any,
        box: tuple[int, int, int, int],
        curves: list[tuple[str, str, NDArray[Any], NDArray[Any]]],
        title: str,
        ylabel: str,
        tick_labels: dict[float, str],
        bands: list[tuple[NDArray[Any], NDArray[Any], NDArray[Any], str]] | None = None,
        origin: int | None = None,
    ) -> None:
        x, y, width, height = box
        left, right, top, bottom = x + 95, x + width - 20, y + 48, y + height - 63
        values = [curve[3] for curve in curves]
        if bands:
            values.extend(v for _, lo, hi, _ in bands for v in (lo, hi))
        low, high = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
        pad = max((high - low) * 0.1, abs(high) * 1e-5, 1e-6)
        low, high = low - pad, high + pad
        xmin, xmax = min(float(c[2].min()) for c in curves), max(float(c[2].max()) for c in curves)

        def px(v: float) -> float:
            return left + (v - xmin) / (xmax - xmin) * (right - left)

        def py(v: float) -> float:
            return bottom - (v - low) / (high - low) * (bottom - top)

        draw.text((x + 15, y + 5), title, fill="#172b4d", font=font)
        draw.text((left, y + 27), ylabel, fill="#475569", font=font)
        decimals = max(0, min(8, int(np.ceil(-np.log10((high - low) / 5))) + 1))
        for j in range(6):
            v = low + (high - low) * j / 5
            yy = py(v)
            draw.line((left, yy, right, yy), fill="#e2e8f0")
            draw.text((x + 4, yy - 8), f"{v:.{decimals}f}", fill="#475569", font=font)
        if bands:
            for xx, lo, hi, color in bands:
                polygon = [(px(float(t)), py(float(v))) for t, v in zip(xx, lo)]
                polygon += [(px(float(t)), py(float(v))) for t, v in zip(xx[::-1], hi[::-1])]
                draw.polygon(polygon, fill=color)
        for _, color, xx, vv in curves:
            points = [(px(float(t)), py(float(v))) for t, v in zip(xx, vv)]
            draw.line(points, fill=color, width=3)
        if origin is not None:
            origin_x = px(origin)
            for yy in range(top, bottom, 12):
                draw.line((origin_x, yy, origin_x, yy + 6), fill="#475569", width=2)
        draw.line((left, top, left, bottom, right, bottom), fill="#94a3b8", width=2)
        for tick, label in tick_labels.items():
            if xmin <= tick <= xmax:
                draw.text((px(tick) - 23, bottom + 9), label, fill="#475569", font=font)
        for j, (label, color, _, _) in enumerate(curves):
            legend_x = left + (j % 3) * 205
            yy = bottom + 32 + (j // 3) * 21
            draw.line((legend_x, yy + 8, legend_x + 25, yy + 8), fill=color, width=3)
            draw.text((legend_x + 30, yy), label, fill="#172b4d", font=font)

    for title, indices in (("fixed", report["fixed_examples"]), ("error_percentiles", ranked)):
        image = Image.new("RGB", (1800, 85 + 395 * len(indices)), "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (25, 12),
            "RWKV7 direct 30-minute close forecast | actual vs median | 60% / 80% marginal intervals",
            fill="#172b4d",
            font=heading,
        )
        rule = (
            "Seeded windows chosen before inference"
            if title == "fixed"
            else "Examples nearest P10, P50, P90 of whole-path close MAE"
        )
        draw.text((25, 47), rule, fill="#475569", font=font)
        for row, i in enumerate(indices):
            origin = len(history[i]) - 1
            total = origin + 1 + actual.shape[1]
            for col in range(2):
                begin = origin if col else 0
                xx = np.arange(begin, total)
                future_x = np.arange(origin, total)
                observed = np.concatenate((history[i], actual[i]))[begin:]
                curves = [
                    ("Actual close", "#23364d", xx, observed),
                    ("Predicted median", colors[0], future_x, np.r_[anchors[i], median[i]]),
                    ("Last close", colors[1], future_x, np.full(len(future_x), anchors[i])),
                ]
                bands = [
                    (future_x, np.r_[anchors[i], prices[i, :, 0]], np.r_[anchors[i], prices[i, :, -1]], "#e1eff7"),
                    (future_x, np.r_[anchors[i], prices[i, :, 1]], np.r_[anchors[i], prices[i, :, -2]], "#b8dbea"),
                ]
                ticks = {float(j): str(times[i, j])[11:16] for j in np.linspace(begin, total - 1, 6, dtype=int)}
                identity = identities[i]
                label = f"{identity['exchange']} {identity['symbol']} | {identity['context_end'][:16]} | window {i}"
                panel(
                    draw, (10 + col * 900, 80 + row * 395, 880, 370), curves, label, "CNY/share", ticks, bands, origin
                )
        image.save(destination / f"{title}.png")
    image = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (25, 15),
        f"Frozen test windows: {report['windows']} | checkpoint step {report['checkpoint_step']}",
        fill="#172b4d",
        font=heading,
    )
    horizon = np.arange(1, actual.shape[1] + 1)
    ticks = {float(j): str(j) for j in (1, 5, 15, 30) if j <= len(horizon)}
    curves = [
        (name, colors[i], horizon, np.asarray(result["mae_pct_by_horizon"]))
        for i, (name, result) in enumerate(report["point"].items())
    ]
    panel(draw, (10, 70, 880, 410), curves, "Close MAE by forecast minute", "% origin", ticks)
    for position, key, label in (
        (1, "coverage_by_horizon", "Interval coverage"),
        (2, "width_bps_by_horizon", "Interval width (bp)"),
    ):
        curves = []
        for name in ("sorted", "empirical"):
            for nominal in (60, 80):
                result = report["probabilistic"][name][f"interval_{nominal}"]
                curves.append((f"{name} {nominal}%", colors[len(curves)], horizon, np.asarray(result[key])))
        if position == 1:
            curves += [
                ("nominal 60%", "#aaaaaa", horizon, np.full(len(horizon), 0.6)),
                ("nominal 80%", "#bbbbbb", horizon, np.full(len(horizon), 0.8)),
            ]
        panel(
            draw,
            (910 if position == 1 else 10, 70 if position == 1 else 550, 880, 410),
            curves,
            label,
            "fraction" if position == 1 else "bp",
            ticks,
        )
    curves = [
        (name, colors[i], levels, np.asarray(report["probabilistic"][name]["calibration_error_by_quantile"]) + levels)
        for i, name in enumerate(("raw", "sorted", "empirical"))
    ]
    curves.append(("ideal", "#aaaaaa", levels, levels))
    panel(draw, (910, 550, 880, 410), curves, "Quantile calibration (nominal vs observed)", "fraction", {})
    image.save(destination / "quality.png")
