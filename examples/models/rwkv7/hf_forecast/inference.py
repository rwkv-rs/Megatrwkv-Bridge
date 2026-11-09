# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a published RWKV7 quantile forecaster on raw CSV or encoded NPZ input."""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from processing_rwkv7_forecast import prepare_features
from transformers import AutoModel


logger = logging.getLogger(__name__)


def main() -> None:
    """Load one history window and write all horizon/quantile predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Caizus/rwkv7-ashare-1min-quantile-21m")
    parser.add_argument("--revision", help="Immutable Hugging Face model commit")
    parser.add_argument("--input", type=Path, required=True, help="Raw CSV or unnormalized features/anchor_price NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = (
        AutoModel.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        .to(args.device)
        .eval()
    )
    origin = None
    if args.input.suffix == ".npz":
        with np.load(args.input, allow_pickle=False) as sample:
            features = np.asarray(sample["features"], dtype=np.float32)
            anchor = float(sample["anchor_price"].item())
    else:
        with args.input.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        timestamps = np.array([row["timestamp"] for row in rows], dtype="datetime64[us]")
        bars = np.array([[float(row[key]) for key in ("open", "high", "low", "close", "volume")] for row in rows])
        market = np.array(
            [
                [float(row[key]) for key in ("market_return", "market_volatility", "market_up_fraction")]
                for row in rows
            ],
            dtype=np.float32,
        )
        features, anchor = prepare_features(bars, timestamps, market)
        origin = timestamps[-1]
    if not np.isfinite(features).all() or not np.isfinite(anchor) or anchor <= 0:
        raise ValueError("Input features and anchor price must be finite, with a positive anchor price.")
    with torch.inference_mode():
        result = model(
            features=torch.from_numpy(features).unsqueeze(0).to(args.device),
            anchor_price=torch.tensor([anchor], dtype=torch.float64, device=args.device),
        )
    raw, ordered, prices = (
        x[0].cpu().numpy() for x in (result.raw_quantiles, result.quantiles, result.price_quantiles)
    )
    levels = model.config.backbone_config["forecast_quantiles"]
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("horizon_minutes", "timestamp", "quantile", "raw_cumulative_return", "cumulative_return", "close_price")
        )
        for horizon in range(len(raw)):
            stamp = str(origin + np.timedelta64(horizon + 1, "m")) if origin is not None else ""
            for index, level in enumerate(levels):
                writer.writerow(
                    (horizon + 1, stamp, level, raw[horizon, index], ordered[horizon, index], prices[horizon, index])
                )
    logger.info("Saved %d horizons x %d quantiles to %s", len(raw), len(levels), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
