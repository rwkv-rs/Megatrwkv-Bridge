# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a selected forecast checkpoint as a standalone Transformers model."""

import argparse
import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from .configuration_rwkv7_forecast import RWKV7ForecastConfig
from .modeling_rwkv7_forecast import RWKV7ForQuantileForecasting


logger = logging.getLogger(__name__)


def main() -> None:
    """Write safetensors, AutoClass code, synthetic example and public metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    backbone = payload["run"]["hf_config"]
    if backbone.get("forecast_task") != "direct_quantile":
        raise ValueError("Export requires a direct_quantile checkpoint.")
    args.output.mkdir(parents=True, exist_ok=False)
    RWKV7ForecastConfig.register_for_auto_class("AutoConfig")
    RWKV7ForQuantileForecasting.register_for_auto_class("AutoModel")
    config = RWKV7ForecastConfig(backbone_config=backbone)
    model = RWKV7ForQuantileForecasting(config).to(dtype=torch.bfloat16).eval()
    state = dict(payload["state_dict"])
    # Persist derived constants so Transformers' meta-device loading restores
    # them just like parameters, rather than materializing uninitialized buffers.
    state.update(feature_mean=model.feature_mean.clone(), feature_std=model.feature_std.clone())
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        if value.dtype != state[name].dtype or not torch.equal(value, state[name]):
            raise AssertionError(f"Export changed tensor {name}")
    model.save_pretrained(args.output, safe_serialization=True)
    directory = Path(__file__).resolve().parent
    for name in ("README.md", "requirements.txt", "inference.py", "processing_rwkv7_forecast.py"):
        shutil.copy2(directory / name, args.output / name)
    shutil.copy2(directory.parents[3] / "LICENSE", args.output / "LICENSE")
    summary = json.loads(args.metrics.read_text())
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "selected_step": payload["step"],
        "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "parameter_count": sum(t.numel() for t in model.parameters()),
        "tensor_count": len(model.state_dict()),
        "source_tensor_count": len(payload["state_dict"]),
        "training_code_revision": "0fdda583a44ad54f823e0f1c872c94652ac36b25",
        "transformers_rwkv_revision": "f93dd4e01e0b565ef0390f76c6726d1d121ebbab",
        "data_manifest_sha256": payload["run"]["manifest_sha256"],
        "selected_checkpoint_criterion": "Earliest validation raw-pinball minimum; median validation winner is the same step.",
        "source_training_steps": 3592,
        "continuation_stopped_at_step": 15231,
        "continuation_outcome": "No validation improvement; original step-500 selection retained.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # A complete synthetic morning prefix makes the CLI runnable without data downloads.
    times = np.arange(np.datetime64("2026-11-06T09:30"), np.datetime64("2026-11-06T10:35"), np.timedelta64(1, "m"))
    position = np.arange(len(times), dtype=np.float64)
    close = 10 * np.exp(0.0002 * np.sin(position / 7))
    opening = np.concatenate((close[:1], close[:-1]))
    bars = np.column_stack(
        (opening, np.maximum(opening, close) + 0.01, np.minimum(opening, close) - 0.01, close, 10000 + position * 20)
    )
    market = np.column_stack(
        (0.002 * np.sin(position / 9), np.full(len(times), 0.15), np.full(len(times), 0.5))
    ).astype(np.float32)
    with (args.output / "example_bars.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "market_return",
                "market_volatility",
                "market_up_fraction",
            )
        )
        for stamp, row, aggregate in zip(times, bars, market):
            writer.writerow((str(stamp), *row, *aggregate))
    from .processing_rwkv7_forecast import prepare_features

    features, anchor = prepare_features(bars, times, market)
    np.savez(args.output / "example_inputs.npz", features=features, anchor_price=np.array(anchor))
    logger.info(
        "Exported %d exact tensors and %d parameters to %s",
        len(model.state_dict()),
        provenance["parameter_count"],
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
