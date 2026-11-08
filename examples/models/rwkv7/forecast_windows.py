# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze month/time-stratified forecast origins independently of future labels."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict

import numpy as np

from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset, manifest_digest


logger = logging.getLogger(__name__)


def select_windows(args: argparse.Namespace) -> None:
    """Select raw stride-one indices, stratifying by month and intraday minute."""
    dataset = OHLCVQuantileWindowDataset(
        args.data, split=args.split, seq_length=args.context, horizon=args.horizon, stride=1
    )
    excluded = set()
    if args.exclude_dates:
        excluded = {r["context_end"][:10] for r in json.loads(args.exclude_dates.read_text())}
    rng = np.random.default_rng(args.seed)
    candidates = rng.choice(
        dataset.num_windows, min(dataset.num_windows, args.count * (64 if excluded else 4)), replace=False
    )
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index in np.sort(candidates):
        info, start, arrays = dataset.window(int(index))
        end = start + args.context
        stamp = str(arrays["timestamps"][end - 1])
        if stamp[:10] in excluded:
            continue
        buckets[stamp[:7] + "/" + stamp[11:16]].append(
            {
                "index": int(index),
                "window_stride": 1,
                "exchange": info["exchange"],
                "symbol": info["symbol"],
                "context_end": stamp,
                "timestamps": [str(t) for t in arrays["timestamps"][end : end + args.horizon]],
            }
        )
    for values in buckets.values():
        rng.shuffle(values)
    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[dict[str, object]] = []
    while len(selected) < args.count and keys:
        remaining = []
        for key in keys:
            selected.append(buckets[key].pop())
            if buckets[key]:
                remaining.append(key)
            if len(selected) == args.count:
                break
        keys = remaining
    if len(selected) != args.count:
        raise ValueError(f"Only {len(selected)} eligible candidates for {args.count} requested origins.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), np.asarray([r["index"] for r in selected], dtype=np.int64))
    args.output.with_suffix(".json").write_text(json.dumps(selected, indent=2) + "\n")
    report = {
        "manifest_sha256": manifest_digest(args.data),
        "split": args.split,
        "seed": args.seed,
        "context": args.context,
        "horizon": args.horizon,
        "stride": 1,
        "windows": len(selected),
        "selection": "seeded candidate pool; round-robin month/intraday-minute strata; no future label selection",
        "excluded_dates": sorted(excluded),
        "date_range": [
            min(str(r["context_end"])[:10] for r in selected),
            max(str(r["context_end"])[:10] for r in selected),
        ],
        "origin_time_counts": dict(sorted(Counter(str(r["context_end"])[11:16] for r in selected).items())),
    }
    args.output.with_suffix(".selection.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Frozen selection: %s", report)
