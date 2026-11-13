# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Megatron indexed (``.bin/.idx``) storage for direct-quantile training windows.

Random windows over per-instrument memory maps read several files per window
and trigger page readahead on each, which bounds training throughput. Packing
stores each window-bearing segment as one contiguous float32 sequence of the
Megatron :class:`~megatron.core.datasets.indexed_dataset.IndexedDataset`, so a
window is two small positioned reads from one file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from numpy.typing import NDArray

from megatron.bridge.data.datasets.ohlcv import FEATURE_NAMES, PRICE_SCALE, SPLITS, manifest_digest


if TYPE_CHECKING:
    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset


logger = logging.getLogger(__name__)


def packed_identity(dataset: OHLCVQuantileWindowDataset) -> dict[str, Any]:
    """Describe the inventory, inputs and caches that determine packed windows."""
    if dataset.stride != 1 or dataset.cycle_length is not None or dataset.selection is not None:
        raise ValueError("Packing requires the complete stride-one window inventory.")
    columns = dataset.feature_columns
    if columns is None:
        if dataset.covariates is None:
            columns = list(range(len(FEATURE_NAMES)))
        else:
            from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES

            columns = list(range(len(FORECAST_FEATURE_NAMES)))
    return {
        "version": 1,
        "split": dataset.split,
        "source_manifest_sha256": manifest_digest(dataset.root),
        "covariates_sha256": manifest_digest(dataset.covariates) if dataset.covariates else None,
        "pvw_cache_sha256": manifest_digest(dataset.pvw_cache) if dataset.pvw_cache else None,
        "columns": list(columns),
        "context": dataset.seq_length,
        "horizon": dataset.horizon,
        "windows": dataset.num_windows,
    }


def _instrument_sequences(
    path: str,
    *,
    root: Path,
    covariates: Path | None,
    pvw_cache: Path | None,
    split_index: int,
    columns: list[int],
    context: int,
    horizon: int,
) -> tuple[NDArray[np.float32], list[int]]:
    """Return one instrument's concatenated segment sequences and their lengths."""
    directory = root / path
    segments = np.load(directory / "segments.npy", allow_pickle=False)
    features = np.load((covariates or root) / path / "features.npy", mmap_mode="r", allow_pickle=False)
    close = np.load(directory / "close.npy", mmap_mode="r", allow_pickle=False)
    pvw = (
        np.load(pvw_cache / "train" / path / "values.npy", mmap_mode="r", allow_pickle=False)
        if pvw_cache is not None
        else None
    )
    parts, lengths, consumed = [], [], 0
    for offset, length, split in segments:
        windows = max(0, int(length) - context - horizon + 1)
        if split != split_index or not windows:
            continue
        rows = np.empty((length, len(columns) + 1), dtype=np.float32)
        rows[:, :-1] = features[offset : offset + length][:, columns]
        segment_close = close[offset : offset + length]
        # Labels are differences of this column: 100 * log(close / anchor).
        rows[:, -1] = PRICE_SCALE * np.log(segment_close / segment_close[0])
        parts.append(rows.ravel())
        lengths.append(rows.size)
        if pvw is not None:
            parts.append(np.asarray(pvw[consumed : consumed + windows], dtype=np.float32).ravel())
            lengths[-1] += windows * context
        consumed += windows
    if pvw is not None and consumed != len(pvw):
        raise ValueError(f"PVW shard {path} does not match its training windows.")
    return (np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)), lengths


def pack_quantile_windows(dataset: OHLCVQuantileWindowDataset, output: Path, *, workers: int) -> None:
    """Write ``output/train.{bin,idx}`` and ``manifest.json`` in raw window order.

    Each sequence is one segment with at least one window: its ``[bars, columns + 1]``
    rows, whose last column is ``100 * log(close / first segment close)``, followed by
    its ``[windows, context]`` PVW rows when the dataset has a PVW cache. Each
    instrument is one document. Raw window ``i`` of ``dataset`` is packed window ``i``.
    """
    identity = packed_identity(dataset)
    if dataset.pvw_cache is not None and dataset.split != "train":
        raise ValueError("Only the training split has a complete per-window PVW cache.")
    output.mkdir(parents=True, exist_ok=False)
    builder = IndexedDatasetBuilder(str(output / "train.bin"), dtype=np.float32)
    operation = partial(
        _instrument_sequences,
        root=dataset.root,
        covariates=dataset.covariates,
        pvw_cache=dataset.pvw_cache,
        split_index=SPLITS.index(dataset.split),
        columns=identity["columns"],
        context=dataset.seq_length,
        horizon=dataset.horizon,
    )
    paths = [instrument["path"] for instrument in dataset.instruments]
    windows = 0

    def write(values: NDArray[np.float32], lengths: list[int]) -> None:
        nonlocal windows
        builder.add_document(values, lengths)
        width = len(identity["columns"]) + 1 + (dataset.seq_length if dataset.pvw_cache else 0)
        offset = dataset.seq_length * (dataset.seq_length + dataset.horizon - 1) if dataset.pvw_cache else 0
        bars = (np.asarray(lengths) + offset) // width
        windows += int((bars - dataset.seq_length - dataset.horizon + 1).sum())

    if workers == 1:
        for path in paths:
            write(*operation(path))
    else:
        # Bound in-flight shards; each instrument can hold hundreds of megabytes.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending: deque[Any] = deque()
            for number, path in enumerate(paths, start=1):
                pending.append(executor.submit(operation, path))
                if len(pending) >= 2 * workers:
                    write(*pending.popleft().result())
                if number % 250 == 0:
                    logger.info("Packed %d/%d instruments", number, len(paths))
            while pending:
                write(*pending.popleft().result())
    if windows != identity["windows"]:
        raise AssertionError(f"Packed {windows} windows, expected {identity['windows']}.")
    builder.finalize(str(output / "train.idx"))
    with (output / "train.idx").open("rb") as stream:
        index_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {"identity": identity, "pvw": dataset.pvw_cache is not None, "index_sha256": index_sha256}, indent=2
        )
        + "\n"
    )
    logger.info("Packed %d %s windows into %s", windows, dataset.split, output)


class PackedQuantileWindows:
    """Read features and cumulative close-return labels from packed segments."""

    def __init__(self, directory: Path) -> None:
        self.metadata = json.loads((directory / "manifest.json").read_text())
        identity = self.metadata["identity"]
        self.context, self.horizon = identity["context"], identity["horizon"]
        self.channels = len(identity["columns"]) + 1
        self.pvw = self.metadata["pvw"]
        # Positioned file reads avoid mmap readahead for scattered windows.
        self.data = IndexedDataset(str(directory / "train"), mmap=False)
        lengths = self.data.sequence_lengths.astype(np.int64)
        width = self.channels + (self.context if self.pvw else 0)
        offset = self.context * (self.context + self.horizon - 1) if self.pvw else 0
        self.bars = (lengths + offset) // width
        windows = self.bars - self.context - self.horizon + 1
        self.cumulative = np.cumsum(windows)
        if int(self.cumulative[-1]) != identity["windows"]:
            raise ValueError("Packed index does not contain the recorded window count.")

    def read(self, raw_indices: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return ``[n, context, inputs]`` features and ``[n, horizon]`` labels."""
        sequences = np.searchsorted(self.cumulative, raw_indices, side="right")
        local = raw_indices - np.where(sequences > 0, self.cumulative[sequences - 1], 0)
        inputs = self.channels - 1 + self.pvw
        features = np.empty((len(raw_indices), self.context, inputs), dtype=np.float32)
        labels = np.empty((len(raw_indices), self.horizon), dtype=np.float32)
        span = self.context + self.horizon
        for row, (sequence, start) in enumerate(zip(sequences.tolist(), local.tolist(), strict=True)):
            values = self.data.get(sequence, offset=start * self.channels, length=span * self.channels)
            values = values.reshape(span, self.channels)
            features[row, :, : self.channels - 1] = values[: self.context, :-1]
            log_close = values[:, -1].astype(np.float64)
            labels[row] = log_close[self.context :] - log_close[self.context - 1]
            if self.pvw:
                offset = int(self.bars[sequence]) * self.channels + start * self.context
                features[row, :, -1] = self.data.get(sequence, offset=offset, length=self.context)
        return features, labels
