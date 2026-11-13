# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local minute-bar preparation and memory-mapped next-bar windows."""

import bisect
import hashlib
import json
import logging
import math
import multiprocessing
import random
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Iterator, TypedDict

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
FEATURE_NAMES = ("open_return", "close_return", "upper_wick", "lower_wick", "log_volume")
SPLITS = ("train", "validation", "test")
PRICE_SCALE = 100.0
VOLUME_SCALE = 10.0
logger = logging.getLogger(__name__)


class InstrumentInfo(TypedDict):
    """Location and coverage of one prepared canonical instrument."""

    path: str
    exchange: str
    symbol: str
    bars: int
    start: str
    end: str


class SplitCounts(TypedDict):
    """Counts of retained encoded bars and contiguous segments."""

    segments: int
    bars: int


class OHLCVManifest(TypedDict):
    """Versioned feature, provenance and shard contract."""

    version: int
    source: str
    revision: str
    train_end: str
    validation_end: str
    feature_names: list[str]
    price_scale: float
    volume_scale: float
    instruments: list[InstrumentInfo]
    splits: dict[str, SplitCounts]


class WindowInfo(InstrumentInfo):
    """Instrument metadata with the selected sequence's bounds."""

    split: str
    offset: int
    length: int


class ShardArrays(TypedDict):
    """Memory maps for one instrument."""

    features: NDArray[np.float32]
    close: NDArray[np.float64]
    timestamps: NDArray[np.datetime64]
    segments: NDArray[np.int64]


def encode_bars(bars: NDArray[np.float64]) -> NDArray[np.float32]:
    """Encode a valid contiguous OHLCV series, dropping its first anchor bar.

    Prices are log ratios and wicks, multiplied by 100. Volume is log1p / 10.
    These fixed units require no fitted statistics or future observations.
    """
    previous = bars[:-1, 3]
    opening, high, low, close, volume = bars[1:].T
    return np.column_stack(
        (
            np.log(opening / previous) * PRICE_SCALE,
            np.log(close / previous) * PRICE_SCALE,
            np.log(high / np.maximum(opening, close)) * PRICE_SCALE,
            np.log(np.minimum(opening, close) / low) * PRICE_SCALE,
            np.log1p(volume) / VOLUME_SCALE,
        )
    ).astype(np.float32)


def decode_bars(features: NDArray[np.float32] | NDArray[np.float64], *, previous_close: float) -> NDArray[np.float64]:
    """Reconstruct a generated path in CNY/share and shares, in OHLCV order.

    Prices use FP64 while decoding cumulative returns. Volume remains a
    continuous point estimate. Nonfinite predictions fail rather than being
    silently clipped into a plausible-looking path.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 5 or not np.isfinite(features).all():
        raise ValueError("Predictions must be a finite [time, 5] array.")
    if not math.isfinite(previous_close) or previous_close <= 0 or (features[:, 2:] < 0).any():
        raise ValueError("Expected a positive anchor price and nonnegative wick/volume features.")
    with np.errstate(over="raise", invalid="raise", under="raise"):
        close = previous_close * np.exp(np.cumsum(features[:, 1] / PRICE_SCALE))
        previous = np.concatenate(([previous_close], close[:-1]))
        opening = previous * np.exp(features[:, 0] / PRICE_SCALE)
        high = np.maximum(opening, close) * np.exp(features[:, 2] / PRICE_SCALE)
        low = np.minimum(opening, close) * np.exp(-features[:, 3] / PRICE_SCALE)
        volume = np.expm1(features[:, 4] * VOLUME_SCALE)
    return np.column_stack((opening, high, low, close, volume))


def load_instrument(path: Path) -> tuple[str, str, NDArray[np.datetime64], NDArray[np.float64]]:
    """Read one canonical Parquet file, preserving leading zeroes in symbols."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(path).read(columns=["exchange", "symbol", "timestamp", *OHLCV_COLUMNS])
    exchanges = table["exchange"].unique().to_pylist()
    symbols = table["symbol"].unique().to_pylist()
    if (
        len(exchanges) != 1
        or exchanges[0] not in ("SH", "SZ", "BJ")
        or len(symbols) != 1
        or not isinstance(symbols[0], str)
        or len(symbols[0]) != 6
        or not symbols[0].isascii()
        or not symbols[0].isdigit()
    ):
        raise ValueError(f"Expected one instrument with a string symbol in {path}.")
    timestamps = table["timestamp"].to_numpy().astype("datetime64[us]")
    if np.isnat(timestamps).any() or (np.diff(timestamps) <= np.timedelta64(0, "us")).any():
        raise ValueError(f"Timestamps must be non-null, unique and increasing in {path}.")
    bars = np.column_stack([table[column].to_numpy() for column in OHLCV_COLUMNS]).astype(np.float64)
    return exchanges[0], symbols[0], timestamps, bars


def contiguous_segments(timestamps: NDArray[np.datetime64], bars: NDArray[np.float64]) -> list[tuple[int, int]]:
    """Find valid one-minute runs, cutting at invalid rows, missing minutes and overnight.

    The 11:30 bar and the 13:00/13:01 bar are consecutive trading minutes, so a
    run continues through the lunch break and covers the whole trading day.
    """
    if not len(timestamps):
        return []
    minutes = timestamps.astype("datetime64[m]").astype(np.int64) % (24 * 60)
    in_session = ((minutes >= 570) & (minutes <= 690)) | ((minutes >= 780) & (minutes <= 900))
    aligned = timestamps == timestamps.astype("datetime64[m]")
    opening, high, low, close, volume = bars.T
    valid = (
        np.isfinite(bars).all(axis=1)
        & (bars[:, :4] > 0).all(axis=1)
        & (volume >= 0)
        & (high >= np.maximum(opening, close))
        & (low <= np.minimum(opening, close))
        & in_session
        & aligned
    )
    step = np.diff(timestamps)
    lunch = (minutes[:-1] == 690) & (minutes[1:] >= 780) & (minutes[1:] <= 781) & (step <= np.timedelta64(91, "m"))
    linked = (step == np.timedelta64(1, "m")) | lunch
    starts = np.flatnonzero(valid & ~np.concatenate(([False], valid[:-1] & linked)))
    ends = np.flatnonzero(valid & ~np.concatenate((valid[1:] & linked, [False]))) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def prepare_ohlcv(
    source: Path,
    output: Path,
    *,
    train_end: str,
    validation_end: str,
    revision: str,
    workers: int = 1,
) -> OHLCVManifest:
    """Prepare all canonical instrument files into per-instrument mmap shards.

    Split boundaries are exclusive local dates. Invalid rows and discontinuities
    cut the series; no filling or adjustment is performed. The destination must
    be new, so a partially prepared or previously used dataset is never reused.
    """
    train_boundary, validation_boundary = np.datetime64(train_end, "D"), np.datetime64(validation_end, "D")
    if np.isnat(train_boundary) or np.isnat(validation_boundary) or train_boundary >= validation_boundary:
        raise ValueError("Expected train_end < validation_end as ISO dates.")
    if not revision.strip():
        raise ValueError("Record the source dataset revision.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    paths = sorted(source.glob("data/stock_1m/*/*.parquet"))
    if not paths:
        raise ValueError("Expected a snapshot root containing data/stock_1m/{exchange}/{symbol}.parquet.")
    output.mkdir(parents=True, exist_ok=False)
    manifest: OHLCVManifest = {
        # Version 2 segments continue through the lunch break.
        "version": 2,
        "source": "neigezhu/china-a-share-1min-ohlcv",
        "revision": revision,
        "train_end": train_end,
        "validation_end": validation_end,
        "feature_names": list(FEATURE_NAMES),
        "price_scale": PRICE_SCALE,
        "volume_scale": VOLUME_SCALE,
        "instruments": [],
        "splits": {split: {"segments": 0, "bars": 0} for split in SPLITS},
    }
    worker = partial(
        _prepare_instrument, output=output, train_boundary=train_boundary, validation_boundary=validation_boundary
    )
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else nullcontext()
    )
    with pool as executor:
        results = executor.map(worker, paths, chunksize=1) if executor is not None else map(worker, paths)
        for count, (instrument, splits) in enumerate(results, start=1):
            if instrument is not None:
                manifest["instruments"].append(instrument)
            for split in SPLITS:
                manifest["splits"][split]["segments"] += splits[split]["segments"]
                manifest["splits"][split]["bars"] += splits[split]["bars"]
            if count % 100 == 0 or count == len(paths):
                logger.info("Prepared %d/%d instruments", count, len(paths))
    if not manifest["instruments"]:
        raise ValueError("No valid contiguous bars found.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _prepare_instrument(
    path: Path, *, output: Path, train_boundary: np.datetime64, validation_boundary: np.datetime64
) -> tuple[InstrumentInfo | None, dict[str, SplitCounts]]:
    exchange, symbol, timestamps, bars = load_instrument(path)
    if exchange != path.parent.name or symbol != path.stem:
        raise ValueError(f"Canonical path does not match its instrument identity: {path}.")
    chunks, closes, times, segments = [], [], [], []
    splits: dict[str, SplitCounts] = {split: {"segments": 0, "bars": 0} for split in SPLITS}
    offset = 0
    # These indices are constant for all intraday segments in an instrument.
    split_indices = [int(np.searchsorted(timestamps, boundary)) for boundary in (train_boundary, validation_boundary)]
    for start, end in contiguous_segments(timestamps, bars):
        boundaries = [start, *(index for index in split_indices if start < index < end), end]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right - left < 3:
                continue
            features = encode_bars(bars[left:right])
            split_index = int(timestamps[left] >= train_boundary) + int(timestamps[left] >= validation_boundary)
            split = SPLITS[split_index]
            segments.append((offset, len(features), split_index))
            splits[split]["segments"] += 1
            splits[split]["bars"] += len(features)
            chunks.append(features)
            closes.append(bars[left + 1 : right, 3])
            times.append(timestamps[left + 1 : right])
            offset += len(features)
    if not chunks:
        return None, splits
    relative = f"{exchange}/{symbol}"
    directory = output / relative
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "features.npy", np.concatenate(chunks))
    np.save(directory / "close.npy", np.concatenate(closes))
    np.save(directory / "timestamps.npy", np.concatenate(times))
    np.save(directory / "segments.npy", np.asarray(segments, dtype=np.int64))
    return InstrumentInfo(
        path=relative, exchange=exchange, symbol=symbol, bars=offset, start=str(times[0][0]), end=str(times[-1][-1])
    ), splits


class OHLCVWindowDataset(Dataset[dict[str, NDArray[np.float32]]]):
    """Read windows lazily without materializing billions of window indices.

    ``target_length`` repeats a deterministic traversal to satisfy Bridge's
    sample budget. A seeded affine permutation distributes training windows
    across instruments without an O(number-of-bars) permutation array. Resume
    uses the native consumed-sample counter with the same immutable manifest.
    """

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        seq_length: int,
        horizon: int = 1,
        stride: int = 1,
        target_length: int | None = None,
        seed: int | None = None,
        keep_tail: bool = False,
        epochs: int | None = None,
        cycle_length: int | None = None,
        covariates: Path | None = None,
        window_indices: Path | None = None,
    ) -> None:
        if seq_length <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("seq_length, horizon and stride must be positive.")
        self.root, self.seq_length, self.horizon, self.stride = Path(root), seq_length, horizon, stride
        self.covariates = covariates
        self.selection = np.load(window_indices, allow_pickle=False) if window_indices is not None else None
        if covariates is not None:
            metadata = json.loads((covariates / "manifest.json").read_text())
            if metadata["source_manifest_sha256"] != manifest_digest(self.root):
                raise ValueError("Covariates do not match the prepared dataset manifest.")
        if keep_tail and (stride != seq_length or horizon != 1):
            raise ValueError("Tail windows require stride=seq_length and horizon=1.")
        if epochs is not None and epochs <= 0:
            raise ValueError("epochs must be positive.")
        self.keep_tail, self.epochs = keep_tail, epochs
        self.cycle_length = cycle_length
        manifest: OHLCVManifest = json.loads((self.root / "manifest.json").read_text())
        if (
            manifest["version"] not in (1, 2)
            or manifest["feature_names"] != list(FEATURE_NAMES)
            or manifest["price_scale"] != PRICE_SCALE
            or manifest["volume_scale"] != VOLUME_SCALE
        ):
            raise ValueError("Prepared dataset has an incompatible feature contract.")
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}.")
        self.split = split
        self.split_index = SPLITS.index(split)
        self.instruments: list[InstrumentInfo] = []
        self.cumulative: list[int] = []
        count = 0
        for instrument in manifest["instruments"]:
            segments = np.load(self.root / instrument["path"] / "segments.npy", mmap_mode="r", allow_pickle=False)
            windows = int(self._window_counts(segments).sum())
            if windows:
                self.instruments.append(instrument)
                count += windows
                self.cumulative.append(count)
        if count == 0:
            raise ValueError(f"No {split} windows for context={seq_length}, horizon={horizon}.")
        self.num_windows = count
        self.length = target_length if target_length is not None else count
        if self.length <= 0:
            raise ValueError("target_length must be positive.")
        self.multiplier, self.offset = 1, 0
        if seed is not None and count > 1:
            rng = random.Random(seed)
            self.multiplier = rng.randrange(1, count)
            while math.gcd(self.multiplier, count) != 1:
                self.multiplier = rng.randrange(1, count)
            self.offset = rng.randrange(count)
        self._cached_path: str | None = None
        self._arrays: ShardArrays | None = None
        self._segment_counts = np.empty(0, dtype=np.int64)

    def _window_counts(self, segments: NDArray[np.int64]) -> NDArray[np.int64]:
        if self.keep_tail:
            counts = (segments[:, 1] - 1 + self.seq_length - 1) // self.seq_length
        else:
            counts = np.maximum(0, (segments[:, 1] - self.seq_length - self.horizon) // self.stride + 1)
        return np.where(segments[:, 2] == self.split_index, counts, 0)

    def __len__(self) -> int:
        """Return the requested native sampler budget."""
        return self.length

    def raw_indices(self, indices: NDArray[np.int64]) -> NDArray[np.int64]:
        """Resolve sampler indices to the unpermuted split inventory."""
        if self.cycle_length is not None:
            indices = indices % self.cycle_length
        if self.selection is not None:
            return self.selection[indices % len(self.selection)]
        return (indices * self.multiplier + self.offset) % self.num_windows

    def window(self, index: int) -> tuple[WindowInfo, int, ShardArrays]:
        """Locate a window and mmap its instrument; retain only one open shard."""
        if self.cycle_length is not None:
            index %= self.cycle_length
        if self.selection is not None:
            index = int(self.selection[index % len(self.selection)])
        else:
            index = (index * self.multiplier + self.offset) % self.num_windows
        instrument_index = bisect.bisect_right(self.cumulative, index)
        instrument = self.instruments[instrument_index]
        previous = self.cumulative[instrument_index - 1] if instrument_index else 0
        local_index = index - previous
        if self._cached_path != instrument["path"]:
            self._arrays = self._load_shard(instrument["path"])
            self._segment_counts = np.cumsum(self._window_counts(self._arrays["segments"]))
            self._cached_path = instrument["path"]
        assert self._arrays is not None
        segment_index = int(np.searchsorted(self._segment_counts, local_index, side="right"))
        previous_segment = int(self._segment_counts[segment_index - 1]) if segment_index else 0
        offset, length, _ = self._arrays["segments"][segment_index]
        segment: WindowInfo = {**instrument, "split": self.split, "offset": int(offset), "length": int(length)}
        start = int(offset) + (local_index - previous_segment) * self.stride
        return segment, start, self._arrays

    def _load_shard(self, path: str) -> ShardArrays:
        directory = self.root / path
        feature_dir = directory if self.covariates is None else self.covariates / path
        return ShardArrays(
            features=np.load(feature_dir / "features.npy", mmap_mode="r", allow_pickle=False),
            close=np.load(directory / "close.npy", mmap_mode="r", allow_pickle=False),
            timestamps=np.load(directory / "timestamps.npy", mmap_mode="r", allow_pickle=False),
            segments=np.load(directory / "segments.npy", mmap_mode="r", allow_pickle=False),
        )

    def __getitem__(self, index: int) -> dict[str, NDArray[np.float32]]:
        """Return aligned teacher-forcing inputs and next-bar labels."""
        sample = {
            "features": np.zeros((self.seq_length, 5), dtype=np.float32),
            "labels": np.zeros((self.seq_length, 5), dtype=np.float32),
            "loss_mask": np.zeros(self.seq_length, dtype=np.float32),
        }
        # Complete the final native global batch without repeating real labels.
        if self.epochs is not None and index >= self.num_windows * self.epochs:
            return sample
        segment, start, arrays = self.window(index)
        end = min(start + self.seq_length + 1, segment["offset"] + segment["length"])
        values = arrays["features"][start:end]
        valid = len(values) - 1
        sample["features"][:valid] = values[:-1]
        sample["labels"][:valid] = values[1:]
        sample["loss_mask"][:valid] = 1
        return sample


def manifest_digest(root: Path) -> str:
    """Fingerprint the prepared dataset selection and source revision."""
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


class OHLCVQuantileWindowDataset(OHLCVWindowDataset):
    """Complete historical windows with future cumulative close-return labels.

    ``feature_columns`` selects covariate-cache columns for model input. When
    omitted, PVW windows use the cache's selected columns and covariate windows
    use every column, as recorded by checkpoints that predate the selection.
    ``packed`` reads the same windows from a Megatron ``.bin/.idx`` directory
    written by :func:`~megatron.bridge.data.datasets.ohlcv_indexed.pack_quantile_windows`.
    """

    def __init__(
        self,
        *args: Any,
        pvw_cache: Path | None = None,
        feature_columns: list[int] | None = None,
        packed: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pvw_cache = pvw_cache
        self.feature_columns = feature_columns
        self.pvw_metadata: dict[str, Any] | None = None
        self._pvw_selection: NDArray[np.int64] | None = None
        self._pvw_values: NDArray[np.float32] | None = None
        self._shards: OrderedDict[int, tuple[ShardArrays, NDArray[np.int64], NDArray[np.float32] | None]] = (
            OrderedDict()
        )
        if pvw_cache is not None:
            if self.covariates is None or self.stride != 1:
                raise ValueError("PVW requires the source covariates and a stride-one window inventory.")
            metadata = json.loads((pvw_cache / "manifest.json").read_text())
            if (
                metadata["source_manifest_sha256"] != manifest_digest(self.root)
                or metadata["covariates_sha256"] != manifest_digest(self.covariates)
                or (metadata["context"], metadata["horizon"]) != (self.seq_length, self.horizon)
                or metadata["windows"][self.split] != self.num_windows
            ):
                raise ValueError("PVW cache does not match the source window inventory.")
            self.pvw_metadata = metadata
            if self.feature_columns is None:
                self.feature_columns = metadata["columns"]
            if self.split != "train":
                self._pvw_selection = np.load(pvw_cache / f"{self.split}-indices.npy", mmap_mode="r")
                self._pvw_values = np.load(pvw_cache / f"{self.split}-values.npy", mmap_mode="r")
        self.packed = None
        if packed is not None:
            from megatron.bridge.data.datasets.ohlcv_indexed import PackedQuantileWindows, packed_identity

            self.packed = PackedQuantileWindows(packed)
            if self.packed.metadata["identity"] != packed_identity(self):
                raise ValueError("Packed windows do not match this dataset's inventory, features or caches.")

    def iter_raw_windows(
        self, raw_indices: NDArray[np.int64]
    ) -> Iterator[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], ShardArrays, NDArray[np.float32] | None]
    ]:
        """Gather raw split indices by instrument, retaining each input position."""
        instruments = np.searchsorted(self.cumulative, raw_indices, side="right")
        order = np.argsort(instruments, kind="stable")
        boundaries = np.flatnonzero(np.diff(instruments[order])) + 1
        for positions in np.split(order, boundaries):
            if not len(positions):
                continue
            instrument = int(instruments[positions[0]])
            if instrument not in self._shards:
                path = self.instruments[instrument]["path"]
                arrays = self._load_shard(path)
                prefix = np.cumsum(self._window_counts(arrays["segments"]))
                values = None
                if self.pvw_cache is not None and self.split == "train":
                    values = np.load(self.pvw_cache / "train" / path / "values.npy", mmap_mode="r")
                self._shards[instrument] = arrays, prefix, values
                if len(self._shards) > 128:
                    self._shards.popitem(last=False)
            self._shards.move_to_end(instrument)
            arrays, prefix, values = self._shards[instrument]
            raw = raw_indices[positions]
            local = raw - (self.cumulative[instrument - 1] if instrument else 0)
            segment = np.searchsorted(prefix, local, side="right")
            prior = np.where(segment > 0, prefix[segment - 1], 0)
            starts = arrays["segments"][segment, 0] + (local - prior) * self.stride
            if self._pvw_selection is not None:
                assert self._pvw_values is not None
                rows = np.searchsorted(self._pvw_selection, raw)
                if np.any(rows >= len(self._pvw_selection)) or np.any(self._pvw_selection[rows] != raw):
                    raise ValueError("The requested evaluation window is not in the frozen PVW cache.")
                denoised = self._pvw_values[rows]
            else:
                denoised = values[local] if values is not None else None
            yield positions, raw, starts, arrays, denoised

    def __getitems__(self, indices: list[int]) -> list[dict[str, NDArray[np.float32]]]:
        """Read a microbatch in instrument order and restore sampler order."""
        sampler_indices = np.asarray(indices, dtype=np.int64)
        valid = np.arange(len(indices))
        if self.epochs is not None:
            valid = valid[sampler_indices < self.num_windows * self.epochs]
        raw_indices = self.raw_indices(sampler_indices[valid])
        channels = len(FEATURE_NAMES)
        if self.covariates is not None and self.feature_columns is None:
            from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES

            channels = len(FORECAST_FEATURE_NAMES)
        elif self.feature_columns is not None:
            channels = len(self.feature_columns) + (self.pvw_metadata is not None)
        features = np.zeros((len(indices), self.seq_length, channels), dtype=np.float32)
        labels = np.zeros((len(indices), self.horizon), dtype=np.float32)
        mask = np.zeros_like(labels)
        if self.packed is not None:
            features[valid], labels[valid] = self.packed.read(raw_indices)
            mask[valid] = 1
            return [
                {"features": values, "labels": target, "loss_mask": weights}
                for values, target, weights in zip(features, labels, mask, strict=True)
            ]
        history = np.arange(self.seq_length)
        future = self.seq_length + np.arange(self.horizon)
        for positions, _, starts, arrays, denoised in self.iter_raw_windows(raw_indices):
            rows = valid[positions]
            values = arrays["features"][starts[:, None] + history]
            if self.feature_columns is not None:
                values = values[..., self.feature_columns]
            if denoised is None:
                features[rows] = values
            else:
                features[rows, :, :-1] = values
                features[rows, :, -1] = denoised
            anchor = arrays["close"][starts + self.seq_length - 1]
            labels[rows] = PRICE_SCALE * np.log(arrays["close"][starts[:, None] + future] / anchor[:, None])
            mask[rows] = 1
        return [
            {"features": values, "labels": target, "loss_mask": weights}
            for values, target, weights in zip(features, labels, mask, strict=True)
        ]

    def __getitem__(self, index: int) -> dict[str, NDArray[np.float32]]:
        """Return historical inputs and every future close relative to one anchor."""
        return self.__getitems__([index])[0]
