"""Numerical feature inversion and temporal isolation of OHLCV windows."""

import json

import numpy as np
import pytest


pa = pytest.importorskip("pyarrow", reason="Requires the parquet extra")
pq = pytest.importorskip("pyarrow.parquet", reason="Requires the parquet extra")

from megatron.bridge.data.datasets.ohlcv import (
    OHLCV_COLUMNS,
    OHLCVWindowDataset,
    contiguous_segments,
    decode_bars,
    encode_bars,
    load_instrument,
    prepare_ohlcv,
)


pytestmark = pytest.mark.unit


def test_feature_roundtrip_and_zero_volume():
    bars = np.array([[10, 12, 9, 11, 0], [11.2, 13, 10, 12, 180000], [12, 12, 12, 12, 0]], dtype=float)
    encoded = encode_bars(bars)
    np.testing.assert_allclose(decode_bars(encoded, previous_close=11), bars[1:], rtol=2e-6, atol=1e-7)
    assert encoded[1, 2:].tolist() == [0, 0, 0]
    assert encoded.dtype == np.float32


def test_prefix_features_do_not_depend_on_future():
    bars = np.tile([10, 12, 9, 11, 1000], (10, 1)).astype(float)
    original = encode_bars(bars)
    bars[7:, :4] *= 100
    np.testing.assert_array_equal(encode_bars(bars)[:6], original[:6])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_decoder_rejects_invalid_prediction(bad):
    features = np.zeros((2, 5))
    features[1, 3] = bad
    with pytest.raises(ValueError):
        decode_bars(features, previous_close=10)


def test_gaps_lunch_overnight_and_invalid_bars_cut_state():
    timestamps = np.array(
        [
            "2026-01-05T11:28",
            "2026-01-05T11:29",
            "2026-01-05T13:00",
            "2026-01-05T13:01",
            "2026-01-05T13:02",
            "2026-01-05T13:04",
            "2026-01-06T09:30",
            "2026-01-06T09:31",
        ],
        dtype="datetime64[us]",
    )
    bars = np.tile([10, 11, 9, 10, 0], (8, 1)).astype(float)
    bars[3, 0] = 0
    assert contiguous_segments(timestamps, bars) == [(0, 2), (2, 3), (4, 5), (5, 6), (6, 8)]


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source"
    file = source / "data/stock_1m/SZ/000001.parquet"
    file.parent.mkdir(parents=True)
    timestamps = np.concatenate(
        [
            np.datetime64(f"{day}T09:30", "us") + np.arange(9) * np.timedelta64(1, "m")
            for day in ("2023-12-29", "2024-06-03", "2025-02-03")
        ]
    )
    close = np.arange(len(timestamps)) * 0.01 + 10
    table = pa.table(
        {
            "exchange": ["SZ"] * len(close),
            "symbol": ["000001"] * len(close),
            "timestamp": timestamps,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.arange(len(close)) * 100,
        }
    )
    pq.write_table(table, file)
    output = tmp_path / "prepared"
    prepare_ohlcv(source, output, train_end="2024-01-01", validation_end="2025-01-01", revision="test-revision")
    return source, output


def test_split_isolation_and_single_label_shift(prepared):
    source, output = prepared
    manifest = json.loads((output / "manifest.json").read_text())
    assert {name: values["bars"] for name, values in manifest["splits"].items()} == {
        "train": 8,
        "validation": 8,
        "test": 8,
    }
    assert {s["symbol"] for s in manifest["instruments"]} == {"000001"}
    for split, year in (("train", "2023"), ("validation", "2024"), ("test", "2025")):
        dataset = OHLCVWindowDataset(output, split=split, seq_length=3)
        assert len(dataset) == 5
        for index in range(len(dataset)):
            segment, start, arrays = dataset.window(index)
            assert segment["split"] == split
            assert str(arrays["timestamps"][start]).startswith(year)
            assert str(arrays["timestamps"][start + 3]).startswith(year)
            sample = dataset[index]
            np.testing.assert_array_equal(sample["labels"][:-1], sample["features"][1:])
            np.testing.assert_array_equal(sample["labels"], arrays["features"][start + 1 : start + 4])
    with pytest.raises(FileExistsError):
        prepare_ohlcv(source, output, train_end="2024-01-01", validation_end="2025-01-01", revision="revision")


def test_seeded_traversal_is_bijective_and_resumable(prepared):
    _, output = prepared
    kwargs = dict(split="train", seq_length=2, seed=2026, target_length=30)
    first = OHLCVWindowDataset(output, **kwargs)
    restored = OHLCVWindowDataset(output, **kwargs)
    starts = [first.window(i)[1] for i in range(first.num_windows)]
    assert len(set(starts)) == first.num_windows
    for index in (0, 6, 17, 29):
        np.testing.assert_array_equal(first[index]["features"], restored[index]["features"])
        np.testing.assert_array_equal(first[index]["labels"], first[index + first.num_windows]["labels"])


def test_free_running_window_reserves_entire_horizon(prepared):
    _, output = prepared
    dataset = OHLCVWindowDataset(output, split="test", seq_length=3, horizon=3, stride=2)
    assert len(dataset) == 2
    for index in range(len(dataset)):
        segment, start, _ = dataset.window(index)
        assert start + 6 <= segment["offset"] + segment["length"]
    with pytest.raises(ValueError, match="No test windows"):
        OHLCVWindowDataset(output, split="test", seq_length=64)


def test_multi_instrument_index_and_mmap_switching(prepared):
    source, first_output = prepared
    table = pq.ParquetFile(source / "data/stock_1m/SZ/000001.parquet").read()
    table = table.set_column(table.schema.get_field_index("symbol"), "symbol", pa.array(["000002"] * len(table)))
    pq.write_table(table, source / "data/stock_1m/SZ/000002.parquet")
    output = first_output.parent / "multi"
    prepare_ohlcv(source, output, train_end="2024-01-01", validation_end="2025-01-01", revision="two-instruments")
    dataset = OHLCVWindowDataset(output, split="test", seq_length=3)
    assert len(dataset) == 10
    for index, symbol in ((0, "000001"), (5, "000002"), (4, "000001"), (9, "000002")):
        segment, start, arrays = dataset.window(index)
        assert segment["symbol"] == symbol
        assert start + 4 <= segment["offset"] + segment["length"]
        assert isinstance(arrays["features"], np.memmap)


def test_reject_unsorted_or_mixed_instrument_parquet(tmp_path):
    file = tmp_path / "bad.parquet"
    values = {name: [10, 10] for name in OHLCV_COLUMNS}
    values.update(
        exchange=["SH", "SH"],
        symbol=["600000", "600000"],
        timestamp=np.array(["2026-01-05T09:31", "2026-01-05T09:30"], dtype="datetime64[us]"),
    )
    pq.write_table(pa.table(values), file)
    with pytest.raises(ValueError, match="increasing"):
        load_instrument(file)
    values["symbol"] = ["600000", "600001"]
    pq.write_table(pa.table(values), file)
    with pytest.raises(ValueError, match="one instrument"):
        load_instrument(file)
