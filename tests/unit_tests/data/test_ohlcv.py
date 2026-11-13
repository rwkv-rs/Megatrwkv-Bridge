"""Numerical feature inversion and temporal isolation of OHLCV windows."""

import json
import pathlib

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


def test_lunch_break_links_the_trading_day_but_overnight_cuts():
    timestamps = np.array(
        ["2026-01-05T11:29", "2026-01-05T11:30", "2026-01-05T13:01", "2026-01-05T13:02", "2026-01-06T09:31"],
        dtype="datetime64[us]",
    )
    bars = np.tile([10, 11, 9, 10, 0], (5, 1)).astype(float)
    assert contiguous_segments(timestamps, bars) == [(0, 4), (4, 5)]
    # A missing 13:01 bar is a gap, not the lunch break.
    assert contiguous_segments(np.delete(timestamps, 2), np.delete(bars, 2, axis=0)) == [(0, 2), (2, 3), (3, 4)]


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


def test_fixed_evaluation_cycle_reuses_identical_windows(prepared):
    _, output = prepared
    dataset = OHLCVWindowDataset(
        output,
        split="train",
        seq_length=2,
        stride=2,
        keep_tail=True,
        seed=42,
        cycle_length=3,
    )
    for i in range(3):
        for key in ("features", "labels", "loss_mask"):
            np.testing.assert_array_equal(dataset[i][key], dataset[i + 3][key])


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


def test_parallel_preparation_matches_serial(prepared):
    source, serial = prepared
    parallel = serial.parent / "parallel"
    result = prepare_ohlcv(
        source, parallel, train_end="2024-01-01", validation_end="2025-01-01", revision="test-revision", workers=2
    )
    assert result == json.loads((serial / "manifest.json").read_text())
    for name in ("features", "close", "timestamps", "segments"):
        np.testing.assert_array_equal(
            np.load(parallel / "SZ/000001" / f"{name}.npy"), np.load(serial / "SZ/000001" / f"{name}.npy")
        )


def test_epoch_covers_each_target_once_and_masks_tail_and_final_batch(prepared):
    _, output = prepared
    dataset = OHLCVWindowDataset(
        output, split="train", seq_length=3, stride=3, keep_tail=True, epochs=1, target_length=4
    )
    assert dataset.num_windows == 3 and len(dataset) == 4
    labels = []
    valid_counts = []
    for index in range(4):
        item = dataset[index]
        valid = item["loss_mask"].astype(bool)
        labels.extend(item["labels"][valid])
        valid_counts.append(int(valid.sum()))
    assert valid_counts == [3, 3, 1, 0]
    features = np.load(output / "SZ/000001/features.npy")
    np.testing.assert_array_equal(np.array(labels), features[1:8])
    assert not dataset[3]["features"].any()
    resumed = OHLCVWindowDataset(
        output, split="train", seq_length=3, stride=3, keep_tail=True, epochs=1, target_length=4, seed=2026
    )
    assert sum(resumed[index]["loss_mask"].sum() for index in range(4)) == 7


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


def test_quantile_targets_share_one_anchor_and_stay_inside_split(prepared):
    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset

    _, root = prepared
    for split in ("train", "validation", "test"):
        dataset = OHLCVQuantileWindowDataset(root, split=split, seq_length=3, horizon=4)
        assert dataset.num_windows == 2
        for i in range(len(dataset)):
            _, start, arrays = dataset.window(i)
            sample = dataset[i]
            assert sample["features"].shape == (3, 5)
            assert sample["labels"].shape == sample["loss_mask"].shape == (4,)
            expected = 100 * np.log(arrays["close"][start + 3 : start + 7] / arrays["close"][start + 2])
            np.testing.assert_allclose(sample["labels"], expected, rtol=1e-6)
            np.testing.assert_allclose(
                arrays["close"][start + 2] * np.exp(sample["labels"].astype(float) / 100),
                arrays["close"][start + 3 : start + 7],
                rtol=1e-7,
            )
            assert sample["loss_mask"].sum() == 4
            assert (np.diff(arrays["timestamps"][start : start + 7]) == np.timedelta64(1, "m")).all()


def test_covariates_are_causal_and_reset_at_segments():
    from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES, forecast_covariates

    features = np.ones((12, 5), dtype=np.float32) * 0.01
    timestamps = np.datetime64("2026-01-05T09:30", "us") + np.arange(12) * np.timedelta64(1, "m")
    segments = np.array([[0, 6, 0], [6, 6, 1]])
    market = np.zeros((12, 3))
    original = forecast_covariates(features, timestamps, segments, market)
    changed = features.copy()
    changed[9:] *= 2
    np.testing.assert_array_equal(forecast_covariates(changed, timestamps, segments, market)[:9], original[:9])
    changed = features.copy()
    changed[:6] *= 2
    np.testing.assert_allclose(forecast_covariates(changed, timestamps, segments, market)[6:], original[6:], atol=1e-7)
    assert original.shape == (12, len(FORECAST_FEATURE_NAMES))
    assert original[6, FORECAST_FEATURE_NAMES.index("return_60")] == pytest.approx(0.01)
    assert original[6, FORECAST_FEATURE_NAMES.index("relative_volume_60")] == pytest.approx(0, abs=1e-7)


def test_prepared_covariates_keep_labels_and_frozen_indices(prepared, tmp_path):
    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset
    from megatron.bridge.data.datasets.ohlcv_covariates import prepare_covariates

    _, root = prepared
    output = tmp_path / "covariates"
    prepare_covariates(root, output, workers=1)
    np.save(tmp_path / "indices.npy", np.array([1, 0]))
    original = OHLCVQuantileWindowDataset(root, split="test", seq_length=3, horizon=4)
    augmented = OHLCVQuantileWindowDataset(
        root, split="test", seq_length=3, horizon=4, covariates=output, window_indices=tmp_path / "indices.npy"
    )
    assert augmented[0]["features"].shape == (3, 23)
    np.testing.assert_array_equal(augmented[0]["features"][:, :5], original[1]["features"])
    np.testing.assert_array_equal(augmented[0]["labels"], original[1]["labels"])
    np.testing.assert_array_equal(augmented[2]["features"], augmented[0]["features"])
    columns = [0, 1, 2, 3, 4, 19]
    selected = OHLCVQuantileWindowDataset(
        root,
        split="test",
        seq_length=3,
        horizon=4,
        covariates=output,
        window_indices=tmp_path / "indices.npy",
        feature_columns=columns,
    )
    np.testing.assert_array_equal(selected[0]["features"], augmented[0]["features"][:, columns])
    np.testing.assert_array_equal(selected[0]["labels"], augmented[0]["labels"])


def test_stratified_window_selection_uses_all_stride_one_origins(prepared, tmp_path):
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast_windows.py"
    _, root = prepared
    args = SimpleNamespace(
        data=root,
        output=tmp_path / "origins",
        split="test",
        count=2,
        context=3,
        horizon=4,
        seed=1234,
        exclude_dates=None,
    )
    runpy.run_path(str(entry))["select_windows"](args)
    frozen = json.loads(args.output.with_suffix(".json").read_text())
    assert {r["index"] for r in frozen} == {0, 1}
    assert all(r["window_stride"] == 1 for r in frozen)
    assert len({r["context_end"] for r in frozen}) == 2


def test_direct_quantile_training_defers_test_split_until_model_selection(prepared):
    from types import SimpleNamespace

    from megatron.bridge.recipes.rwkv7.forecast import OHLCVDatasetConfig

    _, root = prepared
    config = OHLCVDatasetConfig(
        root=str(root), seq_length=3, seed=1234, forecast_task="direct_quantile", forecast_horizon=4
    )
    train, validation, test = config.build_datasets(SimpleNamespace(train_samples=2, valid_samples=2, test_samples=2))
    assert train is not None and validation is not None
    assert test is None


def test_pvw_cache_matches_raw_ids_and_masks_final_batch(prepared, tmp_path):
    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset, manifest_digest
    from megatron.bridge.data.datasets.ohlcv_covariates import prepare_covariates

    _, root = prepared
    covariates = tmp_path / "covariates"
    cache = tmp_path / "pvw"
    prepare_covariates(root, covariates, workers=1)
    instrument = json.loads((root / "manifest.json").read_text())["instruments"][0]["path"]
    shard = cache / "train" / instrument
    shard.mkdir(parents=True)
    rows = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(shard / "values.npy", rows)
    np.save(cache / "validation-indices.npy", [1])
    np.save(cache / "validation-values.npy", rows[1:])
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": manifest_digest(root),
                "covariates_sha256": manifest_digest(covariates),
                "context": 3,
                "horizon": 4,
                "windows": {"train": 2, "validation": 2, "test": 2},
                "columns": [0, 1, 2, 3, 4, 10],
            }
        )
    )
    dataset = OHLCVQuantileWindowDataset(
        root,
        split="train",
        seq_length=3,
        horizon=4,
        epochs=1,
        target_length=4,
        seed=1234,
        covariates=covariates,
        pvw_cache=cache,
    )
    mixed = dataset.__getitems__([1, 3, 0, 2])
    np.testing.assert_array_equal(mixed[0]["features"][:, -1], rows[dataset.raw_indices(np.array([1]))[0]])
    np.testing.assert_array_equal(mixed[2]["features"][:, -1], rows[dataset.raw_indices(np.array([0]))[0]])
    assert mixed[1]["features"].shape == (3, 7)
    assert not mixed[1]["loss_mask"].any()
    assert not mixed[3]["loss_mask"].any()
    frozen = OHLCVQuantileWindowDataset(
        root,
        split="validation",
        seq_length=3,
        horizon=4,
        covariates=covariates,
        pvw_cache=cache,
    )
    np.testing.assert_array_equal(frozen[1]["features"][:, -1], rows[1])
    with pytest.raises(ValueError, match="frozen PVW cache"):
        frozen[0]


def test_windows_without_price_movement_carry_no_loss(tmp_path):
    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset

    source = tmp_path / "source"
    file = source / "data/stock_1m/SZ/000002.parquet"
    file.parent.mkdir(parents=True)
    timestamps = np.datetime64("2023-12-29T09:30", "us") + np.arange(9) * np.timedelta64(1, "m")
    # A halted instrument repeats one price, then resumes at bar 4.
    close = np.full(len(timestamps), 10.0)
    close[4:] = 10.5
    table = pa.table(
        {
            "exchange": ["SZ"] * len(close),
            "symbol": ["000002"] * len(close),
            "timestamp": timestamps,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.zeros(len(close)),
        }
    )
    pq.write_table(table, file)
    root = tmp_path / "prepared"
    prepare_ohlcv(source, root, train_end="2024-01-01", validation_end="2025-01-01", revision="test-revision")
    dataset = OHLCVQuantileWindowDataset(root, split="train", seq_length=3, horizon=4)
    assert dataset.num_windows == 2
    flat, moving = dataset.__getitems__([0, 1])
    # The first window's history is flat, so it is masked even though its own
    # future moves: the test reads the history alone, never the outcome.
    assert not flat["loss_mask"].any() and flat["labels"].any()
    # The second window's history contains the resuming bar, so it is kept.
    assert moving["loss_mask"].all()


def test_evaluation_inputs_use_the_checkpoint_feature_columns(prepared, tmp_path, monkeypatch):
    """evaluate must build model inputs from the columns the checkpoint trained on."""
    import runpy
    from types import SimpleNamespace

    import torch

    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset, manifest_digest
    from megatron.bridge.data.datasets.ohlcv_covariates import prepare_covariates

    _, root = prepared
    covariates = tmp_path / "covariates"
    prepare_covariates(root, covariates, workers=1)
    columns = [0, 1, 2, 3, 4, 19]
    entry = pathlib.Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast_quantile.py"
    evaluate_quantiles = runpy.run_path(str(entry))["evaluate_quantiles"]
    levels = tuple(i / 10 for i in range(1, 10))
    payload = {
        "step": 7,
        "run": {
            "manifest_sha256": manifest_digest(root),
            "seed": 1234,
            "covariates": str(covariates),
            "covariates_sha256": manifest_digest(covariates),
            "hf_config": {
                "context_length": 3,
                "forecast_feature_columns": columns,
                "forecast_target_quantiles": [[0.0] * 9] * 4,
            },
        },
    }
    seen = []

    class Model:
        horizon, quantile_levels, volatility_floor = 4, levels, None
        input_size = len(columns)

        def predict_quantiles(self, inputs, *, fix_quantile_crossing=True):
            seen.append(tuple(inputs.shape))
            return torch.zeros(len(inputs), self.horizon, len(levels))

    (tmp_path / "forecast.pt").write_bytes(b"weights")
    args = SimpleNamespace(
        data=root,
        split="test",
        horizon=4,
        max_windows=4,
        windows=None,
        output=tmp_path / "evaluation/metrics.json",
        checkpoint=tmp_path / "forecast.pt",
        covariates=covariates,
        pvw_cache=None,
        baseline_paths=None,
        direct_baseline_paths=None,
        metrics_only=True,
    )
    evaluate_quantiles(args, Model(), payload)
    # Without the checkpoint columns this would be the full 23-channel cache.
    assert seen and all(shape[-1] == len(columns) for shape in seen)
    report = json.loads(args.output.read_text())
    assert "movement_free_windows" in report and "point_excluding_movement_free" in report
    # The conditional reference still reads volatility_60 from every covariate.
    full = OHLCVQuantileWindowDataset(root, split="test", seq_length=3, horizon=4, covariates=covariates)
    assert full[0]["features"].shape[-1] == 23


@pytest.mark.parametrize("with_pvw", [False, True])
def test_packed_windows_match_memory_mapped_windows(prepared, tmp_path, with_pvw):
    from megatron.bridge.data.datasets.ohlcv import OHLCVQuantileWindowDataset, manifest_digest
    from megatron.bridge.data.datasets.ohlcv_covariates import prepare_covariates
    from megatron.bridge.data.datasets.ohlcv_indexed import pack_quantile_windows

    _, root = prepared
    covariates = tmp_path / "covariates"
    prepare_covariates(root, covariates, workers=1)
    cache = None
    if with_pvw:
        cache = tmp_path / "pvw"
        instrument = json.loads((root / "manifest.json").read_text())["instruments"][0]["path"]
        (cache / "train" / instrument).mkdir(parents=True)
        np.save(cache / "train" / instrument / "values.npy", np.arange(6, dtype=np.float32).reshape(2, 3) + 0.5)
        (cache / "manifest.json").write_text(
            json.dumps(
                {
                    "source_manifest_sha256": manifest_digest(root),
                    "covariates_sha256": manifest_digest(covariates),
                    "context": 3,
                    "horizon": 4,
                    "windows": {"train": 2, "validation": 2, "test": 2},
                    "columns": [0, 1, 2, 3, 4, 10],
                }
            )
        )
    kwargs = dict(
        split="train", seq_length=3, horizon=4, covariates=covariates, pvw_cache=cache, feature_columns=[0, 1, 4, 19]
    )
    source = OHLCVQuantileWindowDataset(root, **kwargs)
    pack_quantile_windows(source, tmp_path / "packed", workers=1)
    packed = OHLCVQuantileWindowDataset(
        root, packed=tmp_path / "packed", epochs=1, target_length=4, seed=1234, **kwargs
    )
    shuffled = OHLCVQuantileWindowDataset(root, epochs=1, target_length=4, seed=1234, **kwargs)
    order = [1, 3, 0, 2]
    for expected, actual in zip(shuffled.__getitems__(order), packed.__getitems__(order), strict=True):
        np.testing.assert_array_equal(actual["features"], expected["features"])
        np.testing.assert_allclose(actual["labels"], expected["labels"], rtol=0, atol=1e-5)
        np.testing.assert_array_equal(actual["loss_mask"], expected["loss_mask"])
    with pytest.raises(ValueError, match="Packed windows do not match"):
        OHLCVQuantileWindowDataset(root, packed=tmp_path / "packed", **{**kwargs, "feature_columns": [0, 1]})
