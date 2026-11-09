# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone forecasting feature and Transformers serialization contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from examples.models.rwkv7.hf_forecast.configuration_rwkv7_forecast import RWKV7ForecastConfig
from examples.models.rwkv7.hf_forecast.modeling_rwkv7_forecast import RWKV7ForQuantileForecasting
from examples.models.rwkv7.hf_forecast.processing_rwkv7_forecast import (
    FEATURE_NAMES,
    market_statistics,
    prepare_features,
)
from transformers import AutoConfig, AutoModel, RwkvConfig

from megatron.bridge.data.datasets.ohlcv import encode_bars
from megatron.bridge.data.datasets.ohlcv_covariates import forecast_covariates


@pytest.mark.unit
@pytest.mark.parametrize("length", [65, 80, 100])
def test_standalone_features_match_training_exactly(length):
    rng = np.random.default_rng(1234)
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.001, length)))
    opening = np.concatenate((close[:1], close[:-1]))
    bars = np.column_stack(
        (
            opening,
            np.maximum(opening, close) + 0.02,
            np.minimum(opening, close) - 0.02,
            close,
            rng.integers(0, 100000, length),
        )
    )
    stamps = np.datetime64("2026-11-06T09:30") + np.arange(length) * np.timedelta64(1, "m")
    market = market_statistics(rng.normal(0, 0.1, (length, 8)))
    features, anchor = prepare_features(bars, stamps, market)
    encoded = encode_bars(bars)
    expected = forecast_covariates(encoded, stamps[1:], np.array([[0, len(encoded), 0]]), market[1:])
    np.testing.assert_array_equal(features, expected[-64:])
    assert anchor == close[-1]


@pytest.mark.unit
def test_market_aggregate_counts_only_completed_available_bars():
    result = market_statistics(np.array([[0.1, -0.1, np.nan, 0], [np.nan, 0.2, 0.2, np.nan]]))
    np.testing.assert_allclose(result, [[0, np.sqrt(0.02 / 3), 1 / 3], [0.2, 0, 1]], atol=1e-8)


@pytest.mark.unit
def test_standalone_autoclass_reload_retains_all_tensors_and_fp32_head(tmp_path):
    backbone = RwkvConfig(hidden_size=64, num_hidden_layers=2, context_length=64)
    backbone.update(
        {
            "forecast_input_size": 23,
            "forecast_horizon": 30,
            "forecast_quantiles": [i / 10 for i in range(1, 10)],
            "forecast_feature_names": list(FEATURE_NAMES),
            "forecast_input_mean": [0.003] * 23,
            "forecast_input_std": [0.2] * 23,
        }
    )
    RWKV7ForecastConfig.register_for_auto_class("AutoConfig")
    RWKV7ForQuantileForecasting.register_for_auto_class("AutoModel")
    model = RWKV7ForQuantileForecasting(RWKV7ForecastConfig(backbone_config=backbone.to_dict())).bfloat16().eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded_config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True, dtype=torch.bfloat16).eval()
    assert loaded_config.model_type == "rwkv7_forecast"
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    torch.testing.assert_close(loaded.feature_mean, model.feature_mean, rtol=0, atol=0)
    torch.testing.assert_close(loaded.feature_std, model.feature_std, rtol=0, atol=0)
    features = torch.randn(2, 64, 23)
    with patch.object(
        loaded.hf, "forward", side_effect=lambda **kw: SimpleNamespace(last_hidden_state=kw["inputs_embeds"])
    ):
        result = loaded(features=features, anchor_price=torch.tensor([10.0, 20.0]))
    inputs = (features - loaded.feature_mean.float()) / loaded.feature_std.float()
    hidden = loaded.input_projection(inputs.bfloat16())[:, -1].float()
    expected = torch.nn.functional.linear(
        hidden, loaded.output_projection.weight.float(), loaded.output_projection.bias.float()
    ).view(2, 30, 9)
    torch.testing.assert_close(result.raw_quantiles, expected, rtol=0, atol=0)
    assert result.raw_quantiles.dtype == torch.float32
    assert result.price_quantiles.dtype == torch.float64
    assert torch.all(result.quantiles.diff(dim=-1) >= 0)
