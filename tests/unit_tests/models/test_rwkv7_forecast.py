"""Continuous task heads, state lifecycle, serialization and native loss counts."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import RwkvConfig


pytest.importorskip("transformers.integrations.flash_rwkv2", reason="Requires transformers-rwkv RWKV-7")

from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import RWKV7ForecastConfig, RWKV7ForecastModel
from megatron.bridge.recipes.rwkv7.forecast import forecast_loss, rwkv7_ohlcv_config


pytestmark = pytest.mark.unit


@pytest.fixture
def config(tmp_path):
    return rwkv7_ohlcv_config(
        RwkvConfig(hidden_size=64, num_hidden_layers=2, vocab_size=1, context_length=16),
        data=tmp_path,
        output=tmp_path,
        steps=20,
        micro_batch_size=1,
        global_batch_size=2,
        learning_rate=3e-4,
        warmup_steps=1,
        save_interval=10,
        eval_interval=10,
        eval_iters=0,
        seed=2026,
    ).model


def test_config_checkpoint_and_parameter_contract(config):
    restored = RWKV7ForecastConfig.from_dict(config.as_dict())
    assert restored.get_builder_cls().model_class is RWKV7ForecastModel
    model = RWKV7ForecastModel(config, None)
    assert not any("embed_tokens" in name or "lm_head" in name for name, _ in model.named_parameters())
    assert all(param.dtype == torch.bfloat16 for param in model.parameters())
    other = RWKV7ForecastModel(restored, None)
    other.load_state_dict(model.state_dict(), strict=True)
    for name, value in other.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


def test_task_head_preserves_alignment_and_gradients(config):
    model = RWKV7ForecastModel(config, None)
    features = torch.randn(2, 8, 5)
    labels = torch.randn(2, 8, 5)
    with patch.object(
        model.hf, "forward", side_effect=lambda **kwargs: SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])
    ):
        prediction = model(features)
        loss = model(features, labels=labels)
    assert prediction.dtype == loss.dtype == torch.float32
    assert torch.all(prediction[..., 2:] > 0)
    torch.testing.assert_close(loss, F.smooth_l1_loss(prediction, labels, reduction="none").mean(-1))
    loss.mean().backward()
    assert model.input_projection.weight.grad.abs().sum() > 0
    assert model.output_projection.weight.grad.abs().sum() > 0


def test_generate_feeds_predictions_back_and_resets_state(config):
    model = RWKV7ForecastModel(config, None).eval()
    calls, caches = [], []

    def forward(features, labels=None, *, past_key_values=None):
        calls.append(features.clone())
        caches.append(past_key_values)
        return features.float() + 0.25

    history = torch.ones(1, 8, 5)
    with patch.object(model, "forward", side_effect=forward):
        result = model.generate(history, horizon=3)
        model.generate(history, horizon=1)
    assert [x.shape[1] for x in calls] == [8, 1, 1, 8]
    torch.testing.assert_close(calls[1], result[:, :1])
    torch.testing.assert_close(calls[2], result[:, 1:2])
    assert caches[0] is caches[1] is caches[2]
    assert caches[0] is not caches[3]
    torch.testing.assert_close(result[0, :, 0], torch.tensor([1.25, 1.5, 1.75]))


def test_native_loss_counts_bars_not_channels():
    losses = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    with patch("megatron.bridge.recipes.rwkv7.forecast.validate_local_loss"):
        loss, count, report = forecast_loss(losses, check_for_nan_in_loss=True, check_for_spiky_loss=False)
    assert loss.item() == 10 and count.item() == 4
    torch.testing.assert_close(report["ohlcv huber"], torch.tensor([10.0, 4.0]))
    (loss / count).backward()
    torch.testing.assert_close(losses.grad, torch.full_like(losses, 0.25))


def test_tail_and_dummy_samples_do_not_contribute_loss_or_gradients():
    losses = torch.tensor([[1.0, 2.0, 100.0], [200.0, 300.0, 400.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with patch("megatron.bridge.recipes.rwkv7.forecast.validate_local_loss"):
        loss, count, _ = forecast_loss(losses, loss_mask=mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)
    assert loss.item() == 3 and count.item() == 2
    (loss / count).backward()
    torch.testing.assert_close(losses.grad, mask / 2)


def test_calibrated_head_has_training_prior_and_fp32_output(config):
    config.hf_config.update(
        forecast_initialization="calibrated",
        forecast_target_mean=[0.0, 0.0, 0.03, 0.04, 0.85],
        forecast_head_gain=0.01,
        forecast_head_fp32=True,
    )
    model = RWKV7ForecastModel(config, None)
    hidden = torch.zeros(2, 8, config.hidden_size, dtype=torch.bfloat16)
    outputs = []
    handle = model.output_projection.register_forward_hook(lambda module, inputs, output: outputs.append(output))
    with patch.object(model.hf, "forward", return_value=SimpleNamespace(last_hidden_state=hidden)):
        prediction = model(torch.zeros(2, 8, 5))
    handle.remove()
    assert len(outputs) == 1 and outputs[0].dtype == torch.float32
    torch.testing.assert_close(prediction[0, 0], torch.tensor([0.0, 0.0, 0.03, 0.04, 0.85]), atol=0.003, rtol=0.01)
    assert model.input_projection.bias.norm() > 0
    assert prediction.dtype == torch.float32


def test_input_standardization_is_persisted_and_preserves_encoded_targets(config):
    config.hf_config.update(
        forecast_input_mean=[0.0, 0.0, 0.03, 0.04, 0.85],
        forecast_input_std=[0.1, 0.1, 0.05, 0.05, 0.3],
    )
    model = RWKV7ForecastModel(config, None)
    restored = RWKV7ForecastModel(RWKV7ForecastConfig.from_dict(config.as_dict()), None)
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = []
    handle = model.input_projection.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    features = torch.randn(2, 8, 5)
    labels = torch.randn_like(features)
    with patch.object(
        model.hf, "forward", side_effect=lambda **kwargs: SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])
    ):
        prediction = model(features)
        loss = model(features, labels=labels)
    handle.remove()
    expected = ((features - model.feature_mean.float()) / model.feature_std.float()).bfloat16()
    torch.testing.assert_close(inputs[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(loss, F.smooth_l1_loss(prediction, labels, reduction="none").mean(-1))
    assert not any("feature_mean" in key or "feature_std" in key for key in model.state_dict())
    torch.testing.assert_close(restored.feature_mean, model.feature_mean, rtol=0, atol=0)


def test_channel_losses_use_identical_mask_and_valid_count():
    channels = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    with patch("megatron.bridge.recipes.rwkv7.forecast.validate_local_loss"):
        _, count, report = forecast_loss(
            channels.mean(-1),
            loss_mask=mask,
            channel_losses=channels,
            check_for_nan_in_loss=True,
            check_for_spiky_loss=False,
        )
    assert count == 2
    assert report["ohlcv/open"].tolist() == [5.0, 2.0]
    assert report["ohlcv/volume"].tolist() == [13.0, 2.0]


@pytest.mark.parametrize(
    "name,shape,wd_mult,lr_mult",
    [
        ("hf.layers.1.linear_attn.k_a", (1, 1, 64), 0.0, 1.0),
        ("hf.layers.1.linear_attn.w0", (1, 1, 64), 0.0, 2.0),
        ("hf.layers.1.linear_attn.w1", (64, 32), 0.0, 1.0),
        ("hf.layers.1.linear_attn.r_k", (1, 64), 0.0, 1.0),
        ("hf.layers.1.linear_attn.g_norm.weight", (64,), 0.0, 1.0),
        ("hf.layers.1.linear_attn.k_proj.weight", (64, 64), 1.0, 1.0),
        ("output_projection.weight", (5, 64), 1.0, 1.0),
        ("output_projection.bias", (5,), 0.0, 1.0),
    ],
)
def test_rwkv_optimizer_group_policy(name, shape, wd_mult, lr_mult):
    from megatron.bridge.recipes.rwkv7.forecast import RWKV7ForecastOptimizerOverrides
    from megatron.bridge.training.config import (
        OptimizerConfig,
        OptimizerConfigOverrideProviderContext,
        SchedulerConfig,
    )

    context = OptimizerConfigOverrideProviderContext(
        model=None,
        optimizer_config=OptimizerConfig(lr=1e-4, min_lr=1e-5),
        scheduler_config=SchedulerConfig(),
    )
    overrides = RWKV7ForecastOptimizerOverrides().build_config_overrides(context)
    parameter = torch.nn.Parameter(torch.empty(shape))
    matching = [value for key, value in overrides.items() if key.matches(parameter, name)]
    assert all(value.get("wd_mult", wd_mult) == wd_mult for value in matching)
    assert all(value.get("lr_mult", lr_mult) == lr_mult for value in matching)
    assert any(value.get("wd_mult") == 0.0 for value in matching) == (wd_mult == 0.0)
    assert any(value.get("lr_mult") == 2.0 for value in matching) == (lr_mult == 2.0)
    if lr_mult == 2.0:
        assert any(value.get("max_lr") == 2e-4 and value.get("min_lr") == 2e-5 for value in matching)


@pytest.mark.parametrize("input_size", [5, 23])
def test_direct_quantile_head_loss_gradients_and_rearrangement(config, input_size):
    config.hf_config.update(forecast_task="direct_quantile", forecast_horizon=3, forecast_quantiles=[0.1, 0.5, 0.9])
    config.hf_config["forecast_input_size"] = input_size
    model = RWKV7ForecastModel(config, None)
    features = torch.randn(2, 8, input_size)
    labels = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    with patch.object(
        model.hf, "forward", side_effect=lambda **kw: SimpleNamespace(last_hidden_state=kw["inputs_embeds"])
    ):
        raw = model(features)
        losses = model(features, labels, return_channel_losses=True)
        error = labels.unsqueeze(-1) - raw
        levels = torch.tensor([0.1, 0.5, 0.9])
        torch.testing.assert_close(losses, torch.maximum(levels * error, (levels - 1) * error))
        assert raw.shape == (2, 3, 3) and raw.dtype == torch.float32
        loss, count, report = forecast_loss(
            losses.mean(-1),
            channel_losses=losses.detach(),
            quantile_levels=(0.1, 0.5, 0.9),
            check_for_nan_in_loss=False,
            check_for_spiky_loss=False,
        )
        assert count == 6 and "forecast pinball" in report
        (loss / count).backward()
        assert model.input_projection.weight.grad.abs().sum() > 0
        assert model.output_projection.weight.grad.abs().sum() > 0
        torch.testing.assert_close(model.predict_quantiles(features), raw.sort(-1).values)
        torch.testing.assert_close(model.predict_quantiles(features, fix_quantile_crossing=False), raw)
    assert not any("parametrizations" in name for name, _ in model.named_parameters())
    restored = RWKV7ForecastModel(RWKV7ForecastConfig.from_dict(config.as_dict()), None)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_volatility_scaled_quantiles_score_scaled_units_and_predict_returns(config):
    config.hf_config.update(
        forecast_task="direct_quantile",
        forecast_horizon=3,
        forecast_quantiles=[0.1, 0.5, 0.9],
        forecast_volatility_floor=0.25,
    )
    model = RWKV7ForecastModel(config, None)
    features = torch.randn(2, 8, 5)
    features[0, :, 1] = 0  # flat history: the floor applies
    features[1, :, 1] = 2
    labels = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    scale = torch.tensor([[0.25], [2.0]]) * torch.arange(1, 4).sqrt()
    levels = torch.tensor([0.1, 0.5, 0.9])
    with patch.object(
        model.hf, "forward", side_effect=lambda **kw: SimpleNamespace(last_hidden_state=kw["inputs_embeds"])
    ):
        loss = model(features, labels, return_channel_losses=True)
        prediction = model(features)
        error = (labels.unsqueeze(-1) - prediction) / scale.unsqueeze(-1)
        torch.testing.assert_close(loss, torch.maximum(levels * error, (levels - 1) * error))
        model.eval()
        torch.testing.assert_close(model(features, labels, return_channel_losses=True), loss)
        torch.testing.assert_close(model.predict_quantiles(features), prediction.sort(-1).values)
    restored = RWKV7ForecastModel(RWKV7ForecastConfig.from_dict(config.as_dict()), None)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.volatility_floor == 0.25


def test_scaled_inputs_are_invariant_to_return_magnitude_and_keep_its_level(config):
    from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import forecast_scaled_inputs

    features = torch.randn(2, 8, 6)
    scaled = torch.tensor([True, True, True, True, False, True])
    louder = features.clone()
    louder[..., scaled] *= 3
    reference = forecast_scaled_inputs(features, scaled, floor=1e-3)
    changed = forecast_scaled_inputs(louder, scaled, floor=1e-3)
    torch.testing.assert_close(changed[..., :6], reference[..., :6])
    torch.testing.assert_close(changed[..., 6], reference[..., 6] + torch.log(torch.tensor(3.0)))
    config.hf_config.update(
        forecast_task="direct_quantile",
        forecast_horizon=3,
        forecast_quantiles=[0.1, 0.5, 0.9],
        forecast_input_size=6,
        forecast_scaled_features=[0, 1, 2, 3, 5],
        forecast_volatility_floor=1e-3,
    )
    model = RWKV7ForecastModel(config, None)
    assert model.input_projection.in_features == 7
    with patch.object(
        model.hf, "forward", side_effect=lambda **kw: SimpleNamespace(last_hidden_state=kw["inputs_embeds"])
    ):
        assert model(features).shape == (2, 3, 3)


def test_direct_quantile_training_excludes_market_inputs_and_calibrates_scaled_targets(tmp_path, monkeypatch):
    import numpy as np

    from megatron.bridge.data.datasets.ohlcv_covariates import FORECAST_FEATURE_NAMES, MARKET_FEATURE_NAMES

    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast.py"
    train = runpy.run_path(str(entry))["train"]
    architecture = tmp_path / "architecture.json"
    RwkvConfig(hidden_size=64, num_hidden_layers=2, context_length=16).to_json_file(architecture)
    covariates = tmp_path / "covariates"
    covariates.mkdir()
    (covariates / "manifest.json").write_text("{}")
    expected_columns = [i for i, name in enumerate(FORECAST_FEATURE_NAMES) if name not in MARKET_FEATURE_NAMES]
    created = []

    class Windows:
        num_windows = 20

        def __init__(self, *args, **kwargs):
            created.append(kwargs)

        def __getitem__(self, index):
            # Alternate historical RMS 1 and 2; labels grow with volatility * sqrt(h).
            volatility = 1.0 + index % 2
            features = np.zeros((16, len(expected_columns)), dtype=np.float32)
            features[:, 1] = volatility
            labels = (volatility * np.sqrt(np.arange(1, 31))).astype(np.float32)
            return {"features": features, "labels": labels, "loss_mask": np.ones(30, dtype=np.float32)}

    args = SimpleNamespace(
        config=architecture,
        data=tmp_path / "data",
        output=tmp_path / "run",
        context=16,
        task="direct_quantile",
        forecast_horizon=30,
        forecast_architecture=None,
        covariates=covariates,
        pvw_cache=None,
        validation_indices=None,
        packed_train=None,
        calibration_windows=4,
        normalize_inputs=False,
        head_gain=0.01,
        head_fp32=False,
        rwkv_optimizer=False,
        epochs=None,
        steps=20,
        eval_iters=0,
        warmup_steps=0,
        global_batch_size=2,
        micro_batch_size=1,
        learning_rate=1e-4,
        save_interval=10,
        eval_interval=10,
        seed=2026,
        data_workers=None,
        early_stopping_patience=0,
        resume=None,
        exit_interval=None,
    )
    captured = []
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    with (
        patch("megatron.bridge.data.datasets.ohlcv.OHLCVQuantileWindowDataset", Windows),
        patch("megatron.bridge.data.datasets.ohlcv.manifest_digest", return_value="fixed-manifest"),
        patch(
            "megatron.bridge.training.pretrain.pretrain", side_effect=lambda cfg, forward, **kw: captured.append(cfg)
        ),
    ):
        train(args)
    hf_config = json.loads((args.output / "run.json").read_text())["hf_config"]
    assert hf_config["forecast_feature_columns"] == expected_columns
    assert not set(MARKET_FEATURE_NAMES) & set(hf_config["forecast_feature_names"])
    assert hf_config["forecast_input_size"] == len(expected_columns)
    returns = {"open_return", "close_return", "upper_wick", "lower_wick", "return_5", "return_15", "return_30"}
    returns |= {"return_60", "volatility_5", "volatility_15", "volatility_30", "volatility_60"}
    assert [hf_config["forecast_feature_names"][i] for i in hf_config["forecast_scaled_features"]] == [
        name for name in hf_config["forecast_feature_names"] if name in returns
    ]
    assert all(kwargs["feature_columns"] == expected_columns for kwargs in created)
    assert captured[-1].dataset.feature_columns == expected_columns
    assert hf_config["forecast_volatility_floor"] == pytest.approx(1.0, rel=1e-6)
    # Every target equals its window scale, so all scaled quantiles are one.
    np.testing.assert_allclose(hf_config["forecast_target_quantiles"], 1.0, rtol=1e-6)


def test_quantile_masked_microbatches_preserve_global_normalization():
    prediction = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    kwargs = dict(check_for_nan_in_loss=False, check_for_spiky_loss=False, quantile_levels=(0.5,))
    total, count, _ = forecast_loss(
        prediction, loss_mask=mask, channel_losses=prediction.detach().unsqueeze(-1), **kwargs
    )
    global_grad = torch.autograd.grad(total / count, prediction, retain_graph=True)[0]
    parts = [
        forecast_loss(
            prediction[i : i + 1],
            loss_mask=mask[i : i + 1],
            channel_losses=prediction[i : i + 1].detach().unsqueeze(-1),
            **kwargs,
        )
        for i in range(2)
    ]
    split_grad = torch.autograd.grad(sum(part[0] for part in parts) / sum(part[1] for part in parts), prediction)[0]
    torch.testing.assert_close(global_grad, split_grad)
    assert split_grad[1, 1] == 0


def test_early_stopping_requests_native_exit_after_patience(tmp_path):
    from megatron.bridge.recipes.rwkv7.forecast import ForecastExportCallback

    run = {"hf_config": {"forecast_task": "direct_quantile"}}
    callback = ForecastExportCallback(tmp_path, run=run, patience=1000)
    exported = []
    callback._export = lambda context, name: exported.append((context.state.train_state.step, name))
    cfg = SimpleNamespace(train=SimpleNamespace(exit_interval=None))

    def evaluate(step, pinball):
        context = SimpleNamespace(
            total_loss_dict={"forecast pinball": pinball, "forecast/q0.5": pinball / 2},
            state=SimpleNamespace(train_state=SimpleNamespace(step=step), cfg=cfg),
        )
        with patch("torch.distributed.get_rank", return_value=0):
            callback.on_eval_end(context)

    evaluate(250, 0.20)
    evaluate(500, 0.15)  # best
    for step in (750, 1000, 1250):
        evaluate(step, 0.16)
        assert cfg.train.exit_interval is None, step
    # 1500 - 500 reaches the 1000-step patience.
    evaluate(1500, 0.16)
    assert cfg.train.exit_interval == 1500
    assert callback.best_step == 500 and callback.best_loss == 0.15
    assert ("best-forecast.pt" in name for _, name in exported)
    # A fresh callback resumes the recorded best step, not step zero.
    resumed = ForecastExportCallback(tmp_path, run=run, patience=1000)
    assert resumed.best_step == 500 and resumed.best_loss == 0.15


def test_quantile_evaluation_proper_scores_and_price_units():
    import numpy as np

    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast_quantile.py"
    api = runpy.run_path(str(entry))
    target = np.zeros((2, 3))
    prediction = np.broadcast_to(np.array([-1.0, 0.0, 1.0]), (2, 3, 3))
    scores = api["quantile_metrics"](target, prediction, np.array([0.1, 0.5, 0.9]))
    assert scores["discrete_calibration_violation_by_quantile"][1] == 0
    assert scores["cdf_strict_by_horizon_quantile"][0][1] == 0
    assert scores["cdf_weak_by_horizon_quantile"][0][1] == 1
    assert scores["pinball_bps"] == pytest.approx(100 / 15)
    assert scores["interval_80"]["coverage"] == 1
    assert scores["interval_80"]["width_bps"] == 200
    assert scores["interval_80"]["interval_score_bps"] == 200
    actual = np.full((2, 3), 10.0)
    result = api["point_metrics"](actual, actual + 0.1, np.array([10.0, 10.0]))
    np.testing.assert_allclose(result["mae_cny_by_horizon"], 0.1)
    np.testing.assert_allclose(result["mae_pct_by_horizon"], 1.0)
