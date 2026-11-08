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


def test_centered_channelmix_algebra_and_gradients():
    from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import _CenteredChannelMixWeight

    weight = torch.randn(4, 16, requires_grad=True)
    hidden = torch.randn(3, 16).relu().square().requires_grad_()
    upstream = torch.randn(3, 4)
    actual = F.linear(hidden, _CenteredChannelMixWeight()(weight))
    expected = F.linear(hidden - hidden.mean(-1, keepdim=True), weight)
    actual_grad = torch.autograd.grad(actual, (hidden, weight), upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (hidden, weight), upstream)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


def test_centered_channelmix_preserves_shared_initialization_and_reload(config):
    torch.manual_seed(2026)
    baseline = RWKV7ForecastModel(config, None)
    config.hf_config["forecast_center_channelmix"] = True
    torch.manual_seed(2026)
    centered = RWKV7ForecastModel(config, None)
    baseline_parameters = dict(baseline.named_parameters())
    for name, parameter in centered.named_parameters():
        canonical = name.replace(".parametrizations.weight.original", ".weight")
        torch.testing.assert_close(parameter, baseline_parameters[canonical], rtol=0, atol=0)
    restored = RWKV7ForecastModel(config, None)
    restored.load_state_dict(centered.state_dict(), strict=True)
    for layer, other in zip(centered.hf.layers, restored.hf.layers):
        torch.testing.assert_close(layer.mlp.value.weight, other.mlp.value.weight, rtol=0, atol=0)


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


def test_adamw_update_metrics_exclude_lr_and_decoupled_decay():
    from megatron.bridge.recipes.rwkv7.forecast_diagnostics import adamw_update_direction, tensor_statistics

    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for step in range(1, 4):
        before = parameter.detach().clone()
        parameter.grad = torch.tensor([0.3 * step, -0.1])
        optimizer.step()
        state = optimizer.state[parameter]
        direction = adamw_update_direction(
            state["exp_avg"],
            state["exp_avg_sq"],
            step=step,
            betas=(0.9, 0.95),
            eps=1e-8,
            bias_correction=True,
        )
        torch.testing.assert_close(parameter, before - 0.01 * (direction + 0.1 * before))
    stats = tensor_statistics(torch.tensor([3.0, -4.0]))
    torch.testing.assert_close(stats, torch.tensor([25.0, 2.0, 4.0]))


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
        ("hf.layers.1.mlp.value.parametrizations.weight.original", (64, 256), 1.0, 1.0),
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


@pytest.mark.parametrize("magnitude", [1.0, 1e-8, 5e-13, 2e-12, 0.0])
def test_replay_normalization_matches_piecewise_jacobian(magnitude):
    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast_replay.py"
    reference = runpy.run_path(str(entry))["reference_key"]
    key = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    key[..., 0] = magnitude
    key.requires_grad_()
    scale = torch.ones(4, dtype=torch.bfloat16)
    rate = torch.full_like(key, 0.5)
    _, negative_direction, _ = reference(key, scale, rate, scale, 4)
    # Tangential derivatives scale as 1/norm; the radial derivative vanishes
    # above epsilon and becomes 1/epsilon inside the clamped region.
    negative_direction.float().sum().backward()
    norm = key.detach().float().norm().item()
    expected = torch.full_like(key, -1 / max(norm, 1e-12))
    if norm > 1e-12:
        expected[..., 0] = 0
    torch.testing.assert_close(key.grad, expected, rtol=0.01, atol=0)


def test_replay_recurrent_state_orientation_and_token_order():
    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast_replay.py"
    reference = runpy.run_path(str(entry))["reference_recurrent"]
    r = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    w = torch.zeros_like(r)
    k = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    v = torch.tensor([[[5.0, 7.0], [11.0, 13.0]]])
    zero = torch.zeros_like(r)
    output, state = reference(r, w, k, v, zero, zero, head_size=2)
    decay = torch.exp(torch.tensor(-0.6065306597126334 / 2))
    initial = k[0, 0, :, None] * v[0, 0, None, :]
    final = initial * decay + k[0, 1, :, None] * v[0, 1, None, :]
    expected = torch.stack((r[0, 0] @ initial, r[0, 1] @ final)).bfloat16()[None]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[0, 0], final)
    output.float().sum().backward()
    torch.testing.assert_close(r.grad[0], torch.stack((initial.sum(-1), final.sum(-1))))


def test_readout_reference_head_normalization_and_rkv_gradients():
    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast_replay.py"
    reference = runpy.run_path(str(entry))["reference_readout"]
    x = torch.zeros(1, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    r = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)
    v = r.detach().clone().requires_grad_()
    k = torch.ones_like(r)
    residual = torch.tensor([[0.5, 0.25], [0.125, 0.25]], dtype=torch.bfloat16)
    weight = torch.full((4,), 2.0, dtype=torch.bfloat16)
    bias = torch.ones(4, dtype=torch.bfloat16)
    gate = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]], dtype=torch.bfloat16)
    result = reference(x, r, k, v, residual, weight, bias, gate, head_size=2)
    torch.testing.assert_close(result, torch.tensor([[[1.0, 1.5, 1.28125, 1.625]]], dtype=torch.bfloat16))
    (result[..., 0].float() + result[..., 2].float()).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[20.0, -20.0, 10.0, -10.0]]], dtype=torch.bfloat16))
    torch.testing.assert_close(v.grad, torch.tensor([[[0.5, 0.0, 0.34375, 0.0]]], dtype=torch.bfloat16))


@pytest.mark.parametrize("legacy", [False, True])
def test_lr_rescale_preserves_rwkv_ratios_adam_state_and_scheduler_cursor(legacy):
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    from megatron.bridge.recipes.rwkv7.forecast import ForecastOptimizerCallback

    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter],
                "max_lr": 1e-4 * (1 if legacy else multiplier),
                "min_lr": 1e-5 * (1 if legacy else multiplier),
                "lr_mult": multiplier,
                "wd_mult": 0.0,
            }
            for parameter, multiplier in zip(parameters, (1.0, 2.0))
        ]
    )
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    moments = [optimizer.state[p]["exp_avg"].clone() for p in parameters]
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0,
        max_lr=1e-4,
        min_lr=1e-5,
        lr_warmup_steps=100,
        lr_decay_steps=3592,
        lr_decay_style="cosine",
        start_wd=0.033,
        end_wd=0.033,
        wd_incr_steps=3592,
        wd_incr_style="constant",
    )
    scheduler.step(2500)
    base_lr = optimizer.param_groups[0]["lr"]
    context = SimpleNamespace(
        scheduler=scheduler,
        optimizer=optimizer,
        state=SimpleNamespace(cfg=SimpleNamespace(optimizer=SimpleNamespace(lr=1e-4, min_lr=1e-5))),
    )
    ForecastOptimizerCallback(0.5, rwkv_optimizer=True).on_train_start(context)
    assert scheduler.num_steps == 2500
    assert optimizer.param_groups[0]["lr"] == base_lr * 0.5
    assert optimizer.param_groups[1]["lr"] == base_lr
    assert optimizer.param_groups[0]["max_lr"] == 1e-4
    for parameter, moment in zip(parameters, moments):
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], moment, rtol=0, atol=0)
    saved = scheduler.state_dict()
    scheduler.num_steps = 0
    scheduler.load_state_dict(saved)
    assert scheduler.num_steps == 2500
    assert optimizer.param_groups[0]["lr"] == base_lr * 0.5
    scheduler.step(1)
    assert optimizer.param_groups[1]["lr"] == optimizer.param_groups[0]["lr"] * 2


def test_checkpoint_fork_preserves_source_and_resumes_corrected_groups(tmp_path, monkeypatch):
    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast.py"
    train = runpy.run_path(str(entry))["train"]
    architecture = tmp_path / "architecture.json"
    RwkvConfig(hidden_size=64, num_hidden_layers=2, context_length=16).to_json_file(architecture)
    args = SimpleNamespace(
        config=architecture,
        data=tmp_path / "data",
        output=tmp_path / "source",
        context=16,
        calibration_windows=0,
        normalize_inputs=False,
        head_fp32=False,
        rwkv_optimizer=True,
        center_channelmix=False,
        epochs=None,
        steps=20,
        eval_iters=0,
        warmup_steps=0,
        global_batch_size=2,
        micro_batch_size=1,
        learning_rate=1e-4,
        learning_rate_scale=1.0,
        save_interval=10,
        eval_interval=10,
        seed=2026,
        resume=None,
        fork_from=None,
        checkpoint_step=None,
        exit_interval=None,
        diagnostics_interval=0,
        spike_snapshot=False,
    )
    captured = []
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    with (
        patch("megatron.bridge.data.datasets.ohlcv.OHLCVWindowDataset", return_value=SimpleNamespace(num_windows=20)),
        patch("megatron.bridge.data.datasets.ohlcv.manifest_digest", return_value="fixed-manifest"),
        patch("megatron.bridge.training.utils.checkpoint_utils.checkpoint_exists", return_value=True),
        patch(
            "megatron.bridge.training.pretrain.pretrain", side_effect=lambda cfg, forward, **kw: captured.append(cfg)
        ),
    ):
        train(args)
        source = args.output
        legacy = json.loads((source / "run.json").read_text())
        legacy.pop("optimizer_policy")
        original = json.dumps(legacy)
        (source / "run.json").write_text(original)
        args.output = tmp_path / "continuation"
        args.fork_from, args.checkpoint_step, args.learning_rate_scale = source, 10, 0.5
        train(args)
        assert (source / "run.json").read_text() == original
        assert captured[-1].checkpoint.load == str(source / "checkpoints")
        assert captured[-1].checkpoint.ckpt_step == 10
        assert captured[-1].optimizer_config_override_provider.apply_lr_bounds is False
        saved = json.loads((args.output / "run.json").read_text())
        assert saved["optimizer_policy"] == "rwkv_lm_v2" and saved["learning_rate_scale"] == 0.5
        args.fork_from, args.checkpoint_step, args.resume = None, None, args.output / "checkpoints"
        train(args)
        assert captured[-1].optimizer_config_override_provider.apply_lr_bounds is True
