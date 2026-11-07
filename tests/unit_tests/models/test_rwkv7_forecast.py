"""Continuous task heads, state lifecycle, serialization and native loss counts."""

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
        optimizer_config=OptimizerConfig(),
        scheduler_config=SchedulerConfig(),
    )
    overrides = RWKV7ForecastOptimizerOverrides().build_config_overrides(context)
    parameter = torch.nn.Parameter(torch.empty(shape))
    matching = [value for key, value in overrides.items() if key.matches(parameter, name)]
    assert all(value.get("wd_mult", wd_mult) == wd_mult for value in matching)
    assert all(value.get("lr_mult", lr_mult) == lr_mult for value in matching)
    assert any(value.get("wd_mult") == 0.0 for value in matching) == (wd_mult == 0.0)
    assert any(value.get("lr_mult") == 2.0 for value in matching) == (lr_mult == 2.0)
