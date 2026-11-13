# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous next-bar and direct quantile forecasting with canonical RWKV-7."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from megatron.bridge.models.rwkv7.modeling_rwkv7 import RWKV7Model
from megatron.bridge.models.rwkv7.rwkv7_builder import RWKV7ModelBuilder
from megatron.bridge.models.rwkv7.rwkv7_config import RWKV7ModelConfig


@dataclass(kw_only=True)
class RWKV7ForecastConfig(RWKV7ModelConfig):
    """RWKV-7 with OHLCV input and a task-specific continuous forecasting head."""

    builder: ClassVar[str] = "megatron.bridge.models.rwkv7.modeling_rwkv7_forecast.RWKV7ForecastBuilder"

    def _get_num_floating_point_operations_with_runtime_stats(
        self, batch_size: int, seqlen_sum: int | None = None, **kwargs: Any
    ) -> float:
        total = float(super()._get_num_floating_point_operations_with_runtime_stats(batch_size, seqlen_sum, **kwargs))
        tokens = seqlen_sum if seqlen_sum is not None else batch_size * self.seq_length
        # Remove the token LM head; add continuous input and task output projections.
        inputs = self.hf_config.get("forecast_input_size", 5) + ("forecast_scaled_features" in self.hf_config)
        total += float(tokens * 6 * self.hidden_size * (inputs - self.vocab_size))
        if self.hf_config.get("forecast_task", "next_bar") == "direct_quantile":
            outputs = self.hf_config["forecast_horizon"] * len(self.hf_config["forecast_quantiles"])
            total += batch_size * 6 * self.hidden_size * outputs
            if self.hf_config.get("forecast_architecture") == "cnn_rwkv_attention":
                width = self.hidden_size
                total += tokens * 6 * inputs * (3 * 8 + 7 * width)
                total += batch_size * 6 * (inputs * (8 * 128 + 2 * 128**2 + 2 * 128) + 128**2 + 2 * width * 128)
            return total
        return total + float(tokens * 6 * self.hidden_size * 5)


class RWKV7ForecastHead(nn.Linear):
    """Compute a continuous regression head in FP32 with normal module hooks."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project in FP32 while preserving the parameter storage precision."""
        return F.linear(hidden.float(), self.weight.float(), self.bias.float())


class ForecastFeatureMixer(nn.Module):
    """Encode local patterns and mix named feature tokens at the forecast origin."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.conv = nn.Conv1d(input_size, input_size * 8, 3, groups=input_size, dtype=torch.bfloat16)
        self.project = nn.Linear(input_size * 8, hidden_size, dtype=torch.bfloat16)
        self.feature_projection = nn.Linear(8, 128, dtype=torch.bfloat16)
        self.feature_id = nn.Parameter(torch.empty(input_size, 128, dtype=torch.bfloat16))
        nn.init.normal_(self.feature_id, std=0.02)
        self.query = nn.Linear(hidden_size, 128, dtype=torch.bfloat16)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True, dtype=torch.bfloat16)
        self.output = nn.Linear(128, hidden_size, dtype=torch.bfloat16)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the RWKV input and the per-feature tokens at the last minute."""
        local = F.gelu(self.conv(F.pad(inputs.transpose(1, 2), (2, 0))))
        tokens = self.feature_projection(local[:, :, -1].reshape(-1, self.input_size, 8))
        return self.project(local.transpose(1, 2)), tokens + self.feature_id

    def forward(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Mix features using one query derived from the RWKV forecast state."""
        query = self.query(hidden).unsqueeze(1)
        mixed, _ = self.attention(query, tokens, tokens, need_weights=False)
        return hidden + self.output(mixed.squeeze(1))


class RWKV7ForecastModel(RWKV7Model):
    """Predict next-bar OHLCV features or direct cumulative close-return quantiles.

    The task owns its input projection and regression head. The backbone and
    recurrent state implementation remain those of transformers-rwkv. Training
    supplies already shifted labels; each independent window starts fresh.
    """

    def __init__(self, model_config: RWKV7ForecastConfig, pg_collection: Any) -> None:
        super().__init__(model_config, pg_collection)
        self.hf = self.hf.model
        self.hf.embed_tokens = nn.Identity()
        self.input_size = getattr(self.hf.config, "forecast_input_size", 5)
        # Return-unit inputs are divided by the window's return RMS, whose log
        # is appended as one more channel.
        scaled_features = getattr(self.hf.config, "forecast_scaled_features", None)
        model_inputs = self.input_size + (scaled_features is not None)
        self.feature_mixer: ForecastFeatureMixer | None = None
        if getattr(self.hf.config, "forecast_architecture", "linear") == "cnn_rwkv_attention":
            self.feature_mixer = ForecastFeatureMixer(model_inputs, model_config.hidden_size)
        else:
            self.input_projection = nn.Linear(model_inputs, model_config.hidden_size, dtype=torch.bfloat16)
        self.direct_quantile = getattr(self.hf.config, "forecast_task", "next_bar") == "direct_quantile"
        if self.direct_quantile:
            self.horizon = self.hf.config.forecast_horizon
            self.quantile_levels = tuple(self.hf.config.forecast_quantiles)
        outputs = self.horizon * len(self.quantile_levels) if self.direct_quantile else 5
        head = (
            RWKV7ForecastHead
            if self.direct_quantile or getattr(self.hf.config, "forecast_head_fp32", False)
            else nn.Linear
        )
        self.output_projection = head(model_config.hidden_size, outputs, dtype=torch.bfloat16)
        # Chronos-Bolt-style instance scaling: the head predicts quantiles in
        # units of each window's historical close-return RMS times sqrt(h).
        self.volatility_floor = getattr(self.hf.config, "forecast_volatility_floor", None)
        if scaled_features is None:
            self.scaled_inputs = None
        else:
            self.register_buffer(
                "scaled_inputs",
                torch.tensor(
                    [index in scaled_features for index in range(self.input_size)],
                    device=self.output_projection.weight.device,
                ),
                persistent=False,
            )
        self.normalize_inputs = hasattr(self.hf.config, "forecast_input_mean")
        if self.normalize_inputs:
            # Explicit BF16 storage matches native wrapping and standalone reload.
            for name, values in (
                ("feature_mean", self.hf.config.forecast_input_mean),
                ("feature_std", self.hf.config.forecast_input_std),
            ):
                self.register_buffer(
                    name,
                    torch.tensor(values, device=self.output_projection.weight.device, dtype=torch.bfloat16),
                    persistent=False,
                )
        if getattr(self.hf.config, "forecast_initialization", "linear") == "calibrated":
            # Initialize in FP32 before rounding, as in RWKV-LM's generated weights.
            with torch.no_grad():
                # A separate constant direction keeps zero-volume, unchanged
                # bars away from the zero-variance input of embedding LayerNorm.
                if self.feature_mixer is None:
                    weight = torch.empty(
                        model_config.hidden_size,
                        model_inputs + 1,
                        device=self.input_projection.weight.device,
                        dtype=torch.float32,
                    )
                    nn.init.orthogonal_(weight, gain=(model_config.hidden_size / (model_inputs + 1)) ** 0.5)
                    self.input_projection.weight.copy_(weight[:, :model_inputs])
                    self.input_projection.bias.copy_(weight[:, model_inputs])
                weight = torch.empty_like(self.output_projection.weight, dtype=torch.float32)
                nn.init.orthogonal_(weight, gain=self.hf.config.forecast_head_gain)
                self.output_projection.weight.copy_(weight)
                if self.direct_quantile:
                    bias = torch.tensor(self.hf.config.forecast_target_quantiles, dtype=torch.float32).flatten()
                else:
                    bias = torch.tensor(self.hf.config.forecast_target_mean, dtype=torch.float32)
                    bias[2:].clamp_(min=torch.finfo(torch.float32).eps)
                    bias[2:] = bias[2:] + torch.log(-torch.expm1(-bias[2:]))
                self.output_projection.bias.copy_(bias)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        past_key_values: Any = None,
        return_channel_losses: bool = False,
    ) -> torch.Tensor:
        """Return predictions or FP32 losses for the configured forecast task.

        Args:
            features: Encoded bars with shape ``[batch, time, 5]``.
            labels: Already shifted ``[batch, time, 5]`` next-bar features, or
                ``[batch, horizon]`` cumulative close-return targets. With volatility
                scaling, quantile losses are computed in scaled units.
            return_channel_losses: Retain the channel or quantile dimension in losses.
            past_key_values: HF RWKV cache used only for autoregressive inference.
        """
        inputs = features.float()
        if self.scaled_inputs is not None:
            assert self.volatility_floor is not None
            inputs = forecast_scaled_inputs(inputs, self.scaled_inputs, floor=self.volatility_floor)
        if self.normalize_inputs:
            inputs = (inputs - self.feature_mean.float()) / self.feature_std.float()
        inputs = inputs.to(self.output_projection.weight.dtype)
        if self.feature_mixer is None:
            embeddings = self.input_projection(inputs)
        else:
            embeddings, feature_tokens = self.feature_mixer.encode(inputs)
        hidden = self.hf(
            inputs_embeds=embeddings,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        ).last_hidden_state
        if self.direct_quantile:
            state = hidden[:, -1]
            if self.feature_mixer is not None:
                state = self.feature_mixer(state, feature_tokens)
            prediction = self.output_projection(state).view(-1, self.horizon, len(self.quantile_levels)).float()
            if self.volatility_floor is not None:
                scale = forecast_window_scale(features, floor=self.volatility_floor, horizon=self.horizon)
                if labels is None:
                    return prediction * scale.unsqueeze(-1)
                labels = labels.float() / scale
            elif labels is None:
                return prediction
            losses = forecast_pinball(prediction, labels, self.quantile_levels)
            return losses if return_channel_losses else losses.mean(-1)
        raw = self.output_projection(hidden).float()
        # Positive wicks guarantee high >= max(open, close), low <= min(open, close).
        prediction = torch.cat((raw[..., :2], F.softplus(raw[..., 2:])), dim=-1)
        if labels is None:
            return prediction
        channel_losses = F.smooth_l1_loss(prediction, labels.float(), reduction="none")
        return channel_losses if return_channel_losses else channel_losses.mean(dim=-1)

    @torch.inference_mode()
    def predict_quantiles(self, context: torch.Tensor, *, fix_quantile_crossing: bool = True) -> torch.Tensor:
        """Predict cumulative close-return quantiles from fresh historical state.

        Returns FP32 ``[batch, horizon, quantile]`` values in scaled log-return
        units. Rearrangement sorts only the quantile axis, never forecast time.
        """
        if not self.direct_quantile:
            raise ValueError("predict_quantiles requires a direct_quantile checkpoint.")
        if context.ndim != 3 or context.shape[1] == 0 or context.shape[-1] != self.input_size:
            raise ValueError(f"Expected nonempty [batch, time, {self.input_size}] context.")
        prediction = self(context)
        return prediction.sort(dim=-1).values if fix_quantile_crossing else prediction

    @torch.inference_mode()
    def generate(self, context: torch.Tensor, *, horizon: int) -> torch.Tensor:
        """Prime fresh recurrent state, then feed back each generated bar.

        Returns encoded bars of shape ``[batch, horizon, 5]``. Call in eval mode.
        Only the context pass processes multiple bars; subsequent passes process
        one bar each and carry TokenShift/WKV state without a growing KV cache.
        """
        from transformers.models.rwkv.modeling_rwkv import RwkvCache

        if horizon <= 0 or context.ndim != 3 or context.shape[1] == 0 or context.shape[-1] != 5:
            raise ValueError("Expected nonempty [batch, time, 5] context and a positive horizon.")
        cache = RwkvCache(self.hf.config)
        prediction = self(context, past_key_values=cache)[:, -1:]
        bars = [prediction]
        for _ in range(1, horizon):
            prediction = self(prediction, past_key_values=cache)
            bars.append(prediction)
        return torch.cat(bars, dim=1)


def forecast_window_rms(features: torch.Tensor, *, floor: float) -> torch.Tensor:
    """Return each window's RMS of raw historical close returns, at least ``floor``."""
    return features[..., 1].float().square().mean(dim=-1).sqrt().clamp_min(floor)


def forecast_window_scale(features: torch.Tensor, *, floor: float, horizon: int) -> torch.Tensor:
    """Scale each horizon by the RMS of raw historical close returns and sqrt(h)."""
    steps = torch.arange(1, horizon + 1, device=features.device, dtype=torch.float32)
    return forecast_window_rms(features, floor=floor).unsqueeze(-1) * steps.sqrt()


def forecast_scaled_inputs(features: torch.Tensor, scaled: torch.Tensor, *, floor: float) -> torch.Tensor:
    """Divide return-unit channels by the window RMS and append its log as a channel.

    Inputs then share the output's per-window scale, as in Chronos-Bolt and
    DeepAR, while the log scale retains the volatility level itself.
    """
    rms = forecast_window_rms(features, floor=floor)
    divided = torch.where(scaled, features.float() / rms[:, None, None], features.float())
    level = rms.log()[:, None, None].expand(-1, features.shape[1], 1)
    return torch.cat((divided, level), dim=-1)


def forecast_pinball(predictions: torch.Tensor, labels: torch.Tensor, levels: tuple[float, ...]) -> torch.Tensor:
    """Return the unreduced FP32 pinball loss."""
    error = labels.float().unsqueeze(-1) - predictions
    quantiles = predictions.new_tensor(levels)
    return torch.maximum(quantiles * error, (quantiles - 1) * error)


class RWKV7ForecastBuilder(RWKV7ModelBuilder):
    """Apply the existing RWKV execution constraints to the forecasting model."""

    model_class = RWKV7ForecastModel
