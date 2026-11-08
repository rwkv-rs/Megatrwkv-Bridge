# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous next-bar prediction using the canonical RWKV-7 backbone."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from megatron.bridge.models.rwkv7.modeling_rwkv7 import RWKV7Model
from megatron.bridge.models.rwkv7.rwkv7_builder import RWKV7ModelBuilder
from megatron.bridge.models.rwkv7.rwkv7_config import RWKV7ModelConfig


@dataclass(kw_only=True)
class RWKV7ForecastConfig(RWKV7ModelConfig):
    """RWKV-7 architecture with a five-channel continuous input and output."""

    builder: ClassVar[str] = "megatron.bridge.models.rwkv7.modeling_rwkv7_forecast.RWKV7ForecastBuilder"

    def _get_num_floating_point_operations_with_runtime_stats(
        self, batch_size: int, seqlen_sum: int | None = None, **kwargs: Any
    ) -> float:
        total = super()._get_num_floating_point_operations_with_runtime_stats(batch_size, seqlen_sum, **kwargs)
        tokens = seqlen_sum if seqlen_sum is not None else batch_size * self.seq_length
        # Replace the LM vocabulary head with 5->hidden and hidden->5 projections.
        return total + float(tokens * 6 * self.hidden_size * (10 - self.vocab_size))


class RWKV7ForecastHead(nn.Linear):
    """Compute a continuous regression head in FP32 with normal module hooks."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project in FP32 while preserving the parameter storage precision."""
        return F.linear(hidden.float(), self.weight.float(), self.bias.float())


class _CenteredChannelMixWeight(nn.Module):
    """Apply post-activation centering through the existing down-projection."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Project out the constant activation direction before BF16 rounding."""
        value = weight.float()
        return (value - value.mean(dim=-1, keepdim=True)).to(weight.dtype)


class RWKV7ForecastModel(RWKV7Model):
    """Predict encoded open/close returns, upper/lower wicks and log volume.

    The task owns its input projection and regression head. The backbone and
    recurrent state implementation remain those of transformers-rwkv. Training
    supplies already shifted labels; each independent window starts fresh.
    """

    def __init__(self, model_config: RWKV7ForecastConfig, pg_collection: Any) -> None:
        super().__init__(model_config, pg_collection)
        self.hf = self.hf.model
        self.hf.embed_tokens = nn.Identity()
        self.input_projection = nn.Linear(5, model_config.hidden_size, dtype=torch.bfloat16)
        head = RWKV7ForecastHead if getattr(self.hf.config, "forecast_head_fp32", False) else nn.Linear
        self.output_projection = head(model_config.hidden_size, 5, dtype=torch.bfloat16)
        self.normalize_inputs = hasattr(self.hf.config, "forecast_input_mean")
        if self.normalize_inputs:
            # Explicit BF16 storage matches native wrapping and standalone reload.
            for name, values in (
                ("feature_mean", self.hf.config.forecast_input_mean),
                ("feature_std", self.hf.config.forecast_input_std),
            ):
                self.register_buffer(
                    name,
                    torch.tensor(values, device=self.input_projection.weight.device, dtype=torch.bfloat16),
                    persistent=False,
                )
        if getattr(self.hf.config, "forecast_initialization", "linear") == "calibrated":
            # Initialize in FP32 before rounding, as in RWKV-LM's generated weights.
            with torch.no_grad():
                # A separate constant direction keeps zero-volume, unchanged
                # bars away from the zero-variance input of embedding LayerNorm.
                weight = torch.empty(
                    model_config.hidden_size, 6, device=self.input_projection.weight.device, dtype=torch.float32
                )
                nn.init.orthogonal_(weight, gain=(model_config.hidden_size / 6) ** 0.5)
                self.input_projection.weight.copy_(weight[:, :5])
                self.input_projection.bias.copy_(weight[:, 5])
                weight = torch.empty_like(self.output_projection.weight, dtype=torch.float32)
                nn.init.orthogonal_(weight, gain=self.hf.config.forecast_head_gain)
                self.output_projection.weight.copy_(weight)
                bias = torch.tensor(self.hf.config.forecast_target_mean, dtype=torch.float32)
                bias[2:].clamp_(min=torch.finfo(torch.float32).eps)
                bias[2:] = bias[2:] + torch.log(-torch.expm1(-bias[2:]))
                self.output_projection.bias.copy_(bias)
        if getattr(self.hf.config, "forecast_center_channelmix", False):
            for layer in self.hf.layers:
                parametrize.register_parametrization(layer.mlp.value, "weight", _CenteredChannelMixWeight())

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        past_key_values: Any = None,
        return_channel_losses: bool = False,
    ) -> torch.Tensor:
        """Return encoded predictions or FP32 per-bar Huber losses.

        Args:
            features: Encoded bars with shape ``[batch, time, 5]``.
            labels: Encoded next bars with the same shape, already shifted once.
            past_key_values: HF RWKV cache used only for autoregressive inference.
        """
        inputs = features.float()
        if self.normalize_inputs:
            inputs = (inputs - self.feature_mean.float()) / self.feature_std.float()
        hidden = self.hf(
            inputs_embeds=self.input_projection(inputs.to(self.input_projection.weight.dtype)),
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        ).last_hidden_state
        raw = self.output_projection(hidden).float()
        # Positive wicks guarantee high >= max(open, close), low <= min(open, close).
        prediction = torch.cat((raw[..., :2], F.softplus(raw[..., 2:])), dim=-1)
        if labels is None:
            return prediction
        channel_losses = F.smooth_l1_loss(prediction, labels.float(), reduction="none")
        return channel_losses if return_channel_losses else channel_losses.mean(dim=-1)

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


class RWKV7ForecastBuilder(RWKV7ModelBuilder):
    """Apply the existing RWKV execution constraints to the forecasting model."""

    model_class = RWKV7ForecastModel
