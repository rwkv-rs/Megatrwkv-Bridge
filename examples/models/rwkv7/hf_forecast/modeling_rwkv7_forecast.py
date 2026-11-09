# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone forecasting composition of the public transformers-rwkv backbone."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, RwkvConfig
from transformers.models.rwkv.modeling_rwkv import RwkvModel
from transformers.utils import ModelOutput

from .configuration_rwkv7_forecast import RWKV7ForecastConfig


@dataclass
class RWKV7ForecastOutput(ModelOutput):
    """FP32 cumulative-return quantiles and optional prices and pinball loss."""

    loss: torch.Tensor | None = None
    raw_quantiles: torch.Tensor | None = None
    quantiles: torch.Tensor | None = None
    price_quantiles: torch.Tensor | None = None


class RWKV7ForQuantileForecasting(PreTrainedModel):
    """Forecast all horizons directly from a fresh RWKV7 history window.

    This model reuses transformers-rwkv and FlashRWKV2 without Megatron imports.
    Features are unnormalized encoded inputs; normalization is part of the model.
    """

    config_class = RWKV7ForecastConfig
    base_model_prefix = "hf"
    main_input_name = "features"
    _no_split_modules = ["RwkvDecoderLayer"]

    def __init__(self, config: RWKV7ForecastConfig) -> None:
        super().__init__(config)
        backbone = RwkvConfig(**config.backbone_config)
        if getattr(backbone, "architecture_version", None) != "rwkv7":
            raise ValueError("Install the pinned transformers-rwkv dependency from requirements.txt.")
        self.hf = RwkvModel(backbone)
        self.hf.embed_tokens = nn.Identity()
        self.input_size = backbone.forecast_input_size
        self.horizon = backbone.forecast_horizon
        self.quantile_levels = tuple(backbone.forecast_quantiles)
        self.input_projection = nn.Linear(self.input_size, backbone.hidden_size)
        self.output_projection = nn.Linear(backbone.hidden_size, self.horizon * len(self.quantile_levels))
        # Native BF16 training rounded these constants before FP32 normalization.
        self.register_buffer("feature_mean", torch.tensor(backbone.forecast_input_mean, dtype=torch.bfloat16))
        self.register_buffer("feature_std", torch.tensor(backbone.forecast_input_std, dtype=torch.bfloat16))
        self.post_init()

    def forward(
        self,
        features: torch.Tensor,
        anchor_price: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        fix_quantile_crossing: bool = True,
        return_dict: bool = True,
    ) -> RWKV7ForecastOutput | tuple[torch.Tensor, ...]:
        """Predict ``[batch, 30, 9]`` scaled cumulative log returns.

        Args:
            features: Unnormalized encoded ``[batch, 64, input_size]`` history.
            anchor_price: Optional last observed close, shaped ``[batch]``.
            labels: Optional cumulative-return labels shaped ``[batch, 30]``.
            fix_quantile_crossing: Sort independently along the quantile axis.
            return_dict: Return a Transformers ModelOutput or its tuple.
        """
        context = self.config.backbone_config["context_length"]
        if features.ndim != 3 or features.shape[1:] != (context, self.input_size):
            raise ValueError(f"Expected [batch, {context}, {self.input_size}] unnormalized features.")
        inputs = (features.float() - self.feature_mean.float()) / self.feature_std.float()
        embedded = self.input_projection(inputs.to(self.input_projection.weight.dtype))
        hidden = self.hf(inputs_embeds=embedded, use_cache=False).last_hidden_state[:, -1]
        raw = F.linear(hidden.float(), self.output_projection.weight.float(), self.output_projection.bias.float())
        raw = raw.view(-1, self.horizon, len(self.quantile_levels))
        quantiles = raw.sort(dim=-1).values if fix_quantile_crossing else raw
        prices = None
        if anchor_price is not None:
            if anchor_price.shape != (features.shape[0],) or not torch.all(anchor_price > 0):
                raise ValueError("anchor_price must contain one positive last observed close per sample.")
            prices = anchor_price.double()[:, None, None] * torch.exp(quantiles.double() / 100)
        loss = None
        if labels is not None:
            if labels.shape != raw.shape[:2]:
                raise ValueError("labels must have shape [batch, forecast_horizon].")
            error = labels.float().unsqueeze(-1) - raw
            levels = raw.new_tensor(self.quantile_levels)
            loss = torch.maximum(levels * error, (levels - 1) * error).mean()
        result = RWKV7ForecastOutput(loss=loss, raw_quantiles=raw, quantiles=quantiles, price_quantiles=prices)
        return result if return_dict else result.to_tuple()

    @torch.inference_mode()
    def predict_quantiles(self, context: torch.Tensor, *, fix_quantile_crossing: bool = True) -> torch.Tensor:
        """Return FP32 cumulative-return quantiles from independent history."""
        return self(context, fix_quantile_crossing=fix_quantile_crossing).quantiles
