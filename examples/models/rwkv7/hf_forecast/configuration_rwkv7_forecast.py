# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformers configuration for the continuous RWKV7 forecast head."""

from typing import Any

from transformers import PreTrainedConfig


class RWKV7ForecastConfig(PreTrainedConfig):  # type: ignore[no-untyped-call]  # HF's subclass hook is untyped.
    """Persist the complete RWKV backbone and forecasting feature contract."""

    model_type = "rwkv7_forecast"

    def __init__(self, backbone_config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.backbone_config = backbone_config if backbone_config is not None else {}
