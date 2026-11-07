# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge-managed LoRA using the layer API consumed by FlashRWKV2."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import Linear
from torch import nn

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.module_matcher import ModuleMatcher


class _RWKV7LoRALinear(Linear):
    """Expose Bridge's adapter enable/disable API on a HF PEFT layer."""

    @property
    def _adapter_enabled(self) -> bool:
        return not self.disable_adapters

    def enable_adapter_layers(self) -> None:
        """Enable the adapter in fused RWKV projections."""
        self.enable_adapters(True)

    def disable_adapter_layers(self) -> None:
        """Disable the adapter in fused RWKV projections."""
        self.enable_adapters(False)

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """Merge using the same column-normalized A as the NoRA fused operator."""
        if not getattr(self, "use_nora", {}).get(adapter, False):
            return super().get_delta_weight(adapter)
        a = self.lora_A[adapter].weight.float()
        a = a / (a.norm(dim=0, keepdim=True) + 1e-6)
        return (self.lora_B[adapter].weight.float() @ a * self.scaling[adapter]).to(self.get_base_layer().weight.dtype)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Use normalized A for direct module calls as well as fused execution."""
        if not getattr(self, "use_nora", {}).get("default", False) or self.disable_adapters or self.merged:
            return super().forward(x, *args, **kwargs)
        a = self.lora_A["default"].weight.float()
        a = (a / (a.norm(dim=0, keepdim=True) + 1e-6)).to(x.dtype)
        b = self.lora_B["default"].weight.to(x.dtype)
        return self.base_layer(x) + F.linear(F.linear(x, a), b) * self.scaling["default"]


@dataclass
class RWKV7LoRA(PEFT, ModuleMatcher):
    """LoRA for fused TimeMix and ChannelMix projections.

    Uses the existing PEFT dependency's LoraLayer contract so FlashRWKV2 sees
    the base weight, A, B, and scaling independently. Dropout is zero because
    the fused training API does not implement LoRA dropout.
    """

    target_modules: list[str] = field(default_factory=lambda: ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"])
    dim: int = 32
    alpha: int = 32
    use_nora: bool = False

    def transform(self, module: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
        """Replace selected bias-free projections with FlashRWKV2-compatible LoRA."""
        if isinstance(module, _RWKV7LoRALinear) or self.match(module, name, prefix) is None:
            return module
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise ValueError("RWKV7LoRA targets must be bias-free nn.Linear projections.")
        result = _RWKV7LoRALinear(
            module,
            adapter_name="default",
            config=LoraConfig(r=self.dim, lora_alpha=self.alpha, lora_dropout=0.0, init_lora_weights=True),
            r=self.dim,
            lora_alpha=self.alpha,
        )
        result.use_nora = {"default": self.use_nora}
        return result
