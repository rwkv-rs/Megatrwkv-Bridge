# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF config and lossless replicated weight mapping for RWKV-7."""

from typing import Any

import torch
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import MegatronFSDP
from megatron.core.utils import unwrap_model
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.rwkv7.modeling_rwkv7 import RWKV7Model
from megatron.bridge.models.rwkv7.rwkv7_config import RWKV7ModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


@MegatronModelBridge.register_bridge(source="RwkvForCausalLM", target=RWKV7Model, model_type="rwkv")
class RWKV7Bridge(MegatronModelBridge):
    """Bridge the RWKV-7 architecture provided by transformers-rwkv."""

    MODEL_CONFIG_CLASS = RWKV7ModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True
    ADDITIONAL_FILE_PATTERNS = ["*generation_config.json"]

    def load_weights_hf_to_megatron(
        self, hf_pretrained: Any, megatron_model: Any, allowed_mismatched_params: list[str] | None = None
    ) -> list:
        """Load sharded main weights when native checkpoint loading unwraps FSDP.

        The checkpoint loader passes the inner MegatronFSDP, whose module is
        the mixed-precision wrapper. Import into its RWKV model's distributed
        parameters, then refresh the BF16 compute weights from the main shards.
        """
        models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        models = unwrap_model(models)
        targets = [unwrap_model(model.module) if isinstance(model, MegatronFSDP) else model for model in models]
        super().load_weights_hf_to_megatron(hf_pretrained, targets, allowed_mismatched_params)
        for model in models:
            if isinstance(model, MegatronFSDP):
                model.install_optimized_model_weights()
        return megatron_model if isinstance(megatron_model, list) else [megatron_model]

    def hf_config_to_model_config(self, hf_config: PretrainedConfig) -> RWKV7ModelConfig:
        """Preserve all RWKV architecture fields and configure BF16 training."""
        if getattr(hf_config, "architecture_version", None) != "rwkv7":
            raise ValueError("Expected transformers-rwkv architecture_version='rwkv7'.")
        hf_config.validate_architecture()
        return RWKV7ModelConfig(
            hf_config=hf_config.to_dict(),
            transformer=TransformerConfig(
                num_layers=hf_config.num_hidden_layers,
                hidden_size=hf_config.hidden_size,
                num_attention_heads=hf_config.num_attention_heads,
                ffn_hidden_size=hf_config.intermediate_size,
                kv_channels=hf_config.head_size,
                normalization="LayerNorm",
                layernorm_epsilon=hf_config.layer_norm_epsilon,
                params_dtype=torch.bfloat16,
                bf16=True,
                gradient_accumulation_fusion=False,
            ),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.context_length,
            should_pad_vocab=False,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )

    @classmethod
    def megatron_to_hf_config(cls, model_config: RWKV7ModelConfig) -> dict[str, Any]:
        """Restore the original HF architecture for export."""
        return model_config.resolved_hf_config()

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map unchanged HF parameter names under the replica's ``hf`` prefix."""
        names = ["model.embed_tokens.weight", "lm_head.weight"]
        for norm in ("model.embedding_norm", "model.norm"):
            names.extend(f"{norm}.{param}" for param in ("weight", "bias"))
        for norm in ("input_layernorm", "post_attention_layernorm", "linear_attn.g_norm"):
            names.extend(f"model.layers.*.{norm}.{param}" for param in ("weight", "bias"))
        attention_params = (
            "x_r",
            "x_w",
            "x_k",
            "x_v",
            "x_a",
            "x_g",
            "w0",
            "w1",
            "w2",
            "a0",
            "a1",
            "a2",
            "v0",
            "v1",
            "v2",
            "g1",
            "g2",
            "k_k",
            "k_a",
            "r_k",
            "r_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
        )
        names.extend(f"model.layers.*.linear_attn.{param}" for param in attention_params)
        names.extend(f"model.layers.*.mlp.{param}" for param in ("x_k", "key.weight", "value.weight"))
        return MegatronMappingRegistry(
            *(ReplicatedMapping(megatron_param=f"hf.{name}", hf_param=name) for name in names)
        )
