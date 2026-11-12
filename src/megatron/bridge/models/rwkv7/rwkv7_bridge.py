# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF config and lossless replicated weight mapping for RWKV-7."""

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import MegatronFSDP
from megatron.core.utils import unwrap_model
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.conversion.peft_bridge import (
    AdapterWeight,
    AdapterWeightConversionTask,
    _split_hf_lora_weight_name,
)
from megatron.bridge.models.rwkv7.modeling_rwkv7 import RWKV7Model
from megatron.bridge.models.rwkv7.rwkv7_config import RWKV7ModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.peft.rwkv7 import _RWKV7LoRALinear


@MegatronModelBridge.register_bridge(source="RwkvForCausalLM", target=RWKV7Model, model_type="rwkv")
class RWKV7Bridge(MegatronModelBridge):
    """Bridge the RWKV-7 architecture provided by transformers-rwkv."""

    MODEL_CONFIG_CLASS = RWKV7ModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True
    ADDITIONAL_FILE_PATTERNS = ["*generation_config.json"]

    def _get_lora_unwrapped_name(self, megatron_param: str) -> str:
        return super()._get_lora_unwrapped_name(megatron_param).replace(".base_layer.", ".")

    def _is_adapter_param_name(self, param_name: str) -> bool:
        return param_name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))

    def _adapter_target_modules(self, adapter_weight_names: Iterable[str]) -> list[str]:
        """Limit PEFT reload to projections that have exported weights."""
        return sorted({_split_hf_lora_weight_name(name)[0] for name in adapter_weight_names})

    def _merge_lora_adapter_weights(
        self,
        megatron_model: list[RWKV7Model],
        converted_weights_dict: dict[str, torch.Tensor],
        adapter_weights: list[AdapterWeight],
    ) -> dict[str, torch.Tensor]:
        # Keep exported A unchanged; normalize only the NoRA merge computation.
        models = unwrap_model(megatron_model)
        weights = []
        for adapter in adapter_weights:
            module = models[adapter.linear_in_weight.vp_stage].get_submodule(adapter.global_base_prefix)
            if "default" in module.merged_adapters:
                continue
            if module.use_nora["default"]:
                a = adapter.linear_in_weight.weight.float()
                a = a / (a.norm(dim=0, keepdim=True) + 1e-6)
                adapter = replace(adapter, linear_in_weight=adapter.linear_in_weight._replace(weight=a))
            weights.append(adapter)
        if not weights:
            return converted_weights_dict
        return super()._merge_lora_adapter_weights(megatron_model, converted_weights_dict, weights)

    def build_adapter_conversion_tasks(
        self,
        megatron_model: RWKV7Model | list[RWKV7Model],
        exclude_adapter_base_prefixes: Iterable[str] | None = None,
    ) -> dict[str, list[AdapterWeightConversionTask]]:
        """Use native PEFT A/B parameters with Bridge's replicated export tasks.

        RWKV7 has one complete model replica per data-parallel rank. All ranks
        construct the same tasks and participate in AutoBridge export; its
        existing save path writes the adapter on rank zero.
        """
        models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        models = unwrap_model(models)
        excluded = tuple(exclude_adapter_base_prefixes or ())
        registry = self.mapping_registry()
        tasks = {}
        for vp_stage, model in enumerate(models):
            for prefix, module in model.named_modules():
                if not isinstance(module, _RWKV7LoRALinear) or prefix.startswith(excluded):
                    continue
                base_mapping = registry.megatron_to_hf_lookup(f"{prefix}.weight")
                hf_prefix = base_mapping.hf_param.removesuffix(".weight")
                factors = []
                for factor in ("lora_A", "lora_B"):
                    linear = getattr(module, factor)["default"]
                    name = f"{prefix}.{factor}.default.weight"
                    factors.append(
                        WeightConversionTask(
                            param_name=name,
                            global_param_name=name,
                            mapping=ReplicatedMapping(megatron_param=name, hf_param=f"{hf_prefix}.{factor}.weight"),
                            pp_rank=0,
                            vp_stage=vp_stage,
                            megatron_module=linear,
                            param_weight=linear.weight,
                        )
                    )
                tasks[prefix] = [
                    AdapterWeightConversionTask(
                        global_base_prefix=prefix,
                        adapter_key=None,
                        alpha=module.lora_alpha["default"],
                        dim=module.r["default"],
                        linear_in_task=factors[0],
                        linear_out_task=factors[1],
                    )
                ]
        return tasks

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
