# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Megatron training interface around the canonical HF RWKV-7 computation."""

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch.utils.checkpoint import checkpoint

from megatron.bridge.models.rwkv7.rwkv7_config import RWKV7ModelConfig


class RWKV7Model(MegatronModule):
    """A complete RWKV-7 replica using transformers-rwkv and FlashRWKV2.

    Labels follow Megatron's convention: the dataset has already shifted them.
    Each forward starts with a fresh recurrent state. Continuous inputs can be
    supplied through ``inputs_embeds`` by a task-specific forward step.
    """

    def __init__(self, model_config: RWKV7ModelConfig, pg_collection: Any) -> None:
        super().__init__(config=model_config.transformer)
        from transformers import RwkvConfig, RwkvForCausalLM

        hf_config = RwkvConfig(**model_config.resolved_hf_config())
        if getattr(hf_config, "architecture_version", None) != "rwkv7":
            raise ValueError("RWKV7 requires transformers-rwkv with its RWKV-7 implementation.")
        if model_config.perform_initialization:
            self.hf = RwkvForCausalLM(hf_config).to(dtype=torch.bfloat16)
        else:
            with torch.device("meta"):
                self.hf = RwkvForCausalLM(hf_config).to(dtype=torch.bfloat16)
            device = (
                torch.device("cpu")
                if model_config.use_cpu_initialization
                else torch.device("cuda", torch.cuda.current_device())
            )
            self.hf.to_empty(device=device)
        self.head_chunk_size = model_config.head_chunk_size
        self.infctx_chunk_size = model_config.infctx_chunk_size
        self.detach_infctx_state = model_config.detach_infctx_state
        self.checkpoint_infctx = model_config.checkpoint_infctx
        if model_config.checkpoint_layers:
            self.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if model_config.checkpoint_tmix:
            for layer in self.hf.model.layers:
                layer.linear_attn.forward = partial(checkpoint, layer.linear_attn.forward, use_reentrant=False)
        self.pg_collection = pg_collection
        self.pre_process = self.post_process = True
        self.share_embeddings_and_output_weights = False
        self.vp_stage = None

    def set_input_tensor(self, input_tensor: Any) -> None:
        """Accept the pipeline scheduler's empty input on a complete replica."""
        self.input_tensor = input_tensor

    def merge_lora(self) -> None:
        """Merge LoRA into base projections before full-model HF export.

        This changes the model in place; save adapter checkpoints before merging.
        """
        from peft.tuners.lora.layer import Linear

        for module in list(self.hf.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, Linear):
                    child.merge()
                    setattr(module, name, child.get_base_layer())

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        packed_seq_params: Any = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return batch-major logits or unreduced, already-aligned token losses."""
        if packed_seq_params is not None or padding_mask is not None:
            raise ValueError("RWKV7 requires equal-length, unpacked sequences.")
        # GPT's triangular mask describes causality, which RWKV enforces itself.
        # HF accepts only a 2-D validity mask and rejects padding.
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Set dataset.create_attention_mask=False for RWKV7.")
        if self.infctx_chunk_size and labels is not None:
            if inputs_embeds is not None or attention_mask is not None:
                raise ValueError("State-passing training accepts unpacked token IDs with no padding mask.")
            losses, states = [], ()
            for start in range(0, input_ids.shape[1], self.infctx_chunk_size):
                args = (
                    input_ids[:, start : start + self.infctx_chunk_size],
                    labels[:, start : start + self.infctx_chunk_size],
                    *states,
                )
                if self.checkpoint_infctx and torch.is_grad_enabled():
                    loss, *states = checkpoint(self._forward_chunk, *args, use_reentrant=False)
                else:
                    loss, *states = self._forward_chunk(*args)
                losses.append(loss)
                if self.detach_infctx_state:
                    states = tuple(state.detach() for state in states)
            return torch.cat(losses, dim=1)
        if self.head_chunk_size and labels is not None:
            hidden = self.hf.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
            return self._head_loss(hidden, labels)
        logits = self.hf(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        if labels is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view_as(
            labels
        )

    def _head_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        from transformers.integrations.flash_rwkv2 import flash_rwkv2_linear_spec, load_flash_rwkv2

        weight, a, b, scale, nora = flash_rwkv2_linear_spec(self.hf.lm_head)
        flash = load_flash_rwkv2("pretrain_head_linear_bf16", hidden, "head projection")

        def project_loss(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            logits = flash.pretrain_head_linear_bf16(
                x.contiguous(), weight, lora_a=a, lora_b=b, lora_scale=scale, use_nora=nora
            )
            return F.cross_entropy(logits.float(), targets, reduction="none")

        flat_hidden, flat_labels = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)
        size = self.head_chunk_size or flat_hidden.shape[0]
        losses = []
        for start in range(0, flat_hidden.shape[0], size):
            args = (flat_hidden[start : start + size], flat_labels[start : start + size])
            if self.head_chunk_size and torch.is_grad_enabled():
                losses.append(checkpoint(project_loss, *args, use_reentrant=False))
            else:
                losses.append(project_loss(*args))
        return torch.cat(losses).view_as(labels)

    def _forward_chunk(self, tokens: torch.Tensor, labels: torch.Tensor, *states: torch.Tensor) -> tuple:
        from transformers.models.rwkv.modeling_rwkv import RwkvCache

        cache = RwkvCache(self.hf.config)
        for index in range(len(states) // 3):
            cache.update_layer(index, *states[index * 3 : index * 3 + 3], training=True)
        hidden = self.hf.model(input_ids=tokens, past_key_values=cache, use_cache=True).last_hidden_state
        next_states = tuple(
            state for index in range(len(self.hf.model.layers)) for state in cache.layer_states(index, hidden)
        )
        return self._head_loss(hidden, labels), *next_states

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None) -> dict:
        """Save every base and adapter parameter as a DP-replicated tensor."""
        return {
            prefix + name: make_sharded_tensor_for_checkpoint(param, prefix + name, prepend_offsets=sharded_offsets)
            for name, param in self.state_dict(keep_vars=True).items()
        }
