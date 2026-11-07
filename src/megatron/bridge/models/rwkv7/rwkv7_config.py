# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder configuration for the transformers-rwkv RWKV-7 implementation."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class RWKV7ModelConfig(BridgeGPTModelConfig):
    """RWKV-7 architecture plus Megatron training and serialization settings.

    The HF configuration stores RWKV-specific ranks and normalization settings;
    the nested transformer configuration owns distributed training settings.
    """

    builder: ClassVar[str] = "megatron.bridge.models.rwkv7.RWKV7ModelBuilder"
    hf_config: dict[str, Any] = field(default_factory=dict)
    head_chunk_size: int = 0
    """Rows per recomputed vocabulary projection; zero materializes full logits."""
    infctx_chunk_size: int = 0
    """Tokens per differentiable state-passing chunk; zero uses a single forward."""
    detach_infctx_state: bool = False
    """Truncate state gradients between chunks when explicitly requested."""
    checkpoint_infctx: bool = False
    """Recompute each state-passing chunk during backward."""
    checkpoint_layers: bool = False
    """Use the HF decoder's activation checkpointing implementation."""
    checkpoint_tmix: bool = False
    """Recompute TimeMix while retaining ChannelMix activations."""

    def resolved_hf_config(self) -> dict[str, Any]:
        """Apply declared architecture overrides to the HF construction config."""
        return {
            **self.hf_config,
            "num_hidden_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.ffn_hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "head_size": self.kv_channels,
            "vocab_size": self.vocab_size,
            "context_length": self.seq_length,
        }

    def _get_num_floating_point_operations_with_runtime_stats(
        self, batch_size: int, seqlen_sum: int | None = None, **kwargs: Any
    ) -> float:
        """Estimate full-parameter training FLOPs with linear sequence scaling.

        Uses forward plus two backward passes. Elementwise operations and LoRA
        GEMMs are omitted, so this estimate is unsuitable for kernel benchmarking.
        """
        c = self.resolved_hf_config()
        h, layers = c["hidden_size"], c["num_hidden_layers"]
        tokens = seqlen_sum if seqlen_sum is not None else batch_size * self.seq_length
        # Four TimeMix linears, two ChannelMix linears, and low-rank projections.
        matrices = layers * (4 * h * h + 2 * h * c["intermediate_size"])
        matrices += layers * 2 * h * (c["decay_low_rank_dim"] + c["a_low_rank_dim"])
        matrices += layers * 2 * h * c["gate_low_rank_dim"]
        matrices += (layers - 1) * 2 * h * c["v_low_rank_dim"]
        matrices += h * c["vocab_size"]
        # S*w + (S*a)*b^T + v*k^T, then S*r: two matrix-vector products
        # (4 FLOPs/entry), decay (1), two outer products (2), two adds (2).
        recurrent = layers * 9 * h * c["head_size"]
        return float(tokens * 3 * (2 * matrices + recurrent))
