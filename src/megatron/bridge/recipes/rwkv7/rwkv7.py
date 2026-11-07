# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV-7 recipes built from an explicit HF architecture."""

from copy import deepcopy

import torch
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.peft.rwkv7 import RWKV7LoRA
from megatron.bridge.recipes.gpt.vanilla_gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def rwkv7_pretrain_config(
    hf_config: PretrainedConfig,
    *,
    pretrained_checkpoint: str | None = None,
    peft: RWKV7LoRA | None = None,
    use_megatron_fsdp: bool = False,
) -> ConfigContainer:
    """Build a BF16 data-parallel pretraining config from a RWKV-7 architecture.

    Configure the data source, tokenizer, output directories and training budget
    before calling ``pretrain``. The default data config uses indexed token data.
    """
    cfg = vanilla_gpt_pretrain_config()
    hf_config = deepcopy(hf_config)
    hf_config.architectures = ["RwkvForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(hf_config).get_model_config()
    cfg.dataset.seq_length = hf_config.context_length
    cfg.dataset.create_attention_mask = False
    cfg.tokenizer.vocab_size = hf_config.vocab_size
    cfg.peft = peft
    cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint
    cfg.model.perform_initialization = pretrained_checkpoint is None
    cfg.optimizer.use_distributed_optimizer = False
    cfg.ddp.use_distributed_optimizer = False
    if use_megatron_fsdp:
        cfg.dist.use_megatron_fsdp = True
        cfg.ddp.use_megatron_fsdp = True
        cfg.ddp.data_parallel_sharding_strategy = "optim_grads"
        cfg.ddp.average_in_collective = False
        cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
        cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.float32
        cfg.ddp.overlap_grad_reduce = True
        cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    return cfg


def rwkv7_peft_config(
    hf_config: PretrainedConfig, *, pretrained_checkpoint: str, dim: int, alpha: int, use_nora: bool = False
) -> ConfigContainer:
    """Build continued pretraining with frozen base weights and RWKV-7 LoRA."""
    return rwkv7_pretrain_config(
        hf_config,
        pretrained_checkpoint=pretrained_checkpoint,
        peft=RWKV7LoRA(dim=dim, alpha=alpha, use_nora=use_nora),
    )


def rwkv7_8gpu_pro6000_config(
    hf_config: PretrainedConfig,
    *,
    pretrained_checkpoint: str | None = None,
    peft: RWKV7LoRA | None = None,
    use_megatron_fsdp: bool | None = None,
) -> ConfigContainer:
    """Build the measured RTX PRO 6000 Blackwell eight-GPU token training recipe.

    Supports the four g1l architectures identified in the RWKV7 example manifest,
    at 4096, 16384 and 65536 tokens. LoRA/NoRA ranks 8, 16 and 32 share the PEFT
    layout. Global batch is eight times microbatch; configure data, learning-rate
    schedule and training budget for the intended convergence experiment.
    """
    architectures = [(24, 2048), (32, 2560), (32, 4096), (61, 4096)]
    architecture = (hf_config.num_hidden_layers, hf_config.hidden_size)
    if (
        architecture not in architectures
        or hf_config.architecture_version != "rwkv7"
        or hf_config.vocab_size != 65536
        or hf_config.head_size != 64
        or hf_config.intermediate_size != 4 * hf_config.hidden_size
        or hf_config.context_length not in (4096, 16384, 65536)
    ):
        raise ValueError("PRO 6000 x8 requires a supported g1l architecture and 4096/16384/65536 context")
    index = architectures.index(architecture)
    context = (4096, 16384, 65536).index(hf_config.context_length)
    # Each entry is (microbatch, recomputation strategy, state-passing chunk).
    layouts = (
        [
            [(2, "none", 0), (1, "tmix", 0), (1, "infctx", 4096)],
            [(1, "none", 0), (1, "layers", 0), (1, "infctx", 2048)],
            [(1, "layers", 0), (1, "layers", 0), (1, "infctx", 1024)],
            [(1, "layers", 0), (1, "layers", 0), (1, "infctx", 1024)],
        ]
        if peft is None
        else [
            [(4, "none", 0), (1, "none", 0), (4, "infctx", 4096)],
            [(2, "none", 0), (2, "tmix", 0), (2, "infctx", 4096)],
            [(1, "none", 0), (1, "tmix", 0), (3, "infctx", 2048)],
            [(2, "tmix", 0), (3, "layers", 0), (2, "infctx", 1024)],
        ]
    )
    fsdp = peft is None and index == 3 if use_megatron_fsdp is None else use_megatron_fsdp
    if fsdp and peft is not None:
        raise ValueError("PRO 6000 x8 FSDP is configured for full-parameter training")
    cfg = rwkv7_pretrain_config(
        hf_config, pretrained_checkpoint=pretrained_checkpoint, peft=peft, use_megatron_fsdp=fsdp
    )
    microbatch, strategy, chunk = layouts[index][context]
    cfg.train.micro_batch_size = microbatch
    cfg.train.global_batch_size = 8 * microbatch
    cfg.model.head_chunk_size = 1024
    cfg.model.infctx_chunk_size = chunk
    cfg.model.checkpoint_infctx = strategy == "infctx"
    cfg.model.checkpoint_tmix = strategy == "tmix"
    cfg.model.checkpoint_layers = strategy == "layers" or (peft is None and index == 3 and context == 2)
    cfg.optimizer.use_distributed_optimizer = cfg.ddp.use_distributed_optimizer = peft is None
    return cfg
