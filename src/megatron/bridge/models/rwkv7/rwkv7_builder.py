# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse Megatron's distributed construction lifecycle for RWKV-7 replicas."""

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.rwkv7.modeling_rwkv7 import RWKV7Model


class RWKV7ModelBuilder(GPTModelBuilder):
    """Build complete RWKV replicas, with standard hooks and DP wrapping."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> RWKV7Model:
        """Construct one replica after validating supported execution settings."""
        cfg = self._model_config
        for field in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
            if getattr(cfg, field) != 1:
                raise ValueError(f"RWKV7 currently requires {field}=1; use data parallelism.")
        if cfg.sequence_parallel or cfg.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("RWKV7 does not support sequence or virtual pipeline parallelism.")
        if cfg.cuda_graph_impl != "none" or cfg.gradient_accumulation_fusion:
            raise ValueError("RWKV7 requires cuda_graph_impl='none' and gradient_accumulation_fusion=False.")
        if not cfg.bf16 or cfg.fp16:
            raise ValueError("FlashRWKV2 training requires BF16.")
        if pre_process is False or post_process is False or vp_stage is not None:
            raise ValueError("RWKV7 builds a complete model replica.")
        device = (
            torch.device("cpu") if cfg.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
        )
        with device:
            return RWKV7Model(cfg, pg_collection)
