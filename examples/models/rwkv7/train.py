# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch RWKV-7 token pretraining or LoRA continued pretraining."""

import argparse
from pathlib import Path

import torch
from transformers import RwkvConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.bridge.peft.rwkv7 import RWKV7LoRA
from megatron.bridge.recipes.rwkv7 import rwkv7_8gpu_pro6000_config, rwkv7_pretrain_config
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import MockGPTDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


class RWKV7ExportCallback(Callback):
    """Export final native training weights and merge adapters after saving."""

    def __init__(self, output: Path, *, hf_config: RwkvConfig, source: Path | None) -> None:
        self.output = output
        self.hf_config = hf_config
        self.source = source

    def on_train_end(self, context: CallbackContext) -> None:
        """Materialize compute weights and save a complete HF checkpoint."""
        if context.state.cfg.dist.use_megatron_fsdp:
            # The public forward lifecycle installs replicated BF16 compute weights
            # after the optimizer and checkpoint paths expose sharded main weights.
            with torch.no_grad():
                tokens = torch.zeros((1, 16), dtype=torch.long, device="cuda")
                for model in context.model:
                    model(input_ids=tokens)
        models = unwrap_model(context.model)
        if context.state.cfg.peft is not None:
            for model in models:
                model.merge_lora()
        bridge = AutoBridge.from_hf_config(self.hf_config)
        bridge.save_hf_pretrained(models, self.output, source_path=self.source)


def main() -> None:
    """Read an explicit local HF config and run a bounded training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Local RWKV-7 config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=["pro6000x8", "generic"], default="pro6000x8")
    parser.add_argument(
        "--export-hf", type=Path, help="Export final full weights, merging adapters, to this directory"
    )
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--mock", action="store_true")
    data.add_argument("--data", help="Prefix of indexed .bin/.idx token data")
    parser.add_argument("--base", help="Initial HF or Megatron checkpoint")
    parser.add_argument("--full-parameter", action="store_true", help="Update all parameters, including with --base")
    parser.add_argument(
        "--megatron-fsdp", action=argparse.BooleanOptionalAction, help="Shard gradients and optimizer with optim_grads"
    )
    parser.add_argument("--resume", help="Full training-state checkpoint directory")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, help="Defaults to --lora-rank")
    parser.add_argument("--nora", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save-interval", type=int, help="Defaults to --steps")
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--head-chunk-size", type=int)
    parser.add_argument("--infctx-chunk-size", type=int)
    parser.add_argument("--detach-infctx-state", action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint-infctx", action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint-tmix", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overlap-grad-reduce", action=argparse.BooleanOptionalAction)
    parser.add_argument("--distributed-optimizer", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.nora and not args.base:
        parser.error("--nora requires --base")
    if args.full_parameter and args.nora:
        parser.error("--nora cannot be combined with --full-parameter")
    if args.megatron_fsdp and args.base and not args.full_parameter:
        parser.error("--megatron-fsdp with --base requires --full-parameter")
    if args.steps <= 0 or args.eval_iters < 0 or args.eval_interval <= 0:
        parser.error("steps and eval interval must be positive; eval iters must be nonnegative")
    if args.save_interval is not None and args.save_interval <= 0:
        parser.error("--save-interval must be positive")
    if any(value is not None and value < 0 for value in (args.head_chunk_size, args.infctx_chunk_size)):
        parser.error("chunk sizes must be nonnegative")
    hf_config = RwkvConfig.from_json_file(args.config)
    if args.seq_length is not None:
        hf_config.context_length = args.seq_length
    elif args.profile == "pro6000x8":
        hf_config.context_length = 4096
    adapter = (
        RWKV7LoRA(
            dim=args.lora_rank,
            alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
            use_nora=args.nora,
        )
        if args.base and not args.full_parameter
        else None
    )
    recipe = rwkv7_8gpu_pro6000_config if args.profile == "pro6000x8" else rwkv7_pretrain_config
    try:
        cfg = recipe(
            hf_config,
            pretrained_checkpoint=args.base,
            peft=adapter,
            use_megatron_fsdp=args.megatron_fsdp if args.profile == "pro6000x8" else bool(args.megatron_fsdp),
        )
    except ValueError as error:
        parser.error(str(error))
    if args.profile == "generic":
        cfg.train.global_batch_size = 8
    if args.mock:
        cfg.dataset = MockGPTDatasetConfig(
            random_seed=1234,
            seq_length=hf_config.context_length,
            create_attention_mask=False,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            dataloader_type="single",
        )
    else:
        cfg.dataset.blend = ([args.data], None)
    output = Path(args.output)
    cfg.checkpoint.save = str(output / "checkpoints")
    cfg.checkpoint.load = args.resume
    cfg.checkpoint.save_interval = args.save_interval if args.save_interval is not None else args.steps
    cfg.logger.tensorboard_dir = str(output / "tb_logs")
    cfg.train.train_iters = args.steps
    if args.micro_batch_size is not None:
        cfg.train.micro_batch_size = args.micro_batch_size
    if args.global_batch_size is not None:
        cfg.train.global_batch_size = args.global_batch_size
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = args.steps
    cfg.validation.eval_iters = args.eval_iters
    cfg.validation.eval_interval = args.eval_interval
    for name in (
        "head_chunk_size",
        "infctx_chunk_size",
        "detach_infctx_state",
        "checkpoint_infctx",
        "checkpoint_layers",
        "checkpoint_tmix",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.model, name, value)
    if cfg.model.detach_infctx_state and not cfg.model.infctx_chunk_size:
        parser.error("--detach-infctx-state requires --infctx-chunk-size")
    if cfg.model.checkpoint_infctx and not cfg.model.infctx_chunk_size:
        parser.error("--checkpoint-infctx requires --infctx-chunk-size")
    if args.overlap_grad_reduce is not None:
        cfg.ddp.overlap_grad_reduce = args.overlap_grad_reduce
    if args.distributed_optimizer is not None:
        if cfg.dist.use_megatron_fsdp and not args.distributed_optimizer:
            parser.error("FSDP requires the distributed optimizer")
        cfg.ddp.use_distributed_optimizer = args.distributed_optimizer
        cfg.optimizer.use_distributed_optimizer = args.distributed_optimizer
    callbacks = []
    if args.export_hf:
        source = Path(args.config).parent
        callbacks.append(
            RWKV7ExportCallback(
                args.export_hf,
                hf_config=hf_config,
                source=source if (source / "tokenizer_config.json").is_file() else None,
            )
        )
    (finetune if args.base else pretrain)(cfg, forward_step, callbacks=callbacks or None)


if __name__ == "__main__":
    main()
