# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure per-rank performance in the native Bridge training loop."""

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.bridge.training.callbacks import Callback, CallbackContext


class RWKV7PerformanceCallback(Callback):
    """Measure complete optimizer steps after a declared warmup window."""

    def __init__(self, output: Path, *, warmup: int) -> None:
        self.output = output
        self.warmup = warmup
        self.steps: list[dict] = []
        self.created = time.perf_counter()
        self.train_started: float | None = None

    def on_train_start(self, context: CallbackContext) -> None:
        """Separate model and data setup from training time."""
        self.train_started = time.perf_counter()

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Start timing after the previous iteration's GPU work completes."""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Record time, memory, loss, and optimizer update health on every rank."""
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "seconds": elapsed,
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_used_bytes": torch.cuda.device_memory_used(),
                "learning_rates": [group["lr"] for group in context.optimizer.param_groups],
                "consumed_samples": context.state.train_state.consumed_train_samples,
                "loss": {name: float(value) for name, value in context.loss_dict.items()},
                "grad_norm": float(context.grad_norm) if context.grad_norm is not None else None,
                "skipped": context.skipped_iter,
            }
        )

    def on_train_end(self, context: CallbackContext) -> None:
        """Save raw records and throughput computed from the slowest rank."""
        rank, world = dist.get_rank(), dist.get_world_size()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / f"rank-{rank}.json").write_text(json.dumps(self.steps, indent=2) + "\n")
        affinity = sorted(os.sched_getaffinity(0))
        (self.output / f"runtime-rank-{rank}.json").write_text(json.dumps({"cpu_affinity": affinity}, indent=2) + "\n")
        records: list = [None] * world
        dist.all_gather_object(records, self.steps)
        if rank:
            return
        seconds = [max(record[i]["seconds"] for record in records) for i in range(self.warmup, len(self.steps))]
        cfg = context.state.cfg
        all_steps = [step for record in records for step in record]
        root = Path(__file__).resolve().parents[3]
        source_files = sorted((root / "src/megatron/bridge/models/rwkv7").glob("*.py"))
        source_files += [root / "src/megatron/bridge/peft/rwkv7.py"]
        source_files += sorted((root / "src/megatron/bridge/recipes/rwkv7").glob("*.py"))
        source_files += sorted(Path(__file__).resolve().parent.glob("*.py"))
        source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
        summary = {
            "world_size": world,
            "gpu": torch.cuda.get_device_name(),
            "cpu_affinity_rank0": affinity,
            "torch": torch.__version__,
            "dependency_versions": {
                name: version(name)
                for name in ("transformers", "megatron-core", "transformer_engine", "flashrwkv2", "peft")
            },
            "rwkv_source_sha256": source_hash,
            "base_checkpoint": Path(cfg.checkpoint.pretrained_checkpoint).name
            if cfg.checkpoint.pretrained_checkpoint
            else None,
            "model_config": cfg.model.resolved_hf_config(),
            "sequence_length": cfg.model.seq_length,
            "global_batch_size": cfg.train.global_batch_size,
            "micro_batch_size": cfg.train.micro_batch_size,
            "warmup_steps": self.warmup,
            "measured_steps": len(seconds),
            "mean_step_seconds": statistics.mean(seconds),
            "median_step_seconds": statistics.median(seconds),
            "step_seconds_stdev": statistics.pstdev(seconds),
            "setup_seconds": self.train_started - self.created,
            "warmup_seconds": sum(max(record[i]["seconds"] for record in records) for i in range(self.warmup)),
            "tokens_per_second": cfg.train.global_batch_size * cfg.model.seq_length / statistics.mean(seconds),
            "peak_allocated_bytes_per_rank": [max(s["allocated_bytes"] for s in record) for record in records],
            "peak_reserved_bytes_per_rank": [max(s["reserved_bytes"] for s in record) for record in records],
            "peak_device_used_bytes_per_rank": [max(s["device_used_bytes"] for s in record) for record in records],
            "finite_grad_norms": all(s["grad_norm"] is not None and math.isfinite(s["grad_norm"]) for s in all_steps),
            "finite_losses": all(math.isfinite(v) for s in all_steps for v in s["loss"].values()),
            "skipped_rank_steps": sum(s["skipped"] for s in all_steps),
            "head_chunk_size": cfg.model.head_chunk_size,
            "infctx_chunk_size": cfg.model.infctx_chunk_size,
            "detach_infctx_state": cfg.model.detach_infctx_state,
            "checkpoint_infctx": cfg.model.checkpoint_infctx,
            "checkpoint_layers": cfg.model.checkpoint_layers,
            "checkpoint_tmix": cfg.model.checkpoint_tmix,
            "use_megatron_fsdp": cfg.dist.use_megatron_fsdp,
            "sharding_strategy": cfg.ddp.data_parallel_sharding_strategy,
            "distributed_optimizer": cfg.optimizer.use_distributed_optimizer,
            "overlap_grad_reduce": cfg.ddp.overlap_grad_reduce,
            "fsdp_main_grads_dtype": str(cfg.ddp.megatron_fsdp_main_grads_dtype),
            "fsdp_grad_comm_dtype": str(cfg.ddp.megatron_fsdp_grad_comm_dtype),
            "gpu_power_limits": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,power.limit", "--format=csv,noheader"], text=True, timeout=10
            ).splitlines(),
            "peft": None
            if cfg.peft is None
            else {
                "rank": cfg.peft.dim,
                "alpha": cfg.peft.alpha,
                "use_nora": cfg.peft.use_nora,
            },
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
