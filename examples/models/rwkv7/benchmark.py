# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run bounded RWKV-7 performance cases through the native Bridge entry point."""

import argparse
import hashlib
import itertools
import json
import logging
import subprocess
import time
from pathlib import Path


logger = logging.getLogger(__name__)


def main() -> None:
    """Screen execution choices or validate selected cases with a longer budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path, help="JSON list of name, checkpoint and config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32])
    parser.add_argument("--modes", choices=["full", "lora", "nora"], nargs="+", default=["nora"])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[4096])
    parser.add_argument(
        "--strategies", choices=["default", "none", "tmix", "layers", "infctx"], nargs="+", default=["default"]
    )
    parser.add_argument("--megatron-fsdp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint-layers", action="store_true", help="Also checkpoint layers inside infctx chunks")
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[None])
    parser.add_argument("--global-batch-size", type=int, help="Fixed global batch; otherwise use 8 * microbatch")
    parser.add_argument("--head-chunk-size", type=int)
    parser.add_argument("--infctx-chunk-size", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps <= args.warmup:
        parser.error("--steps must exceed --warmup")
    if args.megatron_fsdp and args.modes != ["full"]:
        parser.error("--megatron-fsdp requires --modes full")
    if any(mbs is not None and mbs <= 0 for mbs in args.micro_batches):
        parser.error("micro batch sizes must be positive")
    if args.global_batch_size is not None and any(
        args.global_batch_size <= 0 or (mbs is not None and args.global_batch_size % (8 * mbs))
        for mbs in args.micro_batches
    ):
        parser.error("global batch size must be divisible by 8 * each micro batch size")
    models = json.loads(args.models.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    source_files = sorted((root / "src/megatron/bridge/models/rwkv7").glob("*.py"))
    source_files += [root / "src/megatron/bridge/peft/rwkv7.py"]
    source_files += sorted((root / "src/megatron/bridge/recipes/rwkv7").glob("*.py"))
    source_files += sorted(Path(__file__).resolve().parent.glob("*.py"))
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
    cases = (
        (model, seq, mode, rank, strategy, mbs)
        for model, seq, mode, strategy, mbs in itertools.product(
            models, args.seq_lengths, args.modes, args.strategies, args.micro_batches
        )
        for rank in ([0] if mode == "full" else args.ranks)
    )
    for model, seq, mode, rank, strategy, mbs in cases:
        global_batch_size = (
            args.global_batch_size if args.global_batch_size is not None else 8 * mbs if mbs is not None else None
        )
        case = f"{model['name']}-s{seq}-{mode}-r{rank}-{strategy}-mb{mbs}-gb{global_batch_size}"
        case += f"-head{args.head_chunk_size}"
        if mode == "full":
            case += (
                "-fsdp"
                if args.megatron_fsdp
                else "-distopt"
                if args.megatron_fsdp is False or strategy != "default"
                else "-default"
            )
        if strategy == "infctx":
            case += f"-chunk{args.infctx_chunk_size}"
        if args.checkpoint_layers and strategy != "layers":
            case += "-layers"
        output = args.output / case
        output.mkdir(parents=True, exist_ok=True)
        command = [
            "timeout",
            "--kill-after=30s",
            f"{args.timeout}s",
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=8",
            "--standalone",
            "examples/models/rwkv7/train.py",
            "--benchmark",
            "--mock",
            "--config",
            model["config"],
            "--base",
            model["checkpoint"],
            "--output",
            str(output),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--seq-length",
            str(seq),
            "--lora-rank",
            str(rank),
            "--lora-alpha",
            str(rank),
        ]
        if strategy != "default":
            command.extend(["--profile", "generic"])
        for flag, value in (
            ("--micro-batch-size", mbs if mbs is not None else (1 if strategy != "default" else None)),
            (
                "--global-batch-size",
                global_batch_size if global_batch_size is not None else (8 if strategy != "default" else None),
            ),
            (
                "--head-chunk-size",
                args.head_chunk_size
                if args.head_chunk_size is not None
                else (1024 if strategy != "default" else None),
            ),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        if mode == "nora":
            command.append("--nora")
        if mode == "full":
            command.append("--full-parameter")
            if args.megatron_fsdp is not None:
                command.append("--megatron-fsdp" if args.megatron_fsdp else "--no-megatron-fsdp")
            if not args.megatron_fsdp and (strategy != "default" or args.megatron_fsdp is False):
                command.append("--distributed-optimizer")
        if strategy == "layers" or args.checkpoint_layers:
            command.append("--checkpoint-layers")
        if strategy == "tmix":
            command.append("--checkpoint-tmix")
        if args.infctx_chunk_size is not None or strategy == "infctx":
            command.extend(
                ["--infctx-chunk-size", str(args.infctx_chunk_size if args.infctx_chunk_size is not None else 1024)]
            )
        if strategy == "infctx":
            command.append("--checkpoint-infctx")
        if args.resume and (output / "summary.json").exists() and (output / "execution.json").exists():
            previous = json.loads((output / "execution.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            if (
                previous["returncode"] == 0
                and previous["command"] == command
                and summary.get("rwkv_source_sha256") == source_hash
            ):
                logger.info("Already completed: %s", case)
                continue
        logger.info("Starting %s", case)
        (output / "summary.json").unlink(missing_ok=True)
        started = time.monotonic()
        with (output / "train.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout + 60)
            status = {"returncode": completed.returncode}
        status.update({"command": command, "elapsed_seconds": time.monotonic() - started})
        (output / "execution.json").write_text(json.dumps(status, indent=2) + "\n")
        logger.info("Finished %s: %s", case, status["returncode"])
        if completed.returncode:
            raise SystemExit(f"Stopping after execution failure; inspect {output / 'train.log'}")
        summary = json.loads((output / "summary.json").read_text())
        if not summary["finite_losses"] or not summary["finite_grad_norms"] or summary["skipped_rank_steps"]:
            raise SystemExit(f"Stopping after invalid training update; inspect {output / 'summary.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
