# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV-7 construction, loss alignment, conversion and LoRA contracts."""

import csv
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from transformers import RwkvConfig


pytest.importorskip("transformers.integrations.flash_rwkv2", reason="Requires transformers-rwkv RWKV-7")

from megatron.bridge import AutoBridge
from megatron.bridge.models.rwkv7 import RWKV7Bridge, RWKV7Model, RWKV7ModelBuilder, RWKV7ModelConfig
from megatron.bridge.peft.rwkv7 import RWKV7LoRA


pytestmark = pytest.mark.unit


@pytest.fixture
def hf_config():
    return RwkvConfig(
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=128,
        context_length=16,
        architectures=["RwkvForCausalLM"],
    )


def test_builder_config_roundtrip(hf_config):
    cfg = AutoBridge.from_hf_config(hf_config).get_model_config()
    assert isinstance(cfg, RWKV7ModelConfig)
    restored = RWKV7ModelConfig.from_dict(cfg.as_dict())
    assert restored.resolved_hf_config() == cfg.resolved_hf_config()
    assert RWKV7Bridge.megatron_to_hf_config(cfg)["architecture_version"] == "rwkv7"
    assert cfg.position_embedding_type == "none"
    assert not cfg.should_pad_vocab
    cfg.num_layers = 3
    assert cfg.resolved_hf_config()["num_hidden_layers"] == 3


@pytest.mark.parametrize(
    "field", ["tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"]
)
def test_reject_model_parallelism(hf_config, field):
    cfg = RWKV7Bridge().hf_config_to_model_config(hf_config)
    assert hasattr(cfg, field)
    setattr(cfg, field, 2)
    with pytest.raises(ValueError, match=field):
        RWKV7ModelBuilder(cfg).build_model(Mock())


def test_weight_mapping_covers_every_hf_parameter(hf_config):
    from transformers import RwkvForCausalLM

    reference = RwkvForCausalLM(hf_config)
    registry = RWKV7Bridge().mapping_registry()
    for name, _ in reference.named_parameters():
        mapping = registry.megatron_to_hf_lookup("hf." + name)
        assert mapping.hf_param == name


def test_checkpoint_backed_cpu_initialization(hf_config):
    cfg = RWKV7Bridge().hf_config_to_model_config(hf_config)
    cfg.perform_initialization = False
    cfg.use_cpu_initialization = True
    with patch("torch.cuda.current_device", side_effect=AssertionError("CPU construction must not initialize CUDA")):
        model = RWKV7Model(cfg, Mock())
    assert all(param.device.type == "cpu" and param.dtype == torch.bfloat16 for param in model.parameters())
    assert model.hf.model.embed_tokens.weight.shape == (hf_config.vocab_size, hf_config.hidden_size)


def test_labels_are_not_shifted_twice(hf_config):
    cfg = RWKV7Bridge().hf_config_to_model_config(hf_config)
    model = RWKV7Model(cfg, Mock())
    logits = torch.randn(2, 4, hf_config.vocab_size, requires_grad=True)
    labels = torch.randint(hf_config.vocab_size, (2, 4))
    with patch.object(model.hf, "forward", return_value=Mock(logits=logits)):
        loss = model(input_ids=labels, labels=labels)
    torch.testing.assert_close(loss, F.cross_entropy(logits.transpose(1, 2), labels, reduction="none"))
    loss.mean().backward()
    assert logits.grad is not None


@pytest.mark.parametrize("use_nora", [False, True])
def test_lora_fused_spec_and_checkpoint(hf_config, use_nora):
    from transformers.integrations.flash_rwkv2 import flash_rwkv2_linear_spec

    cfg = RWKV7Bridge().hf_config_to_model_config(hf_config)
    model = RWKV7Model(cfg, Mock())
    peft = RWKV7LoRA(dim=4, alpha=8, use_nora=use_nora)
    peft(model)
    peft.set_params_to_save(model)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert all("lora_A" in name or "lora_B" in name for name in trainable)
    assert peft.params_to_save == trainable
    layer = model.hf.model.layers[0].linear_attn.r_proj
    weight, a, b, scale, nora = flash_rwkv2_linear_spec(layer)
    assert a.shape == (4, 64) and b.shape == (64, 4)
    assert scale == 2 and nora == use_nora
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        b.fill_(0.1)
    a_compute = a.float()
    if use_nora:
        a_compute = a_compute / (a_compute.norm(dim=0, keepdim=True) + 1e-6)
    output = F.linear(x, weight) + scale * F.linear(F.linear(x, a_compute.to(x.dtype)), b)
    output.float().square().mean().backward()
    assert layer.lora_A["default"].weight.grad is not None
    assert layer.get_base_layer().weight.grad is None
    with peft.disable_adapter(model):
        assert flash_rwkv2_linear_spec(layer)[1] is None
    assert flash_rwkv2_linear_spec(layer)[1] is not None
    state = model.state_dict()
    restored = peft(RWKV7Model(cfg, Mock()))
    restored.load_state_dict(state)
    for name in trainable:
        torch.testing.assert_close(dict(restored.named_parameters())[name], dict(model.named_parameters())[name])
    model.merge_lora()
    assert all("lora_" not in name for name, _ in model.named_parameters())
    torch.testing.assert_close(model.hf.model.layers[0].linear_attn.r_proj(x), output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("mode", ["scratch", "full", "fsdp", "lora", "nora"])
@pytest.mark.parametrize("export", [False, True])
@pytest.mark.parametrize("profile", ["generic", "pro6000x8"])
def test_training_entry_checkpoint_and_precision(hf_config, tmp_path, monkeypatch, mode, export, profile):
    if profile == "pro6000x8":
        hf_config = RwkvConfig(hidden_size=4096, num_hidden_layers=61, vocab_size=65536, context_length=65536)
    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/train.py"
    config = tmp_path / "config.json"
    hf_config.to_json_file(config)
    argv = [
        str(entry),
        "--config",
        str(config),
        "--output",
        str(tmp_path),
        "--mock",
        "--profile",
        profile,
        "--steps",
        "8",
        "--save-interval",
        "4",
        "--eval-interval",
        "4",
        "--eval-iters",
        "1",
    ]
    if profile == "pro6000x8":
        argv.extend(["--seq-length", "65536"])
    if mode != "scratch":
        argv.extend(["--base", "initial-checkpoint"])
    if mode in {"full", "fsdp"}:
        argv.append("--full-parameter")
    if mode == "fsdp":
        argv.append("--megatron-fsdp")
    if mode in {"lora", "nora"}:
        argv.extend(["--lora-rank", "8"])
    if mode == "nora":
        argv.append("--nora")
    if export:
        argv.extend(["--export-hf", str(tmp_path / "hf-export")])
    monkeypatch.syspath_prepend(str(entry.parent))
    monkeypatch.setattr(sys, "argv", argv)
    with (
        patch("megatron.bridge.training.pretrain.pretrain") as pretrain,
        patch("megatron.bridge.training.finetune.finetune") as finetune,
    ):
        runpy.run_path(str(entry), run_name="__main__")
    call = (pretrain if mode == "scratch" else finetune).call_args
    cfg = call.args[0]
    if export:
        assert len(call.kwargs["callbacks"]) == 1
        assert call.kwargs["callbacks"][0].output == tmp_path / "hf-export"
    else:
        assert call.kwargs["callbacks"] is None
    assert cfg.checkpoint.save_interval == cfg.validation.eval_interval == 4
    assert cfg.validation.eval_iters == 1
    assert cfg.model.bf16 and cfg.model.resolved_hf_config()["wkv_mode"] == "fp32io16"
    if mode in {"lora", "nora"}:
        assert cfg.peft.dim == cfg.peft.alpha == 8
        assert cfg.peft.use_nora == (mode == "nora")
    else:
        assert cfg.peft is None
    if mode != "scratch":
        assert cfg.checkpoint.pretrained_checkpoint == "initial-checkpoint"
        assert not cfg.model.perform_initialization
    if mode == "fsdp" or (profile == "pro6000x8" and mode in {"scratch", "full"}):
        assert cfg.dist.use_megatron_fsdp and cfg.ddp.use_megatron_fsdp
        assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads"
        assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"
        assert cfg.ddp.megatron_fsdp_main_grads_dtype == torch.float32
        assert cfg.ddp.megatron_fsdp_grad_comm_dtype == torch.float32
        assert cfg.ddp.overlap_grad_reduce


_PERFORMANCE = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/performance-a18.csv"


@pytest.mark.parametrize("case", list(csv.DictReader(_PERFORMANCE.open())))
def test_pro6000_recipe_matches_measured_configuration(case):
    from megatron.bridge.recipes.rwkv7 import rwkv7_8gpu_pro6000_config

    layers, hidden = {
        "rwkv-g1l-1b-temp-6331": (24, 2048),
        "rwkv-g1l-3b-temp-5764": (32, 2560),
        "rwkv-g1l-7b-temp-3797": (32, 4096),
        "rwkv-g1l-13b-temp-5497": (61, 4096),
    }[case["model"]]
    config = RwkvConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        vocab_size=65536,
        context_length=int(case["sequence_length"]),
    )
    adapter = (
        None
        if case["mode"] == "full"
        else RWKV7LoRA(
            dim=int(case["rank"]),
            alpha=int(case["alpha"]),
            use_nora=case["mode"] == "nora",
        )
    )
    cfg = rwkv7_8gpu_pro6000_config(config, pretrained_checkpoint="checkpoint", peft=adapter)
    assert cfg.peft is adapter
    assert not cfg.model.perform_initialization
    assert cfg.checkpoint.pretrained_checkpoint == "checkpoint"
    for field in ("micro_batch_size", "global_batch_size"):
        assert getattr(cfg.train, field) == int(case[field])
    for field in ("head_chunk_size", "infctx_chunk_size"):
        assert getattr(cfg.model, field) == int(case[field])
    for field in ("checkpoint_infctx", "checkpoint_layers", "checkpoint_tmix", "detach_infctx_state"):
        assert getattr(cfg.model, field) == (case[field] == "True")
    assert cfg.dist.use_megatron_fsdp == (case["use_megatron_fsdp"] == "True")
    assert cfg.optimizer.use_distributed_optimizer == (case["distributed_optimizer"] == "True")


def test_training_entry_explicit_overrides(tmp_path, monkeypatch):
    entry = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/train.py"
    config = tmp_path / "config.json"
    RwkvConfig(hidden_size=4096, num_hidden_layers=61, vocab_size=65536).to_json_file(config)
    monkeypatch.syspath_prepend(str(entry.parent))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(entry),
            "--config",
            str(config),
            "--output",
            str(tmp_path),
            "--mock",
            "--seq-length",
            "65536",
            "--no-megatron-fsdp",
            "--no-distributed-optimizer",
            "--micro-batch-size",
            "2",
            "--global-batch-size",
            "32",
            "--head-chunk-size",
            "512",
            "--infctx-chunk-size",
            "0",
            "--no-checkpoint-infctx",
            "--no-checkpoint-layers",
        ],
    )
    with patch("megatron.bridge.training.pretrain.pretrain") as pretrain:
        runpy.run_path(str(entry), run_name="__main__")
    cfg = pretrain.call_args.args[0]
    assert not cfg.dist.use_megatron_fsdp and not cfg.optimizer.use_distributed_optimizer
    assert (cfg.train.micro_batch_size, cfg.train.global_batch_size) == (2, 32)
    assert cfg.model.head_chunk_size == 512 and cfg.model.infctx_chunk_size == 0
    assert not cfg.model.checkpoint_infctx and not cfg.model.checkpoint_layers


@pytest.mark.parametrize("mode", ["full", "nora"])
@pytest.mark.parametrize("strategy", ["default", "infctx"])
def test_benchmark_uses_training_recipe(tmp_path, monkeypatch, mode, strategy):
    directory = Path(__file__).resolve().parents[3] / "examples/models/rwkv7"
    config = tmp_path / "config.json"
    RwkvConfig(hidden_size=4096, num_hidden_layers=61, vocab_size=65536).to_json_file(config)
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps([{"name": "13b", "config": str(config), "checkpoint": "checkpoint"}]))
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(directory / "benchmark.py"),
            "--models",
            str(manifest),
            "--output",
            str(tmp_path / "run"),
            "--modes",
            mode,
            "--seq-lengths",
            "65536",
            "--strategies",
            strategy,
        ],
    )

    def run_training(command, **kwargs):
        start = command.index("examples/models/rwkv7/train.py")
        with patch.object(sys, "argv", command[start:]):
            runpy.run_path(str(directory / "train.py"), run_name="__main__")
        output = Path(command[command.index("--output") + 1])
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "finite_losses": True,
                    "finite_grad_norms": True,
                    "skipped_rank_steps": 0,
                }
            )
        )
        return Mock(returncode=0)

    with (
        patch("subprocess.run", side_effect=run_training) as launch,
        patch("megatron.bridge.training.finetune.finetune") as finetune,
    ):
        runpy.run_path(str(directory / "benchmark.py"), run_name="__main__")
    cfg = finetune.call_args.args[0]
    command = launch.call_args.args[0]
    if strategy == "default":
        assert "--profile" not in command and "--micro-batch-size" not in command
        assert cfg.model.checkpoint_layers == (mode == "full")
        assert cfg.train.micro_batch_size == (1 if mode == "full" else 2)
        assert cfg.dist.use_megatron_fsdp == (mode == "full")
    else:
        assert command[command.index("--profile") + 1] == "generic"
        assert not cfg.model.checkpoint_layers and not cfg.dist.use_megatron_fsdp
        assert cfg.train.micro_batch_size == 1
    assert cfg.model.infctx_chunk_size == 1024 and cfg.model.checkpoint_infctx
    assert cfg.model.head_chunk_size == 1024
