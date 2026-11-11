# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV-7 construction, loss alignment, conversion and LoRA contracts."""

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


@pytest.mark.parametrize("use_nora", [False, True])
def test_native_adapter_export(hf_config, tmp_path, use_nora):
    from safetensors.torch import load_file

    bridge = AutoBridge.from_hf_config(hf_config)
    model = RWKV7Model(bridge.get_model_config(), Mock())
    peft = RWKV7LoRA(dim=4, alpha=8, use_nora=use_nora)
    peft(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(std=0.1)
    expected = {
        name.removeprefix("hf.").replace(".default.", "."): parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    assert len(expected) == 24
    exported = list(bridge.export_adapter_weights([model], show_progress=False, with_megatron_names=True))
    bridge.save_hf_adapter([model], tmp_path, peft, show_progress=False, base_model_name_or_path="rwkv7-base")
    assert {item.param_name for item in exported} == expected.keys()
    for item in exported:
        assert item.megatron_param_names == ("hf." + item.param_name.replace(".weight", ".default.weight"),)
        torch.testing.assert_close(item.weight, expected[item.param_name], rtol=0, atol=0)
    saved = load_file(tmp_path / "adapter_model.safetensors")
    assert saved.keys() == {"base_model.model." + name for name in expected}
    for name, parameter in expected.items():
        torch.testing.assert_close(saved["base_model.model." + name], parameter, rtol=0, atol=0)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["r"] == 4 and config["lora_alpha"] == 8
    assert config.get("use_nora") is (True if use_nora else None)
    assert config["base_model_name_or_path"] == "rwkv7-base"
    assert config["target_modules"] == sorted({name.rsplit(".", 2)[0] for name in expected})
    assert config["rank_pattern"] == {} and config["alpha_pattern"] == {}
    assert config["lora_dropout"] == 0.0

    restored = RWKV7Model(bridge.get_model_config(), Mock())
    peft(restored)
    missing, unexpected = restored.hf.load_state_dict(
        {name.removeprefix("base_model.model.").replace(".weight", ".default.weight"): p for name, p in saved.items()},
        strict=False,
    )
    assert not unexpected and all("lora_" not in name for name in missing)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    before = model.hf.model.layers[0].linear_attn.r_proj
    after = restored.hf.model.layers[0].linear_attn.r_proj
    after.base_layer.load_state_dict(before.base_layer.state_dict())
    torch.testing.assert_close(before(x), after(x), rtol=0, atol=0)


def test_native_adapter_export_excluded_layers(hf_config, tmp_path):
    from peft import PeftModel
    from transformers import RwkvForCausalLM

    bridge = AutoBridge.from_hf_config(hf_config)
    model = RWKV7Model(bridge.get_model_config(), Mock())
    base = RwkvForCausalLM(hf_config).to(dtype=torch.bfloat16)
    base.load_state_dict(model.hf.state_dict())
    peft = RWKV7LoRA(dim=4, alpha=8)
    peft(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(std=0.1)
    bridge.save_hf_adapter(
        [model], tmp_path, peft, show_progress=False, exclude_adapter_base_prefixes=["hf.model.layers.0."]
    )

    restored = PeftModel.from_pretrained(base, tmp_path, is_trainable=True, autocast_adapter_dtype=False)
    restored_base = restored.get_base_model()
    trainable = {name for name, parameter in restored_base.named_parameters() if parameter.requires_grad}
    expected = {
        name: parameter
        for name, parameter in model.hf.named_parameters()
        if name.startswith("model.layers.1.") and parameter.requires_grad
    }
    assert len(expected) == 12
    assert trainable == expected.keys()
    for name, parameter in expected.items():
        torch.testing.assert_close(restored_base.get_parameter(name), parameter, rtol=0, atol=0)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(
        restored_base.model.layers[1].linear_attn.r_proj(x),
        model.hf.model.layers[1].linear_attn.r_proj(x),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("use_nora", [False, True])
def test_native_merged_export(hf_config, use_nora):
    from megatron.bridge.training.model_load_save import temporary_distributed_context

    bridge = AutoBridge.from_hf_config(hf_config)
    model = RWKV7Model(bridge.get_model_config(), pg_collection=None)
    expected = {name: parameter.detach().clone() for name, parameter in model.hf.named_parameters()}
    RWKV7LoRA(dim=4, alpha=8, use_nora=use_nora)(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(std=0.1)
        merged = model.hf.model.layers[0].linear_attn.r_proj
        merged.merge()
        for name, module in model.hf.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            weight_name = f"{name}.weight"
            if module is merged:
                expected[weight_name] = module.base_layer.weight.detach().clone()
            else:
                a = module.lora_A["default"].weight.float()
                if use_nora:
                    a = a / (a.norm(dim=0, keepdim=True) + 1e-6)
                weight = expected[weight_name]
                delta = 2 * (module.lora_B["default"].weight.float() @ a)
                expected[weight_name] = (weight.float() + delta).to(weight.dtype)
    with temporary_distributed_context():
        exported = list(bridge.export_hf_weights([model], show_progress=False, cpu=True))
    assert {item.param_name for item in exported} == expected.keys()
    for item in exported:
        torch.testing.assert_close(item.weight, expected[item.param_name], rtol=0, atol=0)
    assert len(list(bridge.export_adapter_weights([model], show_progress=False))) == 24


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


@pytest.mark.parametrize("context", [4096, 16384, 65536])
@pytest.mark.parametrize(
    ("layers", "hidden", "peft", "layouts"),
    [
        (24, 2048, False, [(2, "none", 0), (1, "tmix", 0), (1, "infctx", 4096)]),
        (32, 2560, False, [(1, "none", 0), (1, "layers", 0), (1, "infctx", 2048)]),
        (32, 4096, False, [(1, "layers", 0), (1, "layers", 0), (1, "infctx", 1024)]),
        (61, 4096, False, [(1, "layers", 0), (1, "layers", 0), (1, "infctx", 1024)]),
        (24, 2048, True, [(4, "none", 0), (1, "none", 0), (4, "infctx", 4096)]),
        (32, 2560, True, [(2, "none", 0), (2, "tmix", 0), (2, "infctx", 4096)]),
        (32, 4096, True, [(1, "none", 0), (1, "tmix", 0), (3, "infctx", 2048)]),
        (61, 4096, True, [(2, "tmix", 0), (3, "layers", 0), (2, "infctx", 1024)]),
    ],
)
def test_pro6000_recipe_defaults(layers, hidden, peft, layouts, context):
    from megatron.bridge.recipes.rwkv7 import rwkv7_8gpu_pro6000_config

    config = RwkvConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        vocab_size=65536,
        context_length=context,
    )
    adapter = RWKV7LoRA(dim=32, alpha=32, use_nora=True) if peft else None
    cfg = rwkv7_8gpu_pro6000_config(config, pretrained_checkpoint="checkpoint", peft=adapter)
    microbatch, strategy, chunk = layouts[(4096, 16384, 65536).index(context)]
    assert cfg.peft is adapter
    assert not cfg.model.perform_initialization
    assert cfg.checkpoint.pretrained_checkpoint == "checkpoint"
    assert cfg.train.micro_batch_size == microbatch and cfg.train.global_batch_size == 8 * microbatch
    assert cfg.model.head_chunk_size == 1024 and cfg.model.infctx_chunk_size == chunk
    assert cfg.model.checkpoint_infctx == (strategy == "infctx")
    assert cfg.model.checkpoint_tmix == (strategy == "tmix")
    assert cfg.model.checkpoint_layers == (strategy == "layers" or (not peft and layers == 61 and context == 65536))
    assert not cfg.model.detach_infctx_state
    assert cfg.dist.use_megatron_fsdp == (not peft and layers == 61)
    assert cfg.optimizer.use_distributed_optimizer == (not peft)


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
