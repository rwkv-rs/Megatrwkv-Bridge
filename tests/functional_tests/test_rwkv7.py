# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU RWKV-7 forward parity and fused LoRA optimizer-step checks."""

import pytest
import torch
import torch.nn.functional as F
from transformers import RwkvConfig, RwkvForCausalLM


pytest.importorskip("transformers.integrations.flash_rwkv2", reason="Requires transformers-rwkv RWKV-7")

from megatron.bridge.models.rwkv7 import RWKV7Bridge, RWKV7Model
from megatron.bridge.peft.rwkv7 import RWKV7LoRA


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and FlashRWKV2")


@pytest.mark.parametrize("use_nora", [False, True])
@pytest.mark.parametrize("checkpoint_strategy", ["none", "layers", "tmix"])
@pytest.mark.parametrize(("rank", "alpha"), [(8, 16), (8, 8), (16, 16), (32, 32)])
def test_rwkv7_parity_and_lora_step(use_nora, checkpoint_strategy, rank, alpha):
    torch.manual_seed(2026)
    config = RwkvConfig(
        hidden_size=128,
        num_hidden_layers=2,
        vocab_size=256,
        context_length=64,
        architectures=["RwkvForCausalLM"],
    )
    cfg = RWKV7Bridge().hf_config_to_model_config(config)
    cfg.checkpoint_layers = checkpoint_strategy == "layers"
    cfg.checkpoint_tmix = checkpoint_strategy == "tmix"
    model = RWKV7Model(cfg, pg_collection=None).cuda().train()
    reference = RwkvForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).train()
    reference.load_state_dict(model.hf.state_dict())
    tokens = torch.randint(config.vocab_size, (1, 64), device="cuda")
    labels = tokens.roll(-1, dims=1)
    expected = reference(input_ids=tokens, use_cache=False).logits.float()
    actual = model(input_ids=tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    RWKV7LoRA(dim=rank, alpha=alpha, use_nora=use_nora)(model)
    torch.testing.assert_close(model(input_ids=tokens), expected, rtol=0, atol=0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=1e-3)
    RWKV7LoRA(dim=rank, alpha=alpha, use_nora=use_nora)(reference)
    reference.load_state_dict(model.hf.state_dict())
    frozen_before = {
        name: param.detach().clone() for name, param in model.named_parameters() if not param.requires_grad
    }
    trainable = [param for param in model.parameters() if param.requires_grad]
    adapters_before = [param.detach().clone() for param in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    loss = model(input_ids=tokens, labels=labels).mean()
    assert torch.isfinite(loss)
    loss.backward()
    reference_logits = reference(input_ids=tokens, use_cache=False).logits.float()
    reference_loss = F.cross_entropy(reference_logits.flatten(0, 1), labels.flatten())
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    reference_loss.backward()
    _assert_gradients_match(reference, model.hf)
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in trainable)
    assert all(torch.isfinite(param.grad).all() for param in trainable if param.grad is not None)
    optimizer.step()
    assert any(not torch.equal(param, before) for param, before in zip(trainable, adapters_before, strict=True))
    for name, param in model.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(param, frozen_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_state_passing_preserves_loss_and_gradients(checkpoint_chunks, batch_size, checkpoint_layers):
    torch.manual_seed(2026)
    config = RwkvConfig(
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=128,
        context_length=32,
        architectures=["RwkvForCausalLM"],
    )
    cfg = RWKV7Bridge().hf_config_to_model_config(config)
    reference = RWKV7Model(cfg, pg_collection=None).cuda().train()
    with torch.no_grad():
        for layer in reference.hf.model.layers:
            layer.linear_attn.o_proj.weight.normal_(std=0.02)
            layer.mlp.value.weight.normal_(std=0.02)
    cfg.head_chunk_size = 8
    cfg.infctx_chunk_size = 16
    cfg.checkpoint_infctx = checkpoint_chunks
    cfg.checkpoint_layers = checkpoint_layers
    actual = RWKV7Model(cfg, pg_collection=None).cuda().train()
    actual.load_state_dict(reference.state_dict())
    tokens = torch.randint(config.vocab_size, (batch_size, 32), device="cuda")
    labels = tokens.roll(-1, dims=1)
    reference_loss = reference(tokens, labels=labels)
    actual_loss = actual(tokens, labels=labels)
    torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-5, atol=1e-5)
    _, *expected_states = reference._forward_chunk(tokens, labels)
    states = ()
    for start in range(0, tokens.shape[1], cfg.infctx_chunk_size):
        _, *states = actual._forward_chunk(
            tokens[:, start : start + cfg.infctx_chunk_size],
            labels[:, start : start + cfg.infctx_chunk_size],
            *states,
        )
    for index, (state, expected_state) in enumerate(zip(states, expected_states, strict=True)):
        if index % 3 == 1:
            assert state.dtype == expected_state.dtype == torch.float32
        torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-5)
    reference_loss.mean().backward()
    actual_loss.mean().backward()
    _assert_gradients_match(reference, actual)


def _assert_gradients_match(reference, actual):
    for (name, param), (other_name, other) in zip(
        reference.named_parameters(), actual.named_parameters(), strict=True
    ):
        assert name == other_name
        assert (other.grad is None) == (param.grad is None), name
        if param.grad is None:
            continue
        expected, result = param.grad.float(), other.grad.float()
        if not torch.count_nonzero(expected):
            torch.testing.assert_close(result, expected, rtol=0, atol=0, msg=name)
            continue
        # Chunked BF16 reductions and head VJPs have different rounding orders.
        # A matched FP64 head-backward swap isolates this from state propagation.
        relative_error = (result - expected).norm() / expected.norm()
        cosine = torch.nn.functional.cosine_similarity(result.flatten(), expected.flatten(), dim=0)
        assert relative_error < 0.01, (name, relative_error.item())
        assert cosine >= 0.9999, (name, cosine.item())
