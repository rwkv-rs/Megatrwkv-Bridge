# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay an exact forecast spike batch and inspect the normalized-key VJP."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from megatron.bridge.data.datasets.ohlcv import OHLCVWindowDataset, manifest_digest


logger = logging.getLogger(__name__)


def reference_key(
    key: torch.Tensor,
    scale: torch.Tensor,
    rate: torch.Tensor,
    rate_scale: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match prepare_key's FP32 arithmetic and three BF16 output boundaries."""
    scaled = (key.float() * scale.float()).unflatten(-1, (-1, head_size))
    direction = F.normalize(scaled, dim=-1, eps=1e-12).flatten(-2)
    return (
        (key.float() * (1 + (rate.float() - 1) * rate_scale.float())).bfloat16(),
        (-direction).bfloat16(),
        (direction * rate.float()).bfloat16(),
    )


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Report amplitude and direction differences without scale-free tolerances."""
    a, b = actual.float().flatten(), expected.float().flatten()
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
    }


def norm_summary(values: torch.Tensor) -> dict[str, float | int]:
    """Summarize per-head norms over one valid-position population."""
    flat = values.float().flatten()
    if flat.numel() == 0:
        return {"count": 0}
    return {
        "count": flat.numel(),
        "min": flat.min().item(),
        "p01": torch.quantile(flat, 0.01).item(),
        "median": flat.median().item(),
        "max": flat.max().item(),
    }


def reference_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate the FP32 key-by-value recurrence with BF16 token output."""
    rr, ww, kk, vv, aa, bb = [x.float().unflatten(-1, (-1, head_size)) for x in (r, w, k, v, a, b)]
    state = rr.new_zeros((rr.shape[0], rr.shape[2], head_size, head_size))
    outputs = []
    for t in range(r.shape[1]):
        decay = torch.exp(-0.6065306597126334 * torch.sigmoid(ww[:, t]))
        sa = (state * aa[:, t, :, :, None]).sum(-2)
        state = (
            state * decay[..., None]
            + bb[:, t, :, :, None] * sa[..., None, :]
            + kk[:, t, :, :, None] * vv[:, t, :, None, :]
        )
        outputs.append((state * rr[:, t, :, :, None]).sum(-2).bfloat16())
    return torch.stack(outputs, dim=1).flatten(-2), state


def reference_readout(
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gate: torch.Tensor,
    head_size: int = 64,
    eps: float = 64e-5,
) -> torch.Tensor:
    """Match FP32 head normalization, RKV residual, gate, then BF16 rounding."""
    values = x.float().unflatten(-1, (-1, head_size))
    centered = values - values.mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)
    residual = (r.float() * k.float() * r_k.float().flatten()).unflatten(-1, (-1, head_size)).sum(-1, keepdim=True)
    result = normalized.flatten(-2) * weight.float() + bias.float()
    result = result + (residual * v.float().unflatten(-1, (-1, head_size))).flatten(-2)
    return (result * gate.float()).bfloat16()


def main() -> None:
    """Reconstruct native sampler order and compare key branches on one GPU."""
    from forecast import load_forecaster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=6, help="Zero-based layer index")
    parser.add_argument("--reference-key", action="store_true", help="Swap only key preparation for the reference")
    parser.add_argument("--check-recurrent", action="store_true", help="Compare a captured sample's recurrent VJP")
    parser.add_argument("--check-readout", action="store_true", help="Compare readout and inspect head variances")
    parser.add_argument("--reference-readout", action="store_true", help="Swap the selected layer's readout")
    parser.add_argument("--check-norms", action="store_true", help="Trace all LayerNorm input/gradient scales")
    parser.add_argument("--microbatches", type=int, help="Bound the number of original rank microbatches replayed")
    parser.add_argument("--batch-step", type=int, help="One-based data step; defaults to snapshot diagnostic step")
    args = parser.parse_args()
    if args.microbatches is not None and args.microbatches <= 0:
        parser.error("--microbatches must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    model, payload = load_forecaster(args.snapshot)
    model.train()
    run = payload["run"]
    if manifest_digest(args.data) != run["manifest_sha256"]:
        raise ValueError("Prepared data does not match the spike snapshot.")
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    consumed = snapshot["consumed_samples_before_step"]
    if args.batch_step is not None:
        consumed = (args.batch_step - 1) * run["global_batch_size"]
    dataset = OHLCVWindowDataset(
        args.data,
        split="train",
        seq_length=run["hf_config"]["context_length"],
        stride=run["hf_config"]["context_length"] if run["epochs"] is not None else 1,
        keep_tail=run["epochs"] is not None,
        epochs=run["epochs"],
        seed=run["seed"],
    )
    import flashrwkv2.tmix.wkv_prepare.tilelang as prepare_module

    original = prepare_module.prepare_key
    captured: dict[str, Any] = {}
    head_size = model.hf.config.head_size

    def probe(
        key: torch.Tensor,
        scale: torch.Tensor,
        rate: torch.Tensor,
        rate_scale: torch.Tensor,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        implementation = reference_key if args.reference_key and captured.get("active", False) else original
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = implementation(key, scale, rate, rate_scale, size)
        if captured.get("active", False):
            captured["inputs"] = (key, scale, rate, rate_scale)
            captured["outputs"] = output
            for tensor in output:
                tensor.retain_grad()
        return output

    attention = model.hf.layers[args.layer].linear_attn

    def enter(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["active"] = True

    def leave(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        captured["active"] = False

    handles = [attention.register_forward_pre_hook(enter), attention.register_forward_hook(leave)]
    # The fused projection bypasses nn.Linear.forward; capture x_k at its public operator boundary.
    import flashrwkv2
    import flashrwkv2.tmix.readout.tilelang as readout_module

    original_prepare = flashrwkv2.pretrain_tmix_wkv_prepare_bf16
    original_recurrent = flashrwkv2.pretrain_tmix_wkv7_recurrent_bf16
    original_readout = readout_module.readout
    original_norm = flashrwkv2.infer_embedding_ln0_forward_varlen
    norm_names = {
        module.weight.data_ptr(): name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
    }
    norm_captures: dict[str, tuple] = {}

    def readout(*inputs: Any, **kwargs: Any) -> torch.Tensor:
        implementation = (
            reference_readout if args.reference_readout and captured.get("active", False) else original_readout
        )
        output = implementation(*inputs, **kwargs)
        if args.check_readout and captured.get("active", False):
            captured["readout_inputs"] = inputs[:8]
            captured["readout_output"] = output
            output.retain_grad()
        return output

    def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if not args.check_norms:
            return original_norm(x, weight, bias, **kwargs)
        # A view isolates the norm branch's VJP from the residual skip's VJP.
        branch = x.view_as(x)
        result = original_norm(branch, weight, bias, **kwargs)
        branch.retain_grad()
        result.retain_grad()
        norm_captures[norm_names[weight.data_ptr()]] = (branch, result, weight, bias, kwargs["eps"])
        return result

    def prepare(*inputs: torch.Tensor, **kwargs: Any) -> Any:
        if captured.get("active", False):
            captured["xk"] = inputs[2].detach()
        return original_prepare(*inputs, **kwargs)

    def recurrent(*inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        output: torch.Tensor = original_recurrent(*inputs, **kwargs)
        if args.check_recurrent and captured.get("active", False):
            captured["recurrent_inputs"] = inputs
            captured["recurrent_output"] = output
            output.retain_grad()
        return output

    accumulated = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in model.named_parameters()}
    branches = [torch.zeros_like(attention.k_proj.weight, dtype=torch.float32) for _ in range(2)]
    records, total_loss, total_count = [], 0.0, 0
    micro = run["micro_batch_size"]
    world = run["world_size"]
    with (
        patch.object(prepare_module, "prepare_key", probe),
        patch.object(flashrwkv2, "pretrain_tmix_wkv_prepare_bf16", prepare),
        patch.object(flashrwkv2, "pretrain_tmix_wkv7_recurrent_bf16", recurrent),
        patch.object(readout_module, "readout", readout),
        patch.object(flashrwkv2, "infer_embedding_ln0_forward_varlen", layer_norm),
    ):
        for offset in range(0, run["global_batch_size"], micro * world):
            if args.microbatches is not None and len(records) >= args.microbatches:
                break
            for rank in range(world):
                if args.microbatches is not None and len(records) >= args.microbatches:
                    break
                start = consumed + offset + rank * micro
                samples = [dataset[index] for index in range(start, start + micro)]
                batch = {name: torch.from_numpy(np.stack([s[name] for s in samples])).cuda() for name in samples[0]}
                mask = batch["loss_mask"]
                model.zero_grad(set_to_none=True)
                losses = model(batch["features"], labels=batch["labels"])
                loss = (losses * mask).sum()
                # Keep gradients in the scale of a global mean loss, then normalize for valid tails below.
                denominator = run["global_batch_size"] * dataset.seq_length
                (loss / denominator).backward()
                total_loss += loss.item()
                total_count += int(mask.sum().item())
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        accumulated[name].add_(parameter.grad.float())
                inputs = captured["inputs"]
                outputs = captured["outputs"]
                gradients = tuple(tensor.grad.detach() for tensor in outputs)
                leaves = tuple(tensor.detach().requires_grad_() for tensor in inputs)
                reference = reference_key(leaves[0], leaves[1], leaves[2], leaves[3], head_size)
                expected = torch.autograd.grad(reference, leaves, gradients)
                actual = original(*leaves, head_size)
                actual_gradients = torch.autograd.grad(actual, leaves, gradients)
                key, scale, rate, _ = inputs
                norms = (key.float() * scale.float()).unflatten(-1, (-1, head_size)).norm(dim=-1)
                valid = mask.bool()
                record = {
                    "rank": rank,
                    "microbatch": offset // (micro * world),
                    "sample_start": start,
                    "valid_key_norm": norm_summary(norms[valid]),
                    "padding_key_norm": norm_summary(norms[~valid]),
                    "heads": [
                        {
                            "valid": norm_summary(norms[..., head][valid]),
                            "padding": norm_summary(norms[..., head][~valid]),
                        }
                        for head in range(norms.shape[-1])
                    ],
                    "forward": [comparison(a, b) for a, b in zip(actual, reference)],
                    "backward": [comparison(a, b) for a, b in zip(actual_gradients, expected)],
                }
                branch_keys = []
                for branch in range(2):
                    leaf = tuple(tensor.detach().requires_grad_() for tensor in inputs)
                    upstream = tuple(
                        g if (i == 0) == (branch == 0) else torch.zeros_like(g) for i, g in enumerate(gradients)
                    )
                    dk = torch.autograd.grad(
                        reference_key(leaf[0], leaf[1], leaf[2], leaf[3], head_size), leaf, upstream
                    )[0]
                    branch_keys.append(dk.float())
                    branches[branch].add_(dk.flatten(0, 1).float().T @ captured["xk"].flatten(0, 1).float())
                upstream_direction = -gradients[1].float() + gradients[2].float() * rate.float()
                amplification = branch_keys[1].unflatten(-1, (-1, head_size)).norm(dim=-1)
                location = torch.unravel_index(amplification.argmax(), amplification.shape)
                b, t, h = (int(value.item()) for value in location)
                record["largest_direction_gradient"] = {
                    "sample_index": start + b,
                    "time": t,
                    "head": h,
                    "valid": bool(valid[b, t].item()),
                    "key_norm": norms[b, t, h].item(),
                    "gradient_norm": amplification[b, t, h].item(),
                    "upstream_norm": upstream_direction.unflatten(-1, (-1, head_size))[b, t, h].norm().item(),
                }
                if offset == 0 and rank == 0:
                    torch.save(
                        {
                            "run": run,
                            "diagnostic_step": snapshot["diagnostic_step"],
                            "layer": args.layer,
                            "location": record["largest_direction_gradient"],
                            "key_inputs": tuple((x[b : b + 1] if x.ndim == 3 else x).detach().cpu() for x in inputs),
                            "key_upstream": tuple(g[b : b + 1].cpu() for g in gradients),
                            "key_projection_input": captured["xk"][b : b + 1].cpu(),
                            "key_projection_weight": attention.k_proj.weight.detach().cpu(),
                        },
                        args.output.with_suffix(".pt"),
                    )
                if args.check_recurrent and offset == 0 and rank == 0:
                    from flashrwkv2.tmix.wkv7.tilelang import tmix_wkv7_recurrent

                    leaf = tuple(
                        x[b : b + 1].detach().contiguous().requires_grad_() for x in captured["recurrent_inputs"]
                    )
                    dy = captured["recurrent_output"].grad[b : b + 1].contiguous()
                    expected_y, expected_state = reference_recurrent(*leaf, head_size=head_size)
                    expected_grad = torch.autograd.grad(expected_y, leaf, dy)
                    state = torch.zeros_like(expected_state)
                    cu = torch.tensor([0, leaf[0].shape[1]], device=state.device, dtype=torch.int32)
                    actual_y, final_state = tmix_wkv7_recurrent(
                        *(x.view(-1, x.shape[-1] // head_size, head_size) for x in leaf), state, cu
                    )
                    actual_y = actual_y.view_as(expected_y)
                    actual_grad = torch.autograd.grad(actual_y, leaf, dy)
                    record["recurrent"] = {
                        "output": comparison(actual_y, expected_y),
                        "final_state": comparison(final_state, expected_state),
                        "backward": [comparison(x, y) for x, y in zip(actual_grad, expected_grad)],
                        "state_absmax": expected_state.abs().max().item(),
                    }
                    artifact = torch.load(args.output.with_suffix(".pt"), weights_only=True)
                    artifact["recurrent_inputs"] = tuple(x.detach().cpu() for x in leaf)
                    artifact["recurrent_upstream"] = dy.cpu()
                    torch.save(artifact, args.output.with_suffix(".pt"))
                if args.check_readout:
                    readout_inputs = captured["readout_inputs"]
                    readout_upstream = captured["readout_output"].grad
                    readout_leaves = tuple(x.detach().requires_grad_() for x in readout_inputs)
                    expected_readout = reference_readout(*readout_leaves, head_size=head_size)
                    actual_readout = original_readout(*readout_leaves, head_size=head_size)
                    expected_grads = torch.autograd.grad(expected_readout, readout_leaves, readout_upstream)
                    actual_grads = torch.autograd.grad(actual_readout, readout_leaves, readout_upstream)
                    values = readout_inputs[0].float().unflatten(-1, (-1, head_size))
                    variance = values.var(-1, unbiased=False)
                    gain = actual_grads[0].float().unflatten(-1, (-1, head_size)).norm(dim=-1)
                    gain /= readout_upstream.float().unflatten(-1, (-1, head_size)).norm(dim=-1).clamp_min(1e-30)
                    record["readout"] = {
                        "forward": comparison(actual_readout, expected_readout),
                        "backward": {
                            name: comparison(a, b)
                            for name, a, b in zip(
                                ("x", "r", "k", "v", "r_k", "weight", "bias", "gate"), actual_grads, expected_grads
                            )
                        },
                        "head_variance": [norm_summary(variance[..., h][valid]) for h in range(variance.shape[-1])],
                        "head_vjp_gain": [norm_summary(gain[..., h][valid]) for h in range(variance.shape[-1])],
                    }
                    if offset == 0 and rank == 0:
                        artifact = torch.load(args.output.with_suffix(".pt"), weights_only=True)
                        artifact["readout_inputs"] = tuple(x.detach().cpu() for x in readout_inputs)
                        artifact["readout_upstream"] = readout_upstream.cpu()
                        torch.save(artifact, args.output.with_suffix(".pt"))
                if args.check_norms:
                    norm_records = {}
                    for name, (norm_input, norm_output, weight, bias, eps) in norm_captures.items():
                        variance = norm_input.float().var(-1, unbiased=False)
                        norm_gain = norm_input.grad.float().norm(dim=-1) / norm_output.grad.float().norm(
                            dim=-1
                        ).clamp_min(1e-30)
                        selected = torch.unravel_index(norm_gain.masked_fill(~valid, -1).argmax(), norm_gain.shape)
                        nb, nt = (int(value.item()) for value in selected)
                        leaf_x = norm_input[nb : nb + 1, nt : nt + 1].detach().contiguous().requires_grad_()
                        leaf_w, leaf_b = weight.detach().requires_grad_(), bias.detach().requires_grad_()
                        upstream_norm = norm_output.grad[nb : nb + 1, nt : nt + 1].contiguous()
                        expected_norm = F.layer_norm(
                            leaf_x.float(), (leaf_x.shape[-1],), leaf_w.float(), leaf_b.float(), eps
                        ).bfloat16()
                        actual_norm = original_norm(leaf_x, leaf_w, leaf_b, eps=eps)
                        expected_vjp = torch.autograd.grad(expected_norm, (leaf_x, leaf_w, leaf_b), upstream_norm)
                        actual_vjp = torch.autograd.grad(actual_norm, (leaf_x, leaf_w, leaf_b), upstream_norm)
                        norm_records[name] = {
                            "input_variance": norm_summary(variance[valid]),
                            "vjp_gain": norm_summary(norm_gain[valid]),
                            "largest_gain_location": {
                                "sample_index": start + nb,
                                "time": nt,
                                "variance": variance[nb, nt].item(),
                            },
                            "forward": comparison(actual_norm, expected_norm),
                            "backward": [comparison(a, b) for a, b in zip(actual_vjp, expected_vjp)],
                        }
                    record["layer_norms"] = norm_records
                    norm_captures.clear()
                records.append(record)
                logger.info(
                    "rank=%s microbatch=%s %s", rank, record["microbatch"], record["largest_direction_gradient"]
                )
                del losses, loss, outputs, inputs, reference, actual
                captured.clear()
    for handle in handles:
        handle.remove()
    factor = run["global_batch_size"] * dataset.seq_length / total_count
    key_name = f"hf.layers.{args.layer}.linear_attn.k_proj.weight"
    report = {
        "snapshot_step": snapshot["diagnostic_step"],
        "run": run,
        "torch": torch.__version__,
        "flashrwkv2": flashrwkv2.__version__,
        "consumed_samples": consumed,
        "reference_key": args.reference_key,
        "layer": args.layer,
        "valid_count": total_count,
        "loss": total_loss / total_count,
        "grad_norm": sum(g.square().sum().item() for g in accumulated.values()) ** 0.5 * factor,
        "key_grad_norm": accumulated[key_name].norm().item() * factor,
        "raw_branch_norm": branches[0].norm().item() * factor,
        "direction_branch_norm": branches[1].norm().item() * factor,
        "branch_cosine": F.cosine_similarity(branches[0].flatten(), branches[1].flatten(), dim=0).item(),
        "branch_sum_vs_key_grad": comparison(branches[0] + branches[1], accumulated[key_name]),
        "microbatches": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Saved %s; loss=%s grad_norm=%s", args.output, report["loss"], report["grad_norm"])


if __name__ == "__main__":
    main()
