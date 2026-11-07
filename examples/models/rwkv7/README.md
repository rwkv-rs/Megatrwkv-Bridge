# RWKV-7 pretraining, LoRA and NoRA

RWKV-7 uses the existing `transformers-rwkv` model computation and FlashRWKV2
operators inside a Megatron `MegatronModule`. Bridge owns distributed construction,
optimizer setup, the training loop, and checkpoint management. The implementation
supports BF16 with data parallelism and complete model replicas (TP=PP=CP=1).
Inputs are equal-length, unpacked sequences; recurrent state starts fresh on each
forward. Use `ModelConfig` and `ModelBuilder` for construction.

| Capability | Execution contract |
|---|---|
| HF import/export | `AutoBridge`, builder-backed model config, replicated parameter mappings |
| Token pretraining | Native Bridge loop, indexed or mock GPT token data, shifted labels |
| LoRA / NoRA | Native PEFT lifecycle, fused FlashRWKV2 adapters, merge and HF export |
| DDP | Complete BF16 model replicas; optional native distributed optimizer |
| Megatron FSDP | `optim_grads`, FP32 gradient accumulation/communication, `fsdp_dtensor` checkpoints |
| Long context | Differentiable TokenShift/WKV state passing, chunk and layer recomputation |
| Model parallelism | TP=PP=CP=1; sequence parallelism is disabled |
| Packing / CUDA graphs | Unpacked inputs; these execution paths have not been validated |
| Continuous-valued time series | Task-specific dataset, projection and prediction head are deferred |

## Runtime

Use this project's uv-managed `.venv` with the pinned Megatron-LM submodule initialized.
The runtime must provide the RWKV-7 implementation from
[rwkv-rs/transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv).
The current validation runtime uses Transformers `5.21.0.dev0` from that fork with
`FlashRWKV2==0.1.0a18`, TileLang `0.1.15`, PEFT `0.20.0`, PyTorch
`2.14.0+cu130`, Transformer Engine `2.16.0`, and Megatron-Core
`ba495158226981dec4c5d79d8b7f02c3b981fdc7`. Tokenizer import/export also requires
the fork's `RwkvTrie` implementation from
[tokenizers-rwkv](https://github.com/rwkv-rs/tokenizers-rwkv), tested at
`c5d8dde5ff49c70e4656199d5033a84e03c21b2b` (`0.23.2.dev0`). The implementation requires
`RwkvConfig.architecture_version == "rwkv7"` and the fork's
`transformers.integrations.flash_rwkv2` API. PEFT is already a Bridge dependency.

Dependency manifests are unchanged. Prepare this CUDA runtime before
launching the examples. The commands below run from the Bridge repository root.

## Builder construction and HF export

The builder-backed AutoBridge API loads an existing HF checkpoint into the same
model class used by training. This example exports its weights and tokenizer
assets back to HF:

```bash
timeout 1800s uv run --no-sync python - <<'PY'
from megatron.bridge.models import AutoBridge

bridge = AutoBridge.from_hf_pretrained("HF_CHECKPOINT_DIRECTORY", trust_remote_code=False)
config = bridge.get_model_config()
config.perform_initialization = False
models = bridge.get_model(config, wrap_with_ddp=False, mixed_precision_wrapper=None)
bridge.save_hf_pretrained(models, "HF_EXPORT_DIRECTORY")
PY
```

For training, `--base HF_CHECKPOINT_DIRECTORY` performs this import through the
native finetuning setup. Training checkpoints include optimizer, scheduler, RNG
and data progress and use the native distributed checkpoint format.

## Pretraining

The supplied tiny configuration is for a smoke test. Set your actual architecture
in a local HF `config.json`; the recipe reads architecture dimensions from it.

```bash
timeout 600s uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 \
  examples/models/rwkv7/train.py \
  --config examples/models/rwkv7/tiny-config.json \
  --profile generic --mock --output runs/rwkv7-smoke --steps 20
```

Replace `--mock` with `--data DATA_PREFIX` for indexed token data. The default
NullTokenizer treats the indexed values as token IDs with the configured vocabulary;
tokenization must already have been applied. Increase the number of processes for DP,
keeping global batch size divisible by the process count and microbatch size.

The data config sets `create_attention_mask=False`. RWKV enforces causality through
its recurrent computation. Labels are shifted by the Megatron dataset; the wrapper
computes unreduced cross entropy directly against those labels.

Add `--full-parameter --base HF_CHECKPOINT_DIRECTORY` for full-parameter continued
pretraining. Without `--full-parameter`, `--base` selects PEFT. Full-parameter
training can use `--distributed-optimizer` to shard the optimizer over DP ranks.
For gradient and optimizer sharding, use `--megatron-fsdp` instead; it selects
Megatron FSDP `optim_grads` and the `fsdp_dtensor` checkpoint format. Retain that
flag when restoring its checkpoint. This path explicitly uses FP32 gradient
accumulation and communication, and enables native bucketed gradient reduction
to bound temporary gradient memory.

`--save-interval`, `--eval-interval`, and `--eval-iters` independently control
checkpoint and validation frequency. Saving defaults to the final update and
validation defaults to zero iterations. A bounded continued-pretraining example is:

```bash
timeout 3600s uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/rwkv7/train.py --config HF_CHECKPOINT_DIRECTORY/config.json \
  --base HF_CHECKPOINT_DIRECTORY --full-parameter \
  --data DATA_PREFIX --output runs/rwkv7-full --steps 100 \
  --seq-length 4096 --save-interval 20 --eval-interval 20 --eval-iters 2
```

Resume with the same architecture and execution settings, adding
`--resume runs/rwkv7-full/checkpoints`. `--steps` is the total target update count.
Add `--export-hf runs/rwkv7-full-hf` to export the final complete model after
checkpoint saving. The same flag merges LoRA or NoRA before export. It also works
on the native FSDP training/resume path. Tokenizer and template artifacts are
copied from the HF directory containing `--config` when it includes
`tokenizer_config.json`; a standalone architecture config exports model config
and weights.

To export an existing training checkpoint, retain its architecture, precision,
sharding, and adapter settings, set `--steps` to the checkpoint's saved step,
and supply `--resume CHECKPOINT_DIRECTORY --export-hf HF_EXPORT_DIRECTORY`.
This invokes the final export without performing another optimizer update.

The 13.3B / 64K full-parameter configuration uses gradient and optimizer
sharding together with chunk and layer recomputation:

```bash
timeout 14400s uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/rwkv7/train.py --config HF_13B_CHECKPOINT_DIRECTORY/config.json \
  --base HF_13B_CHECKPOINT_DIRECTORY --full-parameter \
  --mock --output runs/rwkv7-13b-full --steps 20 \
  --seq-length 65536 \
  --save-interval 20 --eval-interval 20 --eval-iters 1 \
  --export-hf runs/rwkv7-13b-full-hf
```

For a resume comparison at a fixed token budget, retain the same final `--steps`
in both processes. Use `--save-interval 4` in the first process and stop it after
checkpoint 4 is saved. Resume with `--save-interval 20` and
`--resume runs/rwkv7-13b-full/checkpoints` in the second process.

## Eight-GPU execution defaults

`train.py` defaults to `--profile pro6000x8` and 4096 tokens. It calls
`rwkv7_8gpu_pro6000_config` to select the table below from the actual layer count,
hidden size, context and full/PEFT mode. Full 13.3B selects FSDP automatically;
smaller full models select the distributed optimizer. PEFT uses ordinary DDP.
Both training and the benchmark use this single recipe. Explicit CLI values
replace recipe values; boolean overrides also accept `--no-...` forms.
A microbatch override retains the recipe's global batch unless
`--global-batch-size` is also supplied. Use `--profile generic` for tiny configs,
other hardware or manual layouts; it retains the architecture's context length.
The PRO 6000 recipe supports the four architectures and three contexts below.


The following configurations target 8 × RTX PRO 6000 Blackwell 96 GiB at 400 W,
with DP=8 and TP=PP=CP=1. Model dimensions come from the actual checkpoint config.
The current weight identities are `rwkv-g1l-1b-temp-6331`,
`rwkv-g1l-3b-temp-5764`, `rwkv-g1l-7b-temp-3797`, and
`rwkv-g1l-13b-temp-5497`, respectively. These are token training workloads.
The [weight identity manifest](weight-identities.json) records SHA-256 content
fingerprints for weight shards, config, tokenizer, special tokens, and templates.

Each cell gives **per-GPU microbatch / recomputation strategy**. `infctx C=N`
means `--infctx-chunk-size N --checkpoint-infctx`, preserving cross-chunk
gradients. `TimeMix` means `--checkpoint-tmix`; `layers` means
`--checkpoint-layers`. All cases use `--head-chunk-size 1024`, BF16 computation,
FP32 WKV state, and FP32 logits and loss.

| Model | PEFT 4096 | PEFT 16384 | PEFT 65536 |
|---|---|---|---|
| 1.5B | 4 / none | 1 / none | 4 / infctx C=4096 |
| 2.9B | 2 / none | 2 / TimeMix | 2 / infctx C=4096 |
| 7.2B | 1 / none | 1 / TimeMix | 3 / infctx C=2048 |
| 13.3B | 2 / TimeMix | 3 / layers | 2 / infctx C=1024 |

LoRA and NoRA share these settings for rank 8, 16, and 32. Alpha defaults to rank.
NoRA rank 32 is the representative PEFT memory check.

| Model | Full 4096 | Full 16384 | Full 65536 | Sharding |
|---|---|---|---|---|
| 1.5B | 2 / none | 1 / TimeMix | 1 / infctx C=4096 | DDP distributed optimizer |
| 2.9B | 1 / none | 1 / layers | 1 / infctx C=2048 | DDP distributed optimizer |
| 7.2B | 1 / layers | 1 / layers | 1 / infctx C=1024 | DDP distributed optimizer |
| 13.3B | 1 / layers | 1 / layers | 1 / infctx C=1024 + layers | FSDP optim_grads |

The recipe and entry defaults are checked against all 27 accepted CSV cases.
The current targeted unit suite passes 61 tests. An additional eight-GPU run
of `rwkv-g1l-1b-temp-6331` completed three NoRA-32 updates with no layout flags:
the entry selected 4096 tokens, micro/global batch 4/32 and head chunk 1024,
with finite losses and gradient norms and zero skipped updates.

These a18 execution defaults passed the 27-case acceptance matrix below;
historical a17 measurements retain their original configurations and timing windows.
Full-parameter DDP
with a distributed optimizer budgets approximately `7.5 × parameter_count` bytes
per GPU for BF16 weights, FP32 gradients, and eight-way sharded FP32 master/Adam
state. The 13.3B configuration therefore shards gradients as well.

For bounded throughput validation, use global batch `8 × microbatch`, 12 updates,
and discard the first four updates. Compare alternatives at the same global batch,
seed, data, optimizer, and token budget. Target at least 8 GiB of device headroom,
including checkpoint saves. State boundaries consume memory proportional to chunk
count when retaining cross-chunk gradients, so reducing chunk size also increases
boundary-state storage.

`benchmark.py` defaults to one NoRA rank-32 / 4K recipe configuration
per explicitly listed model. Select `--modes full` for full-parameter training. Multi-value arguments remain available
for explicitly requested batches. Each run records resolved execution settings,
dependency versions, source fingerprint, power limits, per-rank memory, update
health, setup and warmup time, and slowest-rank steady-state step timing.

## LoRA continued pretraining

Use `--base` to load a full HF or native Megatron checkpoint and train adapters:

```bash
timeout 600s uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 \
  examples/models/rwkv7/train.py \
  --config examples/models/rwkv7/tiny-config.json \
  --mock --base runs/rwkv7-smoke/checkpoints \
  --output runs/rwkv7-lora --steps 20 --lora-rank 8 --lora-alpha 8
```

`RWKV7LoRA` targets `r_proj`, `k_proj`, `v_proj`, `o_proj`, `key`, and `value`.
It uses HF PEFT `LoraLayer` objects because the fused operators consume the base
weight, LoRA A/B weights, and scale directly. LoRA dropout is zero. Base parameters
are frozen; Bridge's PEFT checkpoint filter selects trainable adapter parameters.
Configure `cfg.peft.target_modules` to select a subset or add `lm_head`.

Add `--nora` to use NoRA's column-normalized A matrix, with normalization and
its gradient computed in FP32 by FlashRWKV2. Set both `--lora-rank` and
`--lora-alpha` to 8, 16 or 32 for the requested adapter configurations, or omit
`--lora-alpha` to use the rank. The same
save, resume and merge interfaces apply to LoRA and NoRA.

To resume an adapter run, provide both `--base BASE_CHECKPOINT` and
`--resume ADAPTER_CHECKPOINT_DIRECTORY`, retaining the same architecture and LoRA
settings. For full pretraining resume, use `--resume` without `--base`.

For full-model HF export after training, save the adapter checkpoint first, then
add `--export-hf HF_EXPORT_DIRECTORY` to the training or resume command. The final
callback calls `merge_lora()` on the unwrapped `RWKV7Model` and uses the standard
AutoBridge HF export API. Merging modifies the live model and restores canonical
parameter names after the adapter checkpoint is saved.

## Time-series integration

The executable example trains discrete tokens. A continuous-valued task should
provide its own `DatasetProvider`, input projection, prediction head and forward
step, using the public `pretrain(config, forward_step_func)` entry point.
`RWKV7Model.forward(inputs_embeds=...)` accepts `[batch, time, hidden]` BF16 embeddings
and returns vocabulary logits. For a continuous prediction head, use
`model.hf.model(inputs_embeds=..., use_cache=False).last_hidden_state` as the backbone
output. Keep the input projection and prediction head inside the model so their
parameters participate in optimizer setup and checkpointing. Decide explicitly
whether these task modules remain trainable when applying PEFT.

See [training entry points](../../../docs/training/entry-points.md),
[custom dataset contracts](../../../src/megatron/bridge/data/base.py), and
[RWKV recipes](../../../src/megatron/bridge/recipes/rwkv7/rwkv7.py).

## Verification

```bash
timeout 300s uv run --no-sync python -m pytest tests/unit_tests/models/test_rwkv7.py -q
timeout 600s uv run --no-sync python -m pytest tests/functional_tests/test_rwkv7.py -q
```

Unit coverage includes builder-config serialization, architecture overrides,
unsupported model-parallel settings, complete HF parameter mapping, label alignment,
fused LoRA argument extraction and gradient flow, adapter enable/disable, parameter
save selection, adapter state reload, and merging.
The single-GPU test compares HF and Bridge logits and runs an optimizer step
through fused LoRA, checking finite gradients and unchanged base parameters.

### Current a18 entry-point and recomputation checks

On 2026-11-06, 61 unit tests passed, including the 27 measured configurations,
benchmark-to-training integration, explicit overrides, and all five native entry modes with
and without final HF export: random-initialized full training, HF-backed full
DDP, HF-backed full FSDP, LoRA, and NoRA. The tests check FP32 FSDP gradient
configuration, independent save/evaluation intervals, and alpha defaulting to rank.

Eight CUDA state-passing cases passed with batch sizes one and two, chunk
checkpointing, and layer checkpointing, comparing loss, final recurrent state,
and gradients with the unchunked model. An earlier a18 invocation passed 37
tests (nine unit and 28 CUDA functional cases), including the 24 unchanged
adapter numerical cases. These cover LoRA/NoRA ranks 8, 16, and 32, nonzero
adapters, selective TimeMix and layer checkpointing, frozen base parameters,
and adapter updates.

### Native checkpoint and HF lifecycle checks

Two-rank tiny-model checks passed for full DDP with the distributed optimizer,
full FSDP `optim_grads`, LoRA, and NoRA, each accumulating two microbatches per
optimizer update. Every mode saved at update 4 and resumed in a fresh process to
update 8. Before data-loader construction, the loaded model, optimizer,
scheduler, RNG, step, and consumed-sample fingerprints matched the saved state
exactly on both ranks. Model updates and frozen PEFT base parameters were
checked, and fresh HF processes reproduced each final export's logits exactly.

The DDP/FSDP gradient-accumulation comparison reported the same first-update
loss 5.668353 and gradient norm 648.975 at native log precision. After two
updates, 36 of 69 model tensors were exact, the largest tensor relative L2
error was 0.0022494, and final-logit cosine similarity was 0.99999875 with the
same next-token decision.

Zero-update export from the native FSDP checkpoint at step 9 also passed:
all 69 exported tensors matched the preceding step-9 export exactly.

### Real-checkpoint lifecycle acceptance

Five eight-rank native lifecycle cases passed on 2026-11-06. The first process
used the final update budget from the start and stopped after checkpoint 4;
the second process resumed to update 8 or 20 with the same scheduler budget.
All eight ranks matched model, optimizer, scheduler, RNG, step, and consumed-sample
fingerprints exactly before data-loader construction. Every update had finite
loss and gradient norm with zero skipped updates. PEFT checks verified unchanged
base parameters and updated adapters before merging.

| Checkpoint identity | Mode | Context | Micro/global batch | Save → target update | Validation loss | Fresh HF logits |
|---|---|---:|---:|---:|---:|---|
| `rwkv-g1l-1b-temp-6331` | full DDP | 4,096 | 1/8 | 4 → 8 | 8.800041 | exact |
| `rwkv-g1l-1b-temp-6331` | LoRA 8 | 4,096 | 1/8 | 4 → 8 | 0.933483 | cos 0.99988008 |
| `rwkv-g1l-1b-temp-6331` | NoRA 32 | 4,096 | 1/8 | 4 → 8 | 4.951509 | cos 0.99998391 |
| `rwkv-g1l-13b-temp-5497` | full FSDP | 65,536 | 1/8 | 4 → 20 | 5.102569 | exact |
| `rwkv-g1l-13b-temp-5497` | NoRA 32 | 65,536 | 2/16 | 4 → 20 | 2.102131 | cos 0.99999273 |

The table compares live native logits before adapter merging with a fresh HF
process after export. Adapter merging performs arithmetic into BF16 base
weights; its measured cosine and absolute errors are retained in the
[validation record](validation-a18.json). The public record retains acceptance
summaries and fingerprints; raw per-update and per-rank records remain in local
runtime artifacts. Both full-parameter exports reproduced
logits exactly. Every case retained the next-token decision for both supplied
`open_think` and `fake_think` templates and repeated bounded greedy generation
with `max_new_tokens=16`, `do_sample=False`, EOS 0, and the supplied stop strings.

Tokenizer vocabulary, special tokens, and chat templates matched the source.
All three generation configuration files (`generation_config.json`,
`fake_think_generation_config.json`, and `tools_generation_config.json`) were
preserved byte-for-byte using the family's standard artifact hook. The validation
record includes file SHA-256 identities for all five trained HF exports,
validation/test loss, per-step learning rates and gradient norms, and per-rank
checkpoint/export memory and checkpoint time. Full 13.3B training consumed
10,485,760 tokens; its NoRA run consumed 20,971,520 tokens.

The same record identifies the final uncommitted RWKV source files, registration
and tests by SHA-256, together with dependency revisions and the base commit.
Each performance CSV row retains the source fingerprint actually used by that
measurement.

### Earlier integration checks

Checks completed on 2026-11-04 and 2026-11-05 with RTX PRO 6000 Blackwell GPUs:

- Nine unit tests and 28 CUDA functional tests passed. CUDA coverage includes
  LoRA/NoRA with `alpha=rank=8/16/32` and the independently scaled rank-8/alpha-16
  case, decoder and selective TimeMix checkpointing, differentiable state
  passing with batch sizes one and two, final recurrent state, gradients, frozen
  base weights, and adapter updates on tiny models. Adapter loss and gradients
  are compared with HF using nonzero adapter weights.
- All four checkpoints below completed HF → builder-backed Megatron → HF export.
  All 4,929 persisted tensors matched exactly after reloading in a fresh process.
  HF/Bridge logits matched exactly on identical 64-token inputs for each model,
  including the next-token decision. Tokenizer and template assets were preserved.
- An eight-rank native full-parameter training smoke test completed four optimizer
  steps with finite loss and no skipped updates.
- The same real 1.5B checkpoint completed eight-rank LoRA training at 4K, 16K,
  and 64K. The 64K run used differentiable state passing with FP32 WKV state.
- All four real checkpoints completed 64K forward validation with nonzero
  rank-8 LoRA and NoRA adapters and batch size two. HF reference chunks of 8,192
  tokens and Bridge chunks of 4,096/4,096/2,048/1,024 tokens respectively produced
  identical per-token losses and final recurrent states. WKV states remained FP32.
- Native LoRA and NoRA save and resume executed on two ranks. Before the first
  resumed update, all 1,152 adapter/optimizer tensors matched the saved checkpoint
  exactly; scheduler state, step 2, and 16 consumed samples also matched.
  Python, NumPy, Torch, CUDA, and Megatron tracker RNG matched before data-loader
  construction. Both modes completed resumed steps 3 and 4. LoRA's losses differed
  from the uninterrupted loss reference by 0.055% and 0.018%, respectively.
- Real 1.5B LoRA and NoRA checkpoints were merged and exported. A fresh process
  loaded each export with Transformers and reproduced the merged model's logits
  exactly, then ran bounded greedy generation with the supplied template,
  EOS, and stop strings.
- Native full-parameter training with the distributed optimizer completed save
  and resume on two ranks. Before a resumed update, all 69 model tensors and
  three global FP32 optimizer buffers matched exactly after re-saving and
  reloading in a fresh process; RNG, scheduler, step and consumed samples also
  matched. Adapter training loaded a native full-model checkpoint as its base
  and completed four updates.

All four sizes completed eight-rank 4K LoRA training. The 13.3B checkpoint also
completed eight-rank 64K NoRA training with `alpha=rank=32` on 2026-11-06.
The current a18 performance acceptance uses the execution defaults above, NoRA
rank 32 at all twelve model/context points, and three representative LoRA rank-8 cases.
The actual checkpoint architectures and exact round-trip tensor counts are:

| Checkpoint identity | Parameters | Layers | Hidden size | Tensors |
|---|---:|---:|---:|---:|
| `rwkv-g1l-1b-temp-6331` | 1,527,404,544 | 24 | 2,048 | 795 |
| `rwkv-g1l-3b-temp-5764` | 2,947,735,040 | 32 | 2,560 | 1,059 |
| `rwkv-g1l-7b-temp-3797` | 7,199,141,888 | 32 | 4,096 | 1,059 |
| `rwkv-g1l-13b-temp-5497` | 13,269,245,952 | 61 | 4,096 | 2,016 |

The following single-GPU real-checkpoint forward/backward checks also passed on
2026-11-05, using FlashRWKV2 kernel inspection, nonzero scratch
adapters, and three gradient samples before and after inspection. All six
checks retained the loss within `1e-5`, gradient relative L2 below `0.00582`,
and gradient cosine similarity above `0.999982`. State-passing cases kept
cross-chunk gradients enabled. The vocabulary head used 1,024-row recomputation.

| Checkpoint identity | Context | Strategy | Microbatch | Adapter, alpha=rank | Peak allocated GiB |
|---|---:|---|---:|---|---:|
| `rwkv-g1l-1b-temp-6331` | 16,384 | none | 1 | NoRA, 32 | 66.05 |
| `rwkv-g1l-1b-temp-6331` | 65,536 | infctx, chunk 4,096 | 4 | LoRA, 8 | 66.64 |
| `rwkv-g1l-3b-temp-5764` | 16,384 | tmix | 2 | LoRA, 8 | 76.75 |
| `rwkv-g1l-7b-temp-3797` | 16,384 | tmix | 1 | LoRA, 8 | 70.43 |
| `rwkv-g1l-13b-temp-5497` | 16,384 | layers | 3 | LoRA, 8 | 68.51 |
| `rwkv-g1l-13b-temp-5497` | 65,536 | infctx, chunk 2,048 | 1 | LoRA, 8 | 67.60 |

## Eight-GPU performance experiments

`benchmark.py` launches the native Bridge loop with DP=8 and records complete
optimizer-step time, loss, grad norm, skipped updates, and allocated/reserved
memory for every rank. It excludes the specified warmup steps and computes
throughput using the slowest rank in each iteration. Results include dependency
versions, a RWKV source fingerprint, the complete checkpoint name, and CPU
affinity records for every rank.

Keep a local, uncommitted model manifest containing actual checkpoint identities:

```json
[
  {
    "name": "COMPLETE_WEIGHT_VERSION",
    "checkpoint": "CHECKPOINT_DIRECTORY",
    "config": "CHECKPOINT_DIRECTORY/config.json"
  }
]
```

Select one model and one execution configuration from the defaults above. For
example, with a manifest containing the 1.5B checkpoint:

```bash
timeout 3600s uv run --no-sync python examples/models/rwkv7/benchmark.py \
  --models models.json --output runs/rwkv7-default \
  --modes nora --ranks 32 --seq-lengths 4096 \
  --steps 12 --warmup 4 --resume
```

Use `--modes full` for full-parameter training. `--strategies default` uses the
recipe; explicitly selecting `none`, `tmix`, `layers` or `infctx` uses the generic
profile and the supplied manual layout. These strategies expose execution choices for
explicitly selected comparisons. Each result records resolved micro/global
batch sizes. `--global-batch-size` fixes the global batch for matched comparisons;
omitting it uses `8 * microbatch`. Compile and warmup work precedes the eight
steady-state updates. The a18 runtime uses the library's hardware strategy.

For a manifest selecting the 13.3B checkpoint, its 64K full-parameter default
combines state passing with layer checkpointing:

```bash
timeout 7200s uv run --no-sync python examples/models/rwkv7/benchmark.py \
  --models models-13b.json --output runs/rwkv7-full-64k \
  --modes full --seq-lengths 65536 \
  --steps 12 --warmup 4 --timeout 7200 --resume
```

`--head-chunk-size` limits vocabulary-projection rows and recomputes the head
during backward, preserving ordinary cross entropy. It does not enable the
FlashRWKV2 L2Wrap objective. `--infctx-chunk-size` carries differentiable
TokenShift and FP32 WKV states between contiguous chunks of each independent
sequence. `--checkpoint-infctx` recomputes those chunks. Cross-chunk gradients
remain enabled; `--detach-infctx-state` explicitly selects truncated BPTT and
changes training semantics. State is reset between independent samples.

### Current FlashRWKV2 a18 acceptance

On 2026-11-06, all 27 selected configurations completed 12 optimizer updates:
12 full-parameter cases, 12 NoRA rank-32 cases, and three LoRA rank-8 cases.
Alpha equals rank. Each run excludes four warmup updates and measures the next
eight updates, using the slowest rank per update. All ranks recorded finite
losses and gradient norms and zero skipped updates. Inputs use mock GPT token
data with seed 1234, equal-length unpacked sequences, and all tokens contributing
to cross entropy.

The native Adam recipe uses betas `(0.9, 0.95)`, epsilon `1e-8`, gradient clipping
at 1.0, and cosine learning-rate decay from `3e-4` to `3e-5` over each run's
target update budget. Learning-rate warmup is zero; the four excluded timing
updates still advance training and the scheduler. The constant weight-decay
schedule applies 0.033 to decay-enabled parameter groups and zero to native
no-decay groups.

The hardware is eight RTX PRO 6000 Blackwell Workstation GPUs at 400 W, with
PCIe links across two NUMA domains and no NVLink. CUDA is 13.0, NCCL is 2.30.7,
and the driver is 620.70.06. The precision and recomputation settings are listed
in the execution-default tables above; cross-chunk gradients remain enabled.
PyTorch uses `expandable_segments:True` for its CUDA allocator.

The memory column is the maximum **device-used sample at optimizer-step end**
across all twelve updates and all eight ranks. The largest sample was 84.53 GiB,
leaving at least 11.06 GiB against the driver-reported 97,887 MiB capacity.
The [a18 CSV](performance-a18.csv) also records allocator allocated/reserved
peaks, per-rank memory, timing variation, process wall time, setup and warmup cost,
learning rates, losses, gradient norms, valid token counts, and the exact source fingerprint
used by each run.

| Model | Mode | Context | Micro/global batch | Tokens/s, all 8 GPUs | Mean seconds/update | Peak device GiB/GPU |
|---|---|---:|---:|---:|---:|---:|
| 1.5B | full | 4,096 | 2/16 | 41,518 | 1.578 | 45.88 |
| 1.5B | full | 16,384 | 1/8 | 40,800 | 3.213 | 35.52 |
| 1.5B | full | 65,536 | 1/8 | 42,328 | 12.386 | 33.08 |
| 2.9B | full | 4,096 | 1/8 | 19,283 | 1.699 | 50.76 |
| 2.9B | full | 16,384 | 1/8 | 25,119 | 5.218 | 30.18 |
| 2.9B | full | 65,536 | 1/8 | 26,182 | 20.025 | 43.60 |
| 7.2B | full | 4,096 | 1/8 | 8,197 | 3.997 | 56.46 |
| 7.2B | full | 16,384 | 1/8 | 12,273 | 10.680 | 64.23 |
| 7.2B | full | 65,536 | 1/8 | 12,298 | 42.633 | 80.42 |
| 13.3B | full | 4,096 | 1/8 | 5,308 | 6.173 | 57.55 |
| 13.3B | full | 16,384 | 1/8 | 7,114 | 18.425 | 68.68 |
| 13.3B | full | 65,536 | 1/8 | 5,405 | 96.992 | 83.13 |
| 1.5B | NoRA 32 | 4,096 | 4/32 | 66,970 | 1.957 | 69.58 |
| 1.5B | NoRA 32 | 16,384 | 1/8 | 54,101 | 2.423 | 69.54 |
| 1.5B | NoRA 32 | 65,536 | 4/32 | 54,122 | 38.749 | 70.87 |
| 2.9B | NoRA 32 | 4,096 | 2/16 | 37,136 | 1.765 | 62.18 |
| 2.9B | NoRA 32 | 16,384 | 2/16 | 33,152 | 7.907 | 80.45 |
| 2.9B | NoRA 32 | 65,536 | 2/16 | 29,713 | 35.290 | 63.51 |
| 7.2B | NoRA 32 | 4,096 | 1/8 | 18,936 | 1.730 | 60.66 |
| 7.2B | NoRA 32 | 16,384 | 1/8 | 16,916 | 7.749 | 74.46 |
| 7.2B | NoRA 32 | 65,536 | 3/24 | 16,350 | 96.197 | 84.53 |
| 13.3B | NoRA 32 | 4,096 | 2/16 | 10,480 | 6.254 | 81.88 |
| 13.3B | NoRA 32 | 16,384 | 3/24 | 8,635 | 45.538 | 75.58 |
| 13.3B | NoRA 32 | 65,536 | 2/16 | 8,721 | 120.236 | 79.83 |
| 1.5B | LoRA 8 | 4,096 | 4/32 | 66,448 | 1.973 | 69.08 |
| 2.9B | LoRA 8 | 16,384 | 2/16 | 33,081 | 7.924 | 79.70 |
| 13.3B | LoRA 8 | 65,536 | 2/16 | 8,837 | 118.658 | 78.08 |

### Historical FlashRWKV2 a17 measurements

#### Measured screening results

These runs use `rwkv-g1l-1b-temp-6331`, eight complete BF16 replicas, mock GPT
token samples, ordinary cross entropy, LoRA `alpha=rank=8`, and a 1,024-row
recomputed vocabulary head. Peak memory is the maximum allocated memory across
all eight ranks, including warmup. Timed steps exclude warmup and use each
iteration's slowest rank. Every completed run below had finite losses and no
skipped updates.

| Context | Strategy | Microbatch | Global batch | Timed steps | Tokens/s, all 8 GPUs | Peak GiB/GPU |
|---|---|---:|---:|---:|---:|---:|
| 4,096 | none | 1 | 64 | 5 | 47,117 | 19.32 |
| 4,096 | none | 4 | 64 | 5 | 61,907 | 66.01 |
| 4,096 | none, prepared kernels | 4 | 64 | 5 | 70,003 | 66.01 |
| 4,096 | layers | 8 | 64 | 5 | 51,366 | 12.95 |
| 16,384 | none | 1 | 16 | 2 | 45,985 | 65.98 |
| 16,384 | layers | 2 | 16 | 2 | 41,127 | 12.94 |
| 65,536 | infctx, chunk 4,096 | 1 | 32 | 2 | 34,253 | 19.51 |

Preparing the exact 1.5B/4K/microbatch-4/LoRA-rank-8 forward and backward
workload resolved all 31 requested FlashRWKV2 tuning keys in 2,404.7 seconds.
The scratch model's loss matched before/after preparation; the largest
relative gradient L2 difference was 0.00441 and the smallest gradient cosine
similarity was 0.999990. Requiring the same profiles in a fresh process found
31 hits and no missing keys. The matched eight-rank run improved throughput by
13.1%; its largest relative per-step loss difference was 0.0222%.

The larger models also completed the following 4K screens with the same
precision, loss and head configuration, using one microbatch per rank per
optimizer step:

| Checkpoint identity | Strategy | Microbatch | Global batch | Timed steps | Tokens/s, all 8 GPUs | Peak GiB/GPU |
|---|---|---:|---:|---:|---:|---:|
| `rwkv-g1l-3b-temp-5764` | none | 2 | 16 | 2 | 33,992 | 58.47 |
| `rwkv-g1l-3b-temp-5764` | none | 3 | 24 | 4 | 34,482 | 84.40 |
| `rwkv-g1l-7b-temp-3797` | none | 1 | 8 | 2 | 17,048 | 56.47 |
| `rwkv-g1l-13b-temp-5497` | layers | 2 | 16 | 2 | 7,998 | 32.63 |
| `rwkv-g1l-13b-temp-5497` | tmix | 2 | 16 | 2 | 8,851 | 77.40 |

For the 13.3B checkpoint at the matched micro/global batch of 2/16, selective
TimeMix recompute improved throughput by 10.7% over decoder recompute. The
largest relative per-step mock loss difference across those four updates was
0.58%. The smaller checkpoints' matched comparisons were within 0.03%.

These measurements used eight RTX PRO 6000 Blackwell Workstation GPUs with a
400 W power limit per GPU.

#### 4K adapter-rank screening matrix

On 2026-11-06, all four checkpoints completed eight-rank screens for LoRA and
NoRA with `alpha=rank=8/16/32`. Each run executed three optimizer updates,
excluded the first update from timing, and recorded finite losses and gradient
norms on all ranks with zero skipped updates. These two-step timing windows
screen candidates for longer validation. The precision, ordinary cross entropy,
1,024-row recomputed vocabulary head, and 400 W power limits match the setup above.

| Checkpoint identity | Adapter | Strategy | Micro/global batch | Rank 8 tokens/s | Rank 16 tokens/s | Rank 32 tokens/s | Peak allocated GiB/GPU range |
|---|---|---|---:|---:|---:|---:|---:|
| `rwkv-g1l-1b-temp-6331` | LoRA | none | 4/32 | 69,789 | 69,421 | 68,888 | 66.01–66.47 |
| `rwkv-g1l-1b-temp-6331` | NoRA | none | 4/32 | 69,833 | 69,315 | 69,221 | 66.02–66.50 |
| `rwkv-g1l-3b-temp-5764` | LoRA | none | 2/16 | 38,780 | 38,455 | 38,386 | 58.47–59.13 |
| `rwkv-g1l-3b-temp-5764` | NoRA | none | 2/16 | 38,757 | 38,464 | 38,302 | 58.49–59.18 |
| `rwkv-g1l-7b-temp-3797` | LoRA | none | 1/8 | 20,076 | 19,927 | 19,873 | 56.47–57.46 |
| `rwkv-g1l-7b-temp-3797` | NoRA | none | 1/8 | 20,080 | 19,942 | 19,862 | 56.50–57.54 |
| `rwkv-g1l-13b-temp-5497` | LoRA | tmix | 2/16 | 11,208 | 11,189 | 11,133 | 77.40–79.26 |
| `rwkv-g1l-13b-temp-5497` | NoRA | tmix | 2/16 | 11,196 | 11,163 | 11,133 | 77.43–79.34 |

Throughput is the total across all eight GPUs, using the slowest rank for each
optimizer step. The memory range covers the three adapter ranks in each row.
[The CSV](performance-4k.csv) preserves individual measurements, batch sizes,
timing budgets, and the Python source fingerprint.

#### Long-context validation window

On 2026-11-06, `rwkv-g1l-13b-temp-5497` completed 12 optimizer updates on all
eight GPUs with NoRA `alpha=rank=32`, BF16 computation, FP32 WKV state,
ordinary cross entropy, and a 1,024-row recomputed vocabulary head. It used
differentiable state passing with 1,024-token chunks and chunk checkpointing,
microbatch 2, and global batch 16. Cross-chunk gradients remained enabled.

After three warmup updates, the nine timed updates covered 9,437,184 tokens.
The mean slowest-rank step time was 146.18 seconds and the median was 146.30
seconds. All losses were finite, with no skipped updates.

| Context | Adapter, alpha=rank | Microbatch | Global batch | Timed steps | Tokens/s, all 8 GPUs | Peak allocated GiB/GPU |
|---|---|---:|---:|---:|---:|---:|
| 65,536 | NoRA, 32 | 2 | 16 | 9 | 7,173 | 76.14 |

The table above retains the historical a17 measurement and its original timing window.

For full-parameter pretraining, `train.py --distributed-optimizer` enables
Megatron's optimizer-state sharding. PEFT leaves the base weights replicated and
communicates adapter gradients. Large checkpoint-backed PEFT models skip random
initialization and allocate BF16 weights on the GPU before base-weight loading.

FlashRWKV2 a18 owns kernel selection and caching. Normal training warmup uses
its default cache; the training and benchmark entry points reuse that lifecycle.
