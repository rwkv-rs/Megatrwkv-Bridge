# RWKV-7 pretraining, LoRA and NoRA

RWKV-7 reuses transformers-rwkv model computation and FlashRWKV2 operators
through Megatron Bridge's AutoBridge, ModelConfig, ModelBuilder, training loop,
PEFT and distributed checkpoint interfaces.

## Runtime

Use this project's uv-managed environment and pinned Megatron-LM submodule.
The RWKV runtime requires [transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv),
[FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2), and the corresponding
[tokenizers-rwkv](https://github.com/rwkv-rs/tokenizers-rwkv) tokenizer implementation.
The tested runtime uses Transformers 5.21.0.dev0, FlashRWKV2 0.1.0a18,
PEFT 0.20.0, PyTorch 2.14.0+cu130 and Transformer Engine 2.16.0.
The HF config must specify `architecture_version="rwkv7"`.

Computation and model weights use BF16; WKV state/accumulation, logits and loss
use FP32 (`wkv_mode="fp32io16"`). Execution uses TP=PP=CP=1, with ordinary DDP
or Megatron FSDP `optim_grads`. Inputs are equal-length, unpacked token
sequences. Each independent sample starts with fresh recurrent state.

## Data and pretraining

The native GPT dataset and dataloader consume pretokenized Megatron indexed
files. For `corpus_text_document.bin` and `corpus_text_document.idx`, pass
`--data corpus_text_document`. Prepare token IDs with the tokenizer and special
tokens corresponding to the model config. The training NullTokenizer consumes
those IDs directly. GPTDataset shifts labels; RWKV computes per-token cross
entropy against the already shifted labels.

Run a small smoke test with the supplied architecture and generated tokens:

```bash
timeout 600s uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 \
  examples/models/rwkv7/train.py --profile generic \
  --config examples/models/rwkv7/tiny-config.json \
  --mock --output runs/rwkv7-smoke --steps 20
```

For full-parameter pretraining from random initialization, provide the actual HF
architecture without `--base`:

```bash
timeout 24h uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/rwkv7/train.py --config MODEL_CONFIG.json \
  --data DATA_PREFIX --seq-length 4096 --output runs/rwkv7-pretrain \
  --steps 10000 --save-interval 1000 --eval-interval 1000 --eval-iters 10
```

Add `--base HF_CHECKPOINT_DIRECTORY --full-parameter` for full-parameter
continued pretraining. With `--base` and no `--full-parameter`, training installs
LoRA instead. `--lora-rank` defaults to 32 and alpha defaults to rank; add
`--nora` for NoRA. The adapters target TimeMix `r_proj/k_proj/v_proj/o_proj` and
ChannelMix `key/value`, with frozen base weights.

The example uses the base recipe's optimizer settings, zero LR warmup and a
cosine schedule ending at `--steps`. Configure learning rate, warmup, global
batch and token budget in the recipe for the intended training experiment.
Global batch must be divisible by DP size times microbatch size.

## PRO 6000 eight-GPU defaults

The training entry defaults to `--profile pro6000x8` and 4096 tokens. It calls
`rwkv7_8gpu_pro6000_config` to select the layout from actual architecture fields,
context length and full/PEFT mode. The supported g1l architectures are:

| Checkpoint identity | Layers | Hidden size |
|---|---:|---:|
| `rwkv-g1l-1b-temp-6331` | 24 | 2048 |
| `rwkv-g1l-3b-temp-5764` | 32 | 2560 |
| `rwkv-g1l-7b-temp-3797` | 32 | 4096 |
| `rwkv-g1l-13b-temp-5497` | 61 | 4096 |

All four use vocabulary size 65536, head size 64 and intermediate size four
times hidden size. Retain the complete checkpoint identity when selecting base
weights and recording experiments.

Each cell gives **per-GPU microbatch / recomputation** for eight RTX PRO 6000
Blackwell 96 GiB GPUs. Global batch defaults to eight times microbatch.
`infctx C=N` carries differentiable recurrent state across N-token chunks and
checkpoints each chunk. Cross-chunk gradients remain enabled. All layouts use
head chunk 1024.

| Model | Full 4096 | Full 16384 | Full 65536 |
|---|---|---|---|
| 1.5B | 2 / none | 1 / TimeMix | 1 / infctx C=4096 |
| 2.9B | 1 / none | 1 / layers | 1 / infctx C=2048 |
| 7.2B | 1 / layers | 1 / layers | 1 / infctx C=1024 |
| 13.3B | 1 / layers | 1 / layers | 1 / infctx C=1024 + layers |

Full 13.3B selects native Megatron FSDP `optim_grads`, FP32 gradient
accumulation/communication, overlapped gradient reduction and `fsdp_dtensor`
checkpoints. Smaller full models use DDP with the distributed optimizer.

| Model | PEFT 4096 | PEFT 16384 | PEFT 65536 |
|---|---|---|---|
| 1.5B | 4 / none | 1 / none | 4 / infctx C=4096 |
| 2.9B | 2 / none | 2 / TimeMix | 2 / infctx C=4096 |
| 7.2B | 1 / none | 1 / TimeMix | 3 / infctx C=2048 |
| 13.3B | 2 / TimeMix | 3 / layers | 2 / infctx C=1024 |

LoRA and NoRA ranks 8, 16 and 32 share the PEFT execution layouts and use DDP.
Explicit CLI values override recipe fields; booleans accept `--no-...` forms.
Changing microbatch retains the recipe's global batch unless
`--global-batch-size` is also supplied. Use `--profile generic` for tiny configs,
other architectures or other hardware; it retains the HF context length.

`--checkpoint-tmix` recomputes TimeMix; `--checkpoint-layers` recomputes complete
layers. `--head-chunk-size` bounds the vocabulary projection, while
`--infctx-chunk-size` and `--checkpoint-infctx` control state-passing chunks.
`--detach-infctx-state` explicitly selects truncated BPTT.

## Checkpoint, resume and HF export

`--save-interval`, `--eval-interval` and `--eval-iters` independently control
saving and evaluation. Saving defaults to the last update; evaluation defaults
to zero iterations. Native checkpoints contain model, optimizer, scheduler,
RNG and consumed-sample progress.

Resume with the same architecture, execution settings and adapter configuration,
adding `--resume CHECKPOINT_DIRECTORY`. `--steps` is the total target update
count, including updates completed before resuming.

Add `--export-hf HF_EXPORT_DIRECTORY` to export complete HF weights after
training and checkpoint saving. LoRA/NoRA adapters are merged for export.
Tokenizer, chat templates and generation configs are copied from the directory
containing `--config` when it contains `tokenizer_config.json`. Keep the config
beside the matching tokenizer assets when exporting a complete HF checkpoint.

For direct conversion, use the shared [conversion examples](../../conversion/)
or the builder-backed API:

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("HF_CHECKPOINT_DIRECTORY")
config = bridge.get_model_config()
config.perform_initialization = False
models = bridge.get_model(config, wrap_with_ddp=False, mixed_precision_wrapper=None)
bridge.save_hf_pretrained(models, "HF_EXPORT_DIRECTORY")
```

Training logs, TensorBoard, timers and profiling use the native Bridge
facilities. Write experiment outputs to a runtime directory outside the source
examples.

## Tests

```bash
timeout 300s uv run --no-sync python -m pytest tests/unit_tests/models/test_rwkv7.py -q
timeout 600s uv run --no-sync python -m pytest tests/functional_tests/test_rwkv7.py -q
```

Unit tests cover config serialization, parameter mappings, label alignment,
PEFT lifecycle, the full/PEFT recipe matrix and CLI overrides. CUDA tests cover
HF forward/backward parity, LoRA/NoRA updates, frozen base weights and chunked
state passing with chunk/layer recomputation.
