# AGENTS.md — Megatron Bridge

> **Project:** PyTorch-native bridge between Hugging Face and Megatron-Core.
> Bidirectional checkpoint conversion, pretraining, SFT, and LoRA recipes
> with optimized NVIDIA GPU performance. Package: `megatron.bridge` (Python 3.12).

## Skills

The `skills/` directory contains structured guides for common tasks (adding
models, running experiments, debugging multi-node jobs, performance tuning,
etc.). **Always read the relevant `SKILL.md` before starting any task it
covers — skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Boundaries

**NEVER:**
- Modify files inside `3rdparty/Megatron-LM/` — changes go through the upstream repo
- Run the full test suite — run only the specific tests relevant to your change
- Add required (non-optional) dependencies — use optional extras; submit dependency changes as a separate PR
- Commit secrets, tokens, `.env` files, or environment-specific paths / account names (e.g. `/home/yuya/…`, usernames, cluster hostnames)
- Use bare `print()` — use `logging.getLogger(__name__)` or `print_rank_0()`

**ASK FIRST:**
- Before adding any new dependency to `pyproject.toml`
- Before modifying CI workflows (`.github/workflows/`)
- Before changing public API signatures in `models/conversion/`

**ALWAYS:**
- Run `uv run pre-commit run --all-files` before committing
- Add NVIDIA copyright headers to new Python files (except under `tests/`)
- Sign off commits: `git commit -s -m "message"`
- Use `uv run python -m pytest` and `uv run python -m torch.distributed.run`, not bare `pytest` / `torchrun`
- Use the current year (2026) in generated content — do not default to 2025 or any past year

## Toolchain

| Action | Command | Config |
|--------|---------|--------|
| Install deps | `uv sync` | `pyproject.toml`, `uv.lock` |
| Install dev tools | `uv sync --group dev` | |
| Lint + format | `uv run pre-commit run --all-files` | `ruff.toml`, `.pre-commit-config.yaml` |
| Unit tests | `uv run python -m pytest tests/unit_tests/` | `pyproject.toml [tool.pytest]` |
| Distributed run | `uv run python -m torch.distributed.run --nproc_per_node=2 script.py` | |
| Type check | `uv run mypy --strict path/to/file.py` | |
| Regen lock file | `uv lock` | |
| Init submodule | `git submodule update --init` | `.gitmodules` |

## Code Style

Lint and format are enforced by pre-commit hooks (ruff). See @ruff.toml for
the authoritative rules. For judgment calls not covered by tooling, see
@skills/linting-and-formatting/SKILL.md. Key points the linter cannot catch:

- Type hints required on all public API functions (`X | None`, not `Optional[X]`)
- Google-style docstrings on public classes and functions
- Use `*` separator for functions with multiple same-type parameters
- No arbitrary defaults for config values — be explicit

## Testing

- **No foreign `setattr` on config dataclasses in tests.** When a test applies overrides to a recipe config via `setattr(config_obj, key, value)`, always guard with `if not hasattr(config_obj, key): raise ValueError(...)` first. Setting an attribute that does not exist on the dataclass silently creates a phantom field — the test passes but the recipe would fail for a real user who never sets that key. This applies to all override patterns (`model_overrides`, `checkpoint_overrides`, `config_overrides`, etc.) in `tests/functional_tests/`.

## Contributing

See @CONTRIBUTING.md for the full contributor guide, including:

- Commit and PR title format ([Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<scope>): <description>`)
- PR labeling taxonomy (type, area, state, risk labels)
- Testing conventions (unit preferred, functional max 2 GPUs, L0/L1/L2 tiers)
- Dependency management policy
- DCO sign-off requirements
- CI triggering (`/ok to test <commit-SHA>`)

## Architecture

### Bridge Pattern

Each supported model family lives under `src/megatron/bridge/models/<family>/` with:

| File | Role |
|------|------|
| `bridge.py` | HF ↔ Megatron conversion logic |
| `config_mapping.py` | Maps HF config → Megatron config |
| `param_mapping.py` | Maps parameter names between formats |
| `hf_pretrained/` | HF model definition provider |

Recipes live in `src/megatron/bridge/recipes/<family>/`.

`AutoBridge` (`models/conversion/auto_bridge.py`) auto-selects the correct
bridge from a HF model name or path.

### Megatron-Core Submodule

`3rdparty/Megatron-LM/` is a git submodule pinned to a specific commit,
installed as an editable package via `[tool.uv.sources]` in `pyproject.toml`.
Use `scripts/switch_mcore.sh` to switch versions.

## Tool Compatibility

This file is the single source of agent instructions. For Claude Code
compatibility, create a symlink: `ln -s AGENTS.md CLAUDE.md`

## RWKV Project Integration Rules

### Core Objectives and Implementation Principles

Integrate RWKV7 through Megatron Bridge's native `AutoBridge`, ModelConfig, ModelBuilder,
training loop, distributed checkpoint, and PEFT interfaces, covering weight conversion,
pretraining, SFT, and LoRA. Report supported capabilities based on completed validation.

- **Correspondence Principle:** Before making changes, find an existing implementation with
  similar computation or lifecycle semantics. Follow its directory, type, function, and
  configuration naming; use `RWKV7` / `rwkv7` for model-specific names.
- **Non-Interference Principle:** RWKV changes must preserve other models' conversion,
  training, and checkpoint behavior. Keep model-specific logic in the RWKV7 family and
  integrate through existing hooks, builders, and local composition.
- Reuse existing Megatron-Core and Bridge components first. Implement genuinely missing
  RWKV computation within the family. When using transformers-rwkv, reuse its model
  computation and FlashRWKV2 backend rather than copying models or kernels.
- RWKV uses TokenShift and WKV recurrent state. For a fixed model and batch, decode state
  size does not grow with history length. Count actual recurrent operations when analyzing
  performance rather than applying Transformer KV cache sizing or quadratic attention FLOPs.
- Read `skills/adding-model-support/SKILL.md` before model integration and
  `skills/parity-testing/SKILL.md` before conversion or numerical validation. Follow the
  corresponding skills for training, performance, and external RL validation.
- New files must fall within the task's explicitly authorized scope; clarify first when
  that scope is unclear. Follow the existing family structure and avoid wrappers, standalone
  training frameworks, or cross-project schedulers for one-off tasks.

### Authoritative Implementations and Responsibility Boundaries

| Source | Purpose |
|--------|---------|
| [RWKV7 NumPy](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py) | TokenShift, TimeMix, ChannelMix, and state update formulas |
| [RWKV7 Qwen3.5 example](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py) | Computation reference for the corresponding model variant |
| [RWKV-LM train_temp](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) | Pretraining and backward reference |
| [Albatross](https://github.com/BlinkDL/Albatross) | Low-level inference and numerical execution reference |
| [DPLR mathematics](https://zhiyuan1i.github.io/posts/dplr-mathematics) | Parallel computation of Diagonal Plus Low Rank state transitions |
| [transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv/tree/rwkv) | HF model, config, tokenizer, and chat templates |
| `flashrwkv2` project | RWKV operators, forward/backward, state precision, and performance contracts |
| [vllm-rwkv](https://github.com/rwkv-rs/vllm-rwkv/tree/rwkv) | External rollout / inference interfaces and consumers of exported weights |

This project owns Bridge model integration, configuration and parameter mappings, training
recipes, PEFT, and checkpoint import/export. FlashRWKV2 owns kernel issues;
transformers-rwkv owns HF model and tokenizer issues; the corresponding inference and
evaluation projects own serving and evaluation publication. Make cross-project changes
in a separate session under the target project. First record reproducible inputs,
versions, and error evidence in this project.

### Weights and Model Identity

RWKV7 original weights generally follow this naming convention:

```text
{arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
rwkv7-g1h-7.2b-20260710-ctx10240.pth
```

- `arch_version` distinguishes `rwkv7`, `rwkv7a` (DeepEmbed), and `rwkv7b` (DeepEmbedAttn).
  Inspect the actual config and computation graph before integration; validate each variant separately.
- `data_version`, such as `g1a` or `g1b`, is part of the weight identity. Model configurations,
  experiment names, and results must retain the complete weight version, source, and revision,
  rather than using only a name such as `RWKV7-1.5B`.
- Common parameter sizes include `0.1b`, `0.4b`, `1.5b`, `2.9b`, `7.2b`, and `13.3b`.
  Read architecture parameters from the checkpoint's config rather than inferring them from its size.
- Original releases: [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1/tree/main);
  test weights: [BlinkDL/temp-latest-training-models](https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main);
  HF weights: [rwkv-rs/rwkv7-g1-st](https://huggingface.co/rwkv-rs/rwkv7-g1-st).
- Record `.pth` → HF preparation separately from HF ↔ Megatron conversion. Preserve the
  correspondence between config, tokenizer, vocabulary, special tokens, and chat templates.

### Directory Conventions

| Path | Responsibility |
|------|----------------|
| `src/megatron/bridge/models/rwkv7/` | Bridge, builder-backed config, builder, and model-specific computation |
| `src/megatron/bridge/recipes/rwkv7/` | RWKV7 recipes reusing native training configuration |
| `src/megatron/bridge/peft/rwkv7.py` | RWKV7 PEFT targets and adaptation logic |
| `examples/models/rwkv7/` | Runnable conversion, training, and verification examples |
| `tests/unit_tests/models/test_rwkv7.py` | Unit tests for RWKV7 config, mappings, and model interfaces |

Follow the repository's existing categories for new functional tests, documentation, and
examples. Configure parallelism according to the builder's actual implementation. Validate
each TP, PP, CP, SP, packing, or CUDA graph execution path separately.

### Numerical and Correctness Checks

1. Use `wkv_mode=fp32io16` by default: BF16 model computation and inputs/outputs, with FP32
   WKV state and accumulation. Keep logits, selected-token logprobs, and loss in FP32.
   Preserve FlashRWKV2 rounding boundaries. Configure precision changes explicitly and
   validate them separately.
2. Check TokenShift's previous token, TimeMix/ChannelMix state, the first-layer value residual,
   normalization epsilon, head size, and low-rank dimensions. Reset state between independent
   samples. For consecutive chunks, carry state according to the training semantics and
   explicitly define whether gradients propagate across chunks.
3. Check label shifting, loss masks, padding, and valid token counts. Use labels already
   shifted by Megatron data processing directly for per-token loss. Packed/varlen paths
   must isolate each sequence's state.
4. Verify complete parameter coverage, names, shapes, dtypes, and values through
   HF → Megatron → HF conversion. Require exact round-trip equality for pure layout changes.
   Record arithmetic or dtype transformations and their errors, and verify persisted
   checkpoints after reloading them in a fresh process.
5. Compare HF and Bridge logits, next tokens, and loss using identical weights, token IDs,
   state, and precision. Follow the parity-testing skill's acceptance criteria. When adding
   or replacing recurrent/chunk kernels, compare both outputs and final state and cover
   forward/backward execution.
6. Run a bounded training job with a small model and check finite loss, valid gradients,
   and parameter updates. On resume, verify optimizer, scheduler, step, RNG, and data progress.
   For LoRA, verify frozen base weights, adapter updates, save/resume, and successful HF /
   inference loading after merging and export.
7. Use the NoCoT, FakeCoT, and CoT templates supplied with the weights. Record the actual
   template, EOS/stop behavior, generation budget, and sampling parameters. Preserve
   templates, tokenizer, and special tokens through conversion.
8. For external RL integration, follow `skills/nemo-rl-e2e-testing/SKILL.md` or
   `skills/verl-e2e-testing/SKILL.md` to verify tokens, logprobs, loss masks, weight
   synchronization, and checkpoint/resume. Rollout and trainer must consume the same
   policy weight version.

### Performance, Experiment Records, and Environment

- Pass correctness checks before measuring steady-state performance. Fix the weight version,
  data, valid token count, batch, sequence length, precision, and parallelism configuration.
  Record compilation/warmup overhead separately from steady-state step time, tokens/s, and
  peak GPU memory. Count FLOPs using actual RWKV operations.
- Record training loss, learning rate, grad norm, consumed tokens, checkpoint/resume, and
  export validation results. Use identical data and token budgets for quality comparisons.
  Declare checkpoint and evaluation intervals explicitly in the recipe or experiment
  configuration. Validate performance tuning separately from changes to training semantics.
- Use this project's uv-managed `./.venv` locally and remotely, following the `uv run`
  conventions above. Synchronize source across machines with Git and avoid duplicating
  build, test, or benchmark jobs already managed by CI.
- Record code and dependency revisions, GPU models, execution commands, and checks actually
  passed. Use portable paths in public documentation and commits; keep runtime paths,
  credentials, and service deployment details in external runtime configuration.
