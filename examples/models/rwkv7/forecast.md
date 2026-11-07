# Autoregressive minute OHLCV forecasting

This task predicts the next complete OHLCV bar from historical OHLCV, then feeds
the generated bar back to predict subsequent bars. It uses transformers-rwkv's
RWKV7 computation and FlashRWKV2 through Bridge's model builder, training loop,
DDP, optimizer, scheduler and distributed checkpoints. It trains from random
initialization, with task-specific continuous input and output projections.
The recipe is **early, unoptimized**; throughput and forecast quality have not
been measured.

## Local data

Download [neigezhu/china-a-share-1min-ohlcv](https://huggingface.co/datasets/neigezhu/china-a-share-1min-ohlcv)
separately. Pass the local snapshot root with this layout:

```text
SNAPSHOT/
  data/stock_1m/SH/600519.parquet
  data/stock_1m/SZ/000001.parquet
  metadata/...
```

A subset of canonical instrument files is sufficient to start. Record the
actual downloaded Hub commit SHA with `--revision`. The default Hub
`bars_sample` subset only contains the first up to 64 bars of each instrument;
use the canonical per-instrument files for training.

The five predicted fields are `open`, `high`, `low`, `close`, `volume`.
`turnover` is excluded from the inputs and target. Prices remain unadjusted
CNY/share; volume is in shares. Naive timestamps retain the source's UTC+8
interpretation. Preparation rejects mixed-instrument files and unordered or
duplicate timestamps. Invalid OHLCV rows, missing minutes, lunch and overnight
gaps terminate a sequence. Zero volume is valid. No missing bars are filled.

Prepare a **new** destination, with exclusive chronological split boundaries:

```bash
timeout 24h uv run --no-sync python examples/models/rwkv7/forecast.py prepare \
  --source SNAPSHOT --output DATA/ashare-ohlcv \
  --train-end 2024-01-01 --validation-end 2025-01-01 \
  --revision DATASET_COMMIT_SHA --workers 8
```

Training uses dates before 2024-01-01; validation uses 2024; test uses dates
starting 2025-01-01. These are example experiment boundaries. Choose them for
the downloaded snapshot before comparing models. Windows and their complete
targets remain inside one split, instrument and continuous session. Independent
windows reset all recurrent state. Context length is measured in minute bars;
64 is a small initial setting that fits within a continuous session.

Preparation processes one instrument per worker and writes float32 encoded
features, float64 closing prices and timestamps as memory-mapped NumPy shards.
These uncompressed arrays need roughly 36 bytes per retained bar, plus the
segment manifest (about 125 GB for 3.48 billion bars). Use a smaller instrument
selection for initial experiments. The destination must not already exist;
remove a failed preparation's incomplete destination before retrying. Preparation
reads local files only and does not download the dataset.

## Representation and objective

For a bar with previous close `p`, the five features are:

```text
100 * log(open / p)
100 * log(close / p)
100 * log(high / max(open, close))
100 * log(min(open, close) / low)
log1p(volume) / 10
```

The first raw bar in each segment supplies the anchor and is not a feature row.
Fixed feature units require no fitted normalization statistics. The model's
last three channels use softplus, so decoding gives positive prices, ordered
OHLC and nonnegative volume. Outputs are continuous point estimates, including
volume. The decoder fails on nonfinite or overflowing paths rather than
silently clipping them.

Training uses teacher forcing with a single next-bar shift and Huber loss
averaged over the five encoded channels at each position. Bar counts control
gradient normalization across microbatches and DP ranks. Model computation is
BF16, WKV state and accumulation are FP32 (`wkv_mode=fp32io16`), and prediction
features and loss are FP32. Inference primes a fresh recurrent cache with the
context, then processes one generated bar per step. State size stays fixed with
the generation horizon.

## Train and resume

Use the same RWKV runtime as the [token training example](README.md#runtime).
The local environment needs NumPy and the existing `parquet` extra for PyArrow.
No new dependency extra is introduced.

Start with the supplied tiny architecture and a bounded single-GPU run:

```bash
timeout 1h uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=1 \
  examples/models/rwkv7/forecast.py train \
  --config examples/models/rwkv7/tiny-config.json \
  --data DATA/ashare-ohlcv --output runs/ashare-ohlcv \
  --context 64 --steps 200 --warmup-steps 10 \
  --global-batch-size 32 --micro-batch-size 4 \
  --learning-rate 3e-4 --save-interval 100 --eval-interval 100 --eval-iters 10
```

Supply another RWKV7 architecture JSON to change model capacity. The forecasting
entry sets vocabulary size to one during construction and removes the language
embedding/head. The task owns its separate continuous projections. Current
execution uses TP=PP=CP=1 and no SP or packing. Increase `--nproc_per_node` for
DDP; global batch size must divide evenly by GPU count times microbatch size.

The training sampler uses a seeded affine permutation of available windows,
repeated as needed to cover the explicit optimizer-step budget. It stores no
array proportional to the number of windows. The original chronological split
membership remains fixed. Native checkpoints under `OUTPUT/checkpoints` save
model, optimizer, scheduler, RNG and consumed-sample counters.

For a complete pass over the training bars, replace `--steps` with `--epochs 1`.
Epoch mode uses a stride equal to the context length, so each predictable bar
is a target once per epoch. It retains short and final windows using a per-bar
loss mask. Each segment needs its initial anchor and first input observation;
remaining bars are targets. The last global batch is completed with masked
dummy samples rather than repeated real samples. Gradient normalization uses
the global count of valid target bars across ranks and accumulated microbatches.
Validation and test dates retain their original split membership. `--steps`
continues to select the overlapping-window experiment described above.

For an interrupted run, repeat the training command with the same output and
settings and add `--resume runs/ashare-ohlcv/checkpoints`. `--steps` is the
original total training horizon. The recorded run configuration and prepared
manifest fingerprint must match. Keep prepared shards immutable. A fresh run
requires a new output directory. Use native `--exit-interval 100` with a larger
`--steps` to exercise bounded checkpoint/resume while retaining the LR schedule;
omit the exit option when resuming.

`run.json` records the architecture, dataset manifest hash, schedule, batch
layout and seed. On normal training completion, `forecast.pt` exports task
weights, configuration and completed step for inference. It contains the input
projection, RWKV backbone and regression head. Resume training from the native
checkpoint directory.

## Generate and evaluate

Generate the next 20 bars from the end of one canonical instrument file:

```bash
timeout 10m uv run --no-sync python examples/models/rwkv7/forecast.py predict \
  --checkpoint runs/ashare-ohlcv/forecast.pt \
  --instrument SNAPSHOT/data/stock_1m/SZ/000001.parquet \
  --horizon 20 --output runs/ashare-ohlcv/prediction.csv
```

The default context uses up to the training context length, plus one raw anchor
bar. A partial session uses its available shorter context, requiring at least
two raw bars; history never crosses a session gap. The CSV records instrument
identity, historical context end, actual context length, one-based forecast
step, and OHLCV. Steps are ordered future bars; the output assigns no future
calendar timestamps. Training windows cover continuous intraday minutes, so use
matching intraday horizons when assessing this model.

Evaluate free-running forecasts against a baseline that repeats the last
observed complete OHLCV bar:

```bash
timeout 1h uv run --no-sync python examples/models/rwkv7/forecast.py evaluate \
  --checkpoint runs/ashare-ohlcv/forecast.pt --data DATA/ashare-ohlcv \
  --split test --horizon 20 --max-windows 1000 \
  --output runs/ashare-ohlcv/test-metrics.json
```

The evaluator spreads up to `max-windows` windows across the full split inventory
with a horizon-sized stride within each segment. It feeds back predictions
throughout the horizon, compares in original price/volume units, and reports
per-field MAE/RMSE and persistence MAE/RMSE at every forecast step. Validation
during native training reports teacher-forced loss; use the free-running
evaluator to assess accumulated rollout error. The prepared manifest must match
the training run. Source data covers A-share equities; ETF data is published
separately by the dataset author.

## Stability diagnostics

Add `--diagnostics-interval 10` to native training to record global, layer and
module statistics under `forecast/` in TensorBoard and in `diagnostics.jsonl`.
The callback observes native AdamW without changing gradients or optimizer steps.
It records parameter and gradient L2 norms, RMS and absolute maxima; raw gradients
are reconstructed from the native clipping coefficient. Both the pre-clipping
gradient norm and the post-clipping norm are retained. FP32 master-weight
snapshots provide actual parameter changes (`delta_*`) and relative updates.
`update_*` describes the AdamW direction recovered from those changes after
removing the applied learning rate and decoupled weight decay. This recovery
includes FP32 master-weight rounding, rather than BF16 model-copy rounding.
DP optimizer shards are counted once, and absolute maxima use MAX reductions.

Forward hooks record hidden RMS and absmax at each block, TimeMix and ChannelMix
output, including padded tail positions. These are reduced across DP ranks.
RWKV does not produce softmax attention logits or attention entropy. The native
loss logger additionally reports all five OHLCV channel losses with the same
valid-position mask and denominator as the overall loss.

The callback logs periodic samples plus events whose raw gradient norm exceeds
five times the preceding 100-step median or whose loss exceeds 1.2 times that
median. These event rules trigger recording, not clipping or skipped updates.
The parameter RMS mean-field reference follows the dynamic approximation in
[AdamW Weight RMS, part 2](https://kexue.fm/archives/11404):
`q_t = (1 - lr_t * wd_t)^2 * q_(t-1) + lr_t^2`, initialized from actual parameter
mean squares. It is an approximate reference under the article's assumptions,
not a target for every RWKV parameter. Decay-free parameters have no finite
constant-LR stationary RMS.

For a separately validated initialization experiment, add
`--calibration-windows 4096 --head-gain 0.01 --head-fp32`. Calibration uses only
training windows, initializes the input projection and its constant direction
jointly and orthogonally in FP32 before BF16 rounding, and gives the output head
a small orthogonal gain. The nonzero constant direction keeps unchanged,
zero-volume bars away from a zero-variance embedding. The output biases
encode the training target means, including inverse softplus for positive
channels. `--head-fp32` computes the final projection in FP32 while retaining
BF16 parameter storage and the existing RWKV backbone precision. These choices
are persisted in the HF configuration in `run.json` and exported checkpoints.

Training validation now repeats a fixed, seeded selection of holdout windows
at every evaluation. This separates model changes from changes in evaluation
stocks and times. Evaluation still covers a subset; free-running metrics use
the separate evaluator above.

## Validation commands

```bash
timeout 300s uv run --no-sync python -m pytest \
  tests/unit_tests/data/test_ohlcv.py \
  tests/unit_tests/models/test_rwkv7_forecast.py \
  tests/unit_tests/models/test_rwkv7_forecast_cli.py -q
timeout 600s uv run --no-sync python -m pytest tests/functional_tests/test_rwkv7_forecast.py -q
```

The CPU tests cover feature inversion, zero-volume bars, chronological isolation,
one-time label shifting, deterministic sample traversal, regression gradients,
state reset and CLI validation. The single-GPU test covers a real backward and
optimizer step, cached autoregression against full-context replay, repeatable
state reset and exported-weight loading for prediction in a fresh process.
Run the bounded native training/resume commands above to validate the complete
distributed training lifecycle in the target runtime.
