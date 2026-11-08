# Autoregressive minute OHLCV forecasting

This task predicts the next complete OHLCV bar from historical OHLCV, then feeds
the generated bar back to predict subsequent bars. It uses transformers-rwkv's
RWKV7 computation and FlashRWKV2 through Bridge's model builder, training loop,
DDP, optimizer, scheduler and distributed checkpoints. It trains from random
initialization, with task-specific continuous input and output projections.
These are experimental forecasting tasks. Each run records its data identity,
configuration, measured throughput and held-out forecast scores.

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
The sensitive TimeMix `k_a`, `k_k` and `w0` parameters also have individual groups,
so their parameter and gradient scales are not hidden by module-wide aggregates.
It records parameter and gradient L2 norms, RMS and absolute maxima; raw gradients
are reconstructed from the native clipping coefficient. Both the pre-clipping
gradient norm and the post-clipping norm are retained. FP32 master-weight
snapshots provide actual parameter changes (`delta_*`) and relative updates.
`update_*` describes the bias-corrected AdamW direction computed directly from
the optimizer's FP32 first and second moments and epsilon, excluding learning
rate and decoupled weight decay. Actual `delta_*` includes FP32 master-weight
rounding, rather than BF16 model-copy rounding.
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

`--spike-snapshot` additionally saves `spike-before.pt` at the first detected
event. It contains the exact BF16 model weights used before that optimizer step,
the training configuration and the consumed-sample cursor. This allows a
single-batch numerical investigation without relying on a nearby periodic
checkpoint. It requires diagnostics and preserves an existing snapshot on resume.

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

`--normalize-inputs` additionally centers and standardizes the five encoded input
channels using the same training-only calibration windows. It leaves targets,
loss units and OHLCV decoding unchanged. The model applies these fixed statistics
to both observed context and generated bars during rollout. Statistics are stored
in the HF configuration, and the derived nonpersistent buffers explicitly use
BF16 storage so native training and standalone reload use identical rounded
statistics. Constant channels use a scale of one. This option requires a positive
`--calibration-windows` count and needs a new training run.

`--rwkv-optimizer` selects RWKV-LM's parameter groups: only matrix parameters
whose names end in `.weight` receive weight decay; TimeMix vectors, low-rank
factors and biases do not. The `w0` decay logits use twice the base learning rate.
The recipe sets both native scheduler bounds (`max_lr`, `min_lr`) for `w0`;
`lr_mult` metadata alone does not change the current scheduler's applied rate.
`optimizer-groups.json` records actual rates and bounds after checkpoint loading.
This follows [`train_temp`'s `configure_optimizers` and callback](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp/src), rather than treating a
`[1, 1, hidden]` TimeMix vector as a decay-eligible matrix. The policy is persisted
as `forecast_optimizer` and must be evaluated in a separate run because it
changes optimizer groups and training semantics.

New runs additionally record `optimizer_policy=rwkv_lm_v2` in `run.json`.
Resuming an older run preserves that run's original policy. To apply the
corrected scheduler bounds to an older checkpoint, create a separate continuation
with `--fork-from SOURCE_RUN --checkpoint-step STEP`. The loader constructs the
source parameter groups, restores the FP32 master weights and Adam moments, then
applies the requested policy before the first training update. Subsequent native
checkpoints preserve the corrected groups for normal `--resume`.

`--learning-rate-scale 0.5` multiplies the native scheduler's output uniformly,
preserving the RWKV-LM 1x/2x ratios, schedule position, and checkpoint parameter
group identities. Keep `--learning-rate` at the source run's value when forking.
All data, model, batch, seed, and original training-horizon settings must match;
the continuation permits changes to the scale and optimizer policy. `fork.json`
records the source configuration and checkpoint iteration.

For example, continue an existing one-epoch experiment from iteration 2500 and
stop at iteration 2750 while retaining its full learning-rate schedule:

```bash
timeout 1h uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/rwkv7/forecast.py train \
  --config examples/models/rwkv7/forecast-l24-d256.json \
  --data DATA/ashare-ohlcv --output runs/ashare-continuation \
  --fork-from runs/ashare-source --checkpoint-step 2500 \
  --epochs 1 --context 64 --global-batch-size 12288 --micro-batch-size 512 \
  --learning-rate 1e-4 --learning-rate-scale 0.5 --warmup-steps 100 \
  --save-interval 500 --eval-interval 500 --eval-iters 10 --exit-interval 2750 \
  --calibration-windows 4096 --head-gain 0.01 --head-fp32 --normalize-inputs \
  --rwkv-optimizer --diagnostics-interval 10 --spike-snapshot
```

The callback also writes `steps.jsonl` with loss and pre-clipping gradient norm
at every step. It reloads the preceding history on resume so the 100-step spike
window continues across checkpoint boundaries.

### Exact spike replay

Replay a captured global batch on one GPU, retaining the original rank and
microbatch order and accumulating parameter gradients in FP32:

```bash
timeout 10m uv run --no-sync python examples/models/rwkv7/forecast_replay.py \
  --snapshot runs/ashare-continuation/spike-before.pt \
  --data DATA/ashare-ohlcv --check-recurrent \
  --output runs/ashare-continuation/spike-replay.json
```

The report includes per-head key-norm distributions for valid and padded
positions, raw-key and normalized-direction gradient contributions, their
alignment, and fused-versus-reference forward/backward differences. The optional
recurrent comparison checks output, final FP32 state, and gradients of all six
inputs on a captured sample. A companion `.pt` stores that sample's operator
inputs and upstream gradients. `--reference-key` replaces only the selected
layer's key preparation; `--batch-step` selects another batch at fixed weights.

The snapshot also retains the pre-update FP32 master weights and key-projection
Adam moments and records the actual key updates. `spike-previous.pt` retains the
adjacent step with its original data cursor. Compare both weight points on the
same batch before attributing a large parameter gradient to that parameter's
own update. For example, replace the preceding layers while keeping the spike
batch and the remaining spike weights:

```bash
timeout 10m uv run --no-sync python examples/models/rwkv7/forecast_replay.py \
  --snapshot runs/ashare-continuation/spike-before.pt \
  --weights-from runs/ashare-continuation/spike-previous.pt \
  --replace-parameter 'hf.layers.[0-5].*' \
  --data DATA/ashare-ohlcv --microbatches 1 \
  --output runs/ashare-continuation/previous-upstream.json
```

Omit `--replace-parameter` to substitute all weights, or repeat it for multiple
groups. The report records the exact replaced parameter names. Use
`master_state_dict` for interpolation along an optimizer update, casting each
intermediate weight point to the original model dtype. BF16 endpoint
interpolation synchronizes one-ULP changes at the midpoint and distorts the
measured update path.

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

## Direct multi-horizon close quantiles

`--task direct_quantile --forecast-horizon 30` retains the historical OHLCV input
and the original RWKV7/FlashRWKV2 computation. It projects the final historical
hidden state to 30 future positions and nine quantiles (0.1 through 0.9).
Independent windows reset recurrent state. Each label is
`100 * log(future_close / last_context_close)`. Prices are recovered separately
at each horizon as `last_context_close * exp(prediction / 100)`.

The FP32 loss is standard pinball loss, averaged across quantiles and normalized
across valid future positions using native Bridge DP/microbatch counts. Only
complete context-plus-horizon windows enter this task. Use a fixed step budget;
the next-bar epoch/tail semantics apply to the original task. Calibration uses
training windows only, initializing each horizon/quantile bias from its empirical
quantile and estimating historical input statistics separately.

```bash
timeout 8h uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/models/rwkv7/forecast.py train \
  --task direct_quantile --forecast-horizon 30 \
  --config examples/models/rwkv7/forecast-l24-d256.json \
  --data DATA/ashare-ohlcv --output runs/ashare-direct-quantile \
  --context 64 --steps 3592 --warmup-steps 100 \
  --global-batch-size 12288 --micro-batch-size 768 --learning-rate 1e-4 \
  --save-interval 500 --eval-interval 500 --eval-iters 1 --seed 1234 \
  --calibration-windows 4096 --head-gain 0.01 --head-fp32 --normalize-inputs \
  --rwkv-optimizer --diagnostics-interval 10
```

`best-forecast.pt` stores the earliest checkpoint with the lowest fixed-window
validation pinball loss; `best-validation.json` records its score and step.
`forecast.pt` stores the last training weights. Native checkpoints retain the
optimizer, scheduler, RNG and consumed-sample state for resume. New task settings
and calibration statistics are persisted with the weights.

Direct inference returns `[batch, horizon, quantile]` scaled cumulative returns.
`predict_quantiles(..., fix_quantile_crossing=False)` returns raw outputs;
the default sorts the quantile axis at each horizon. The `predict` CLI writes
raw returns, sorted returns and price quantiles to CSV. Quantile ordering does
not constrain the direction of prices over time.

The direct evaluator writes `paths.npz`, `predictions.csv`, `windows.json`,
metrics JSON, `fixed.png`, `error_percentiles.png` and `quality.png`. A frozen
window file verifies instrument identity, context end and every future timestamp.
Use the same window file and earlier prediction arrays to compare with the
existing autoregressive baseline:

```bash
timeout 1h uv run --no-sync python examples/models/rwkv7/forecast.py evaluate \
  --checkpoint runs/ashare-direct-quantile/best-forecast.pt \
  --data DATA/ashare-ohlcv --split test --horizon 30 \
  --windows PREVIOUS_EVALUATION/windows.json \
  --baseline-paths PREVIOUS_EVALUATION/paths.npz \
  --output runs/ashare-direct-quantile/evaluation/metrics.json
```

The report includes raw quantile crossing, raw/sorted pinball loss, calibration,
60% and 80% coverage, width and interval score, price and return errors,
stock/date groups and a symbol-block bootstrap against last-close persistence.
An empirical training-quantile baseline reuses the calibration sample. Eight
illustrative windows are selected by seed before inference; separate P10/P50/P90
error examples explicitly describe their selection. Figures display actual
historical/future close prices, the forecast origin, the median and marginal
prediction intervals on actual timestamps.

Reference revisions:

- [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting/blob/1224fafa91e1ecd6674f5204066b159e40b5e4a1/src/chronos/chronos_bolt.py): direct horizon/quantile output.
- [TFT](https://github.com/google-research/google-research/blob/0eebfd403ba438bfd14cb70b9b24a7745c997083/tft/libs/utils.py): standard pinball objective.
- [TimesFM](https://github.com/google-research/timesfm/blob/e51928e27119cb17bebc005be2696b75e0a9e688/src/timesfm/configs.py): explicit quantile-crossing configuration.

The implementation uses a linear RWKV readout and quantile-axis rearrangement;
it retains this project's backbone, historical feature encoding and scaling.


### Recorded A-share experiment (2026-11-08)

One run used the 24-layer, width-256 architecture above, seed 1234 and 3592
optimizer steps. The fixed dataset manifest SHA256 was
`9040ce2d9500c033b6ea5d6c20e48bbb75d3cb784e1bb92b0975c9d658ff6e4d`.
Validation selected step 3592 with raw pinball loss 0.14478816. Evaluation reused
1024 frozen windows from 1024 instruments across 365 dates, with forecast origins
between 2025-01-02 and 2026-08-06. All forecasts used 64 observed bars to predict
30 future positions. The earlier autoregressive model was reused.

| Horizon (minutes) | Last-close MAE (%) | Earlier autoregressive MAE (%) | Direct median MAE (%) |
|---|---:|---:|---:|
| 1 | 0.11022 | 0.11172 | 0.10955 |
| 5 | 0.19590 | 0.20190 | 0.19683 |
| 15 | 0.32417 | 0.34831 | 0.32480 |
| 30 | 0.45269 | 0.50576 | 0.45204 |

MAE is normalized by each window's last observed close. The whole-path direct
median MAE was 0.31327%, compared with 0.31337% for last-close persistence.
The symbol-block bootstrap 95% confidence interval for the difference was
[-0.00407, 0.00395] percentage points. At minute 30, direct median RMSE was
0.80773%, versus 0.79122% for persistence. These point forecasts did not show a
clear advantage over persistence.

The mean sorted pinball loss was 12.40774 bp, versus 13.02918 bp for empirical
training quantiles, a 4.77% improvement. The paired symbol-block bootstrap 95%
confidence interval for that difference was [-0.79662, -0.46186] bp.

| Nominal interval | Direct coverage | Empirical coverage | Direct width (bp) | Empirical width (bp) | Direct interval score (bp) | Empirical interval score (bp) |
|---|---:|---:|---:|---:|---:|---:|
| 60% | 61.40% | 62.88% | 61.87 | 59.81 | 113.97 | 122.87 |
| 80% | 81.59% | 82.73% | 99.78 | 105.23 | 151.13 | 172.98 |

These interval statistics average all 30 forecast positions; width and interval
score use cumulative log-return bp. Raw outputs crossed in 10.27% of
window/horizon pairs. Sorting produced ordered delivered intervals; raw pinball
loss was 12.40869 bp.

Training used eight RTX PRO 6000 Blackwell GPUs, global batch 12288, per-GPU
microbatch 768 and two accumulation steps. Median steady-state iteration time
was 1.3264 seconds, or 592907 historical bars/second, excluding the first 100
steps. A 20-second sample measured mean GPU utilization between 94.35% and
95.70% across cards. Peak allocated memory was 25.67 GiB/GPU. Training through
final weight export took 81.99 minutes. No spike events were recorded under
the existing rolling-100-step thresholds; maximum raw gradient norm was 0.05194
and P99 was 0.04047.

The experiment used PyTorch 2.14.0+cu130, transformers-rwkv 5.21.0.dev0,
FlashRWKV2 0.1.0a18 and TileLang 0.1.15. The model implementation revision was
`b83bc1963cdfe0697f9bc2f0279a837675c4cddb`. Exact configuration, dependency
metadata, frozen windows, checkpoint SHA256, raw/sorted arrays, per-minute CSV,
metrics and figures are preserved in the run artifacts.

### Temporal and market covariates

`prepare-covariates` builds a separate mmap feature cache from the prepared
snapshot. It retains the five original inputs and appends 18 covariates:
calendar phase/session, trailing 5/15/30/60-bar returns and RMS returns,
relative volume, trailing up-bar fraction, and contemporaneous equal-weight
market return, cross-sectional volatility and up-stock fraction. Every rolling
calculation resets at the source segment, including session and split boundaries.
The market universe is the source manifest's instrument inventory. Only completed
bars at the input timestamp enter its aggregate. The cache manifest records the
source digest, feature order and temporal contract.

```bash
timeout 90m uv run --no-sync python examples/models/rwkv7/forecast.py prepare-covariates \
  --data "$FORECAST_DATA" --output "$FORECAST_COVARIATES" --workers 16

timeout 20m uv run --no-sync python examples/models/rwkv7/forecast.py select-windows \
  --data "$FORECAST_DATA" --output "$FORECAST_WINDOWS/validation" \
  --split validation --count 12288 --context 64 --horizon 30 --seed 1234
```

Selection uses a seeded candidate pool from stride-one inventory and round-robin
month/intraday-minute strata. It never selects on future prices. JSON records
include the stride, raw index, instrument and timestamps; NPY stores the same
indices for the native validation loader. `--exclude-dates old/windows.json`
excludes complete dates previously inspected when freezing an independent test
set. Existing 1024-window results remain a separate regression comparison.
The current same-session 64+30 contract admits 53 origin minutes in this snapshot;
opening and cross-session forecasts require a separate history-boundary change.

Add `--covariates "$FORECAST_COVARIATES"` and
`--validation-indices "$FORECAST_WINDOWS/validation.npy"` to direct-quantile
training. The checkpoint persists input dimensions, ordered feature names and
feature/validation identity hashes. Calibration uses training windows only.
`calibration.npz` preserves their final input vectors and cumulative-return
labels for fixed ridge (alpha=1) and volatility-quartile/session empirical
references. Neither reference uses test labels for fitting or selection.

Use `evaluate --split train` for fixed training-period diagnostics.
Direct-quantile training omits the native test loader; run explicit test evaluation
after validation-based checkpoint selection.

The callback saves `best-forecast.pt` for raw pinball and
`best-median-forecast.pt` for raw median cumulative-return MAE. Compare these
candidates on the frozen validation set before opening independent test labels;
use normalized price MAE as the primary point criterion and report conditional
reference pinball separately. Evaluation batches independent windows, reports
strict/weak CDFs and discrete calibration violations, and computes paired
confidence intervals separately with date and symbol blocks. `--metrics-only`
retains metrics and window identities while skipping arrays, CSV and figures.

Feature references: [Qlib Alpha158](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py),
[MASTER](https://github.com/SJTU-DMTai/MASTER/blob/de8f58557096abde4216a701b35fc4368158d111/README.md),
[Kronos temporal embeddings](https://arxiv.org/html/2508.02739v1), and
[DeepAR scale/covariate handling](https://arxiv.org/html/1704.04110v3).

### Recorded covariate experiment (2026-11-08)

The covariate run retained the 24-layer, width-256 RWKV7 backbone, seed 1234,
64/30 context/horizon, standard pinball objective, learning-rate schedule and
3592-step budget. It used 23 inputs, eight-way data parallelism, global batch
12288, microbatch 768 and four persistent data-loader workers per rank.
Training implementation revision: `0fdda583a44ad54f823e0f1c872c94652ac36b25`.
The selected step-500 checkpoint SHA256 was
`6619c5b697de383ff66b207766d6b529e09a446c3a92a1307957cba234d91bb0`.

The fixed validation set contained 12288 month/minute-stratified windows across
53 eligible intraday origin minutes. Raw validation pinball was 0.15349910 at
step 500, 0.16264886 at step 1000 and 0.17083074 at step 3592. Both validation
criteria selected step 500. A fresh process reproduced its raw pinball within
0.0001 bp. Selection used validation only; the framework's incidental final test
summary did not enter selection. The recipe now omits that native test loader.

The fixed test cohort excluded all 365 dates in the previously inspected
1024-window cohort. It contained 4096 windows from 2919 instruments across 21
remaining dates, with origins between 2025-01-13 and 2026-07-24 and all 53 eligible
origin minutes. Original and new models used identical window identities,
anchors, actual closes and timestamps.

| Metric | Last close | Previous direct model | Covariate model | Fixed ridge |
|---|---:|---:|---:|---:|
| Whole-path median/point MAE (%) | 0.305990 | 0.304305 | 0.307722 | 0.308657 |
| Minute-30 MAE (%) | 0.437482 | 0.434198 | 0.436400 | 0.440244 |
| Minute-30 RMSE (%) | 0.716538 | 0.720872 | 0.712640 | 0.717806 |
| Minute-30 return-error bias (bp) | 1.7411 | -1.8783 | -5.6527 | 0.7580 |
| Sorted pinball (bp) | — | 12.06683 | 12.23144 | — |

The unconditional and volatility-quartile/session empirical references scored
12.84557 and 12.57593 bp, respectively. The covariate model improved conditional
reference pinball by 2.74%; its paired date-block 95% confidence interval for the
whole-path difference was [-0.68069, -0.07328] bp. Its point-MAE difference against
last close was +0.001732 percentage points, with date-block interval
[-0.00682, 0.00897]. The point objective did not improve. Against the previous
direct model, the mean point difference was +0.003417 percentage points and the
date-block interval was [-0.00537, 0.01004].

The 60%/80% intervals covered 60.51%/80.05%, with mean widths 58.38/93.64 bp
and interval scores 112.88/150.09 bp. Raw quantile crossing occurred in 1.41% of
window/horizon pairs. On the original 1024-window regression cohort, the
covariate model's whole-path MAE was 0.314930%, versus 0.313271% for the previous
direct model and 0.313369% for last close. Its minute-30 return bias was -8.0343
bp, compared with -4.6983 bp for the reused recursive model.

The final step-3592 model achieved 0.291659% MAE on fixed training-period windows,
versus 0.359834% for last close, while validation deteriorated. This run provides
concrete evidence of a training/generalization gap. Adding this feature package
did not improve the primary point objective and does not replace the previous
model. The next controlled experiment should address temporal adaptation and
earlier validation-based stopping before expanding model size or training time.

Training and final export took 92.30 minutes. Median logged iteration time after
the first 100 steps was 1.3376 seconds, corresponding to 587943 historical
bars/second. The run processed 44138496 windows, 2824863744 historical bars and
1324154880 future targets. Both half-hour inspections observed 100% utilization
on all eight GPUs; the initial 20-second sample averaged 89.70–96.05% per GPU.
Peak allocated memory was 25.68 GiB per rank. No spike events, skipped updates or
NaN iterations were observed; maximum raw gradient norm was 0.15450.

All 71 related unit tests and all-file pre-commit checks passed. Strict mypy
reported no errors in five of seven changed implementation files; model/recipe
checking still reports inherited framework `Any` types and dependent return-type
errors. The full check log is retained with the runtime artifacts.
