---
language:
  - zh
  - en
license: apache-2.0
library_name: transformers
tags:
  - rwkv7
  - time-series
  - quantile-regression
  - forecasting
  - finance
datasets:
  - neigezhu/china-a-share-1min-ohlcv
---

# RWKV7 A-share 1-minute direct quantile forecasting

使用中国 A 股分钟 OHLCV 训练的 **21,017,102 参数** RWKV7 分位数预测模型。
输入为 **64 分钟 × 23 个历史特征**，一次输出未来 **1～30 分钟**收盘价的
**0.1、0.2、…、0.9** 九个分位数。中位数是第 5 个输出分位数。

This is a standalone Transformers `PreTrainedModel` with safetensors weights,
custom `AutoConfig` / `AutoModel` registration and a CLI inference script.
It composes the public transformers-rwkv backbone; Megatron Bridge is not an
inference dependency. The checkpoint was trained from random initialization.

## 安装与直接运行 / Quick start

需要 NVIDIA CUDA GPU。已验证环境：Python 3.12、PyTorch 2.14.0+cu130、
FlashRWKV2 0.1.0a18、TileLang 0.1.15、RTX PRO 6000 Blackwell。
安装的是 requirements.txt 中固定 revision 的 **transformers-rwkv** 分支。
该分支通过 `transformers` 包导入，并提供本模型需要的 RWKV7 实现。

```bash
timeout 2m uv venv --python 3.12 .venv
timeout 30m uv pip install -r requirements.txt
timeout 10m uv run --no-sync python inference.py \
  --model Caizus/rwkv7-ashare-1min-quantile-21m \
  --input example_bars.csv --output predictions.csv
```

先从本仓库下载 `requirements.txt`、`inference.py`、
`processing_rwkv7_forecast.py` 和 `example_bars.csv`，再运行以上命令。
`example_bars.csv` 是完整的 **65 行合成样本**，用于验证调用流程。
模型第一次推理可能编译或调优 CUDA kernel；后续调用复用缓存。

若已经下载完整模型仓库，也可用 `--model .` 从本地加载。
生产复现时把模型页面的完整 commit 填入 `--revision COMMIT_SHA`。

## Python API

```python
import numpy as np
import torch
from transformers import AutoModel

model = (
    AutoModel.from_pretrained(
        "Caizus/rwkv7-ashare-1min-quantile-21m",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    .cuda()
    .eval()
)

with np.load("example_inputs.npz", allow_pickle=False) as sample:
    features = torch.from_numpy(sample["features"]).unsqueeze(0).cuda()
    anchor = torch.tensor([float(sample["anchor_price"].item())], device="cuda", dtype=torch.float64)

with torch.inference_mode():
    output = model(features=features, anchor_price=anchor)

raw_returns = output.raw_quantiles  # [B, 30, 9], FP32, before rearrangement
ordered_returns = output.quantiles  # [B, 30, 9], FP32, sorted on quantile axis
prices = output.price_quantiles  # [B, 30, 9], FP64, same units as anchor
median_prices = prices[..., 4]  # [B, 30]
interval_60 = prices[..., [1, 7]]  # q0.2 ... q0.8
interval_80 = prices[..., [0, 8]]  # q0.1 ... q0.9
```

模型内置训练期输入均值与标准差。传入未经标准化的编码特征。
每次调用重置 RWKV 状态，独立窗口不会串联。主干为 BF16，WKV 状态与累计为 FP32
（`fp32io16`）；输出头在 FP32 投影。`fix_quantile_crossing=False` 返回原始顺序。
可选 `labels=[B,30]` 会计算原始输出的标准 pinball loss。

## 自备数据 / Input contract

CSV 必须包含以下列：

```text
timestamp,open,high,low,close,volume,market_return,market_volatility,market_up_fraction
```

每份 CSV 对应一个标的，从实际连续交易时段的起始 bar 到预测起点的完整前缀，
至少 **65 个原始 bar**。时间戳为上海本地时间，不带时区后缀，严格一分钟连续。
第一个 bar 提供上一收盘价锚点，编码时丢弃；最后 64 个编码 bar 输入模型。
滚动统计需要前缀中的更早历史，因此不要随意截断成最后 65 行。
上午、下午、日期、缺失分钟各自构成独立时段，不能跨午休填充。
训练和评估使用完整 64 分钟历史及同一时段内的 30 分钟未来。
相应预测起点位于上午约 10:34～11:00、下午约 14:05～14:30。

原始 OHLC 价格单位为元/股，volume 使用与数据集相同的股数口径。
不在预处理过程中填充缺失 bar 或调整价格。
`processing_rwkv7_forecast.prepare_features` 返回 `[64,23]` 和最后收盘价。

五个基础特征：

| 特征 | 公式 |
|---|---|
| open_return | `100 * log(open / previous_close)` |
| close_return | `100 * log(close / previous_close)` |
| upper_wick | `100 * log(high / max(open, close))` |
| lower_wick | `100 * log(min(open, close) / low)` |
| log_volume | `log1p(volume) / 10` |

后续 18 个特征按顺序为：分钟周期 sin/cos、下午标记、星期周期 sin/cos、
5/15/30/60 分钟累计收益、相同窗口的收益 RMS、60 分钟相对成交量、
60 分钟上涨比例，以及三个同期市场统计。完整名称保存在
`config.json` 的 `backbone_config.forecast_feature_names`。

`market_return` 是同一分钟内固定股票集合中可用的已完成 bar 的
`100 * log(close / previous_close)` 的等权平均值；`market_volatility`
是这些收益的总体标准差；`market_up_fraction` 是正收益的占比。
每个标的在时段/缺失边界后的第一个原始 bar 不产生收益。
`market_statistics([time,stocks])` 可由收益矩阵计算这三个量，缺失值为 NaN。
训练的股票集合是数据 manifest 中的全部标的；实际数据源和集合改变会改变输入分布。
不要用单只股票的收益替代市场统计，或用未来分钟计算它们。

已有正确编码特征时，也可提供 NPZ：`features` 为 `[64,23]`，
`anchor_price` 为正的标量。使用同一脚本的 `--input example_inputs.npz`。

输出 CSV 包含 270 行（30 horizons × 9 quantiles），保留原始与排序后的
累计收益、还原的收盘价及可用的未来时间戳。目标及价格还原为：

```text
y[h] = 100 * log(close[t+h] / close[t])
predicted_close[h,q] = close[t] * exp(predicted_y[h,q] / 100)
```

所有 horizon 共用最后历史收盘价，解码时逐 horizon 直接还原。

## 实测表现 / Evaluation

发布权重为固定 validation 集选出的 **第 500 步**；主训练完成 3592 步。
延长训练至第 15231 步后，验证集没有改善，续训停止，选择仍为第 500 步。
模型是 24 层、hidden size 256、intermediate size 1024，seed 1234。

固定测试集有 **4096 个窗口、2919 个标的、21 个日期**，预测起点覆盖
2025-01-13～2026-07-24。按月份和日内分钟分层抽样，并排除之前评估的
1024 个窗口所在日期。完整数值见 `evaluation.json`。

| 指标 | 本模型 | 参照 |
|---|---:|---:|
| 30 分钟整段中位数 MAE，按起点价格归一化 | 0.30772% | 价格不变：0.30599% |
| 第 30 分钟中位数 MAE | 0.43640% | 价格不变：0.43748% |
| 排序后平均 pinball loss | 12.23144 bp | 训练集无条件分位数：12.84557 bp |
| 排序后平均 pinball loss | 12.23144 bp | 波动率分组 × 时段分位数：12.57593 bp |
| 60% 区间覆盖率 | 60.509% | 标称 60% |
| 80% 区间覆盖率 | 80.054% | 标称 80% |
| 第 30 分钟累计收益预测偏置 | -5.65274 bp | — |

本模型的整段中位数 MAE 没有超过价格不变参照；概率预测损失优于上述经验分位数参照。
发布包保留这些结果，便于复现实验和继续研究。训练日期边界为 2024-01-01，
验证日期边界为 2025-01-01，测试从后者开始；指标中的 bp 为累计对数收益误差单位。

## 文件与版本

- `model-*.safetensors` 与 `model.safetensors.index.json`：BF16 权重分片与索引。
- `config.json`、`configuration_rwkv7_forecast.py`、`modeling_rwkv7_forecast.py`：标准 AutoClass 接口。
- `processing_rwkv7_forecast.py`、`inference.py`：独立预处理与推理。
- `example_bars.csv`、`example_inputs.npz`：同一个合成示例的两种输入格式。
- `provenance.json`、`verification.json`：来源、哈希、精确权重及前向重载校验。
- `evaluation.json`：真实固定测试集结果。

backbone 使用 [transformers-rwkv 固定 revision](https://github.com/rwkv-rs/transformers-rwkv/tree/f93dd4e01e0b565ef0390f76c6726d1d121ebbab)。
数据来源为 [neigezhu/china-a-share-1min-ohlcv](https://huggingface.co/datasets/neigezhu/china-a-share-1min-ohlcv)。
接口遵循 [Transformers 自定义模型规范](https://huggingface.co/docs/transformers/main/en/custom_models)。
代码与权重按 Apache-2.0 发布；依赖及原始数据遵循各自许可证。
