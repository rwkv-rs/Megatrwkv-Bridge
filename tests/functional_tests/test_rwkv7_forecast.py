"""One-GPU continuous RWKV7 training, recurrent rollout and persisted inference."""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import RwkvConfig


pytest.importorskip("transformers.integrations.flash_rwkv2", reason="Requires transformers-rwkv RWKV-7")
pa = pytest.importorskip("pyarrow", reason="Requires the parquet extra")
pq = pytest.importorskip("pyarrow.parquet", reason="Requires the parquet extra")

from megatron.bridge.data.datasets.ohlcv import decode_bars, encode_bars
from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import RWKV7ForecastModel
from megatron.bridge.recipes.rwkv7.forecast import rwkv7_ohlcv_config


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and FlashRWKV2")


@pytest.mark.parametrize("initialization", ["linear", "calibrated"])
def test_forecast_gradients_cached_rollout_and_fresh_process_reload(tmp_path, initialization):
    torch.manual_seed(2026)
    hf = RwkvConfig(hidden_size=64, num_hidden_layers=2, vocab_size=1, context_length=16)
    if initialization != "linear":
        hf.forecast_initialization = "calibrated"
        hf.forecast_target_mean = [0.0, 0.0, 0.03, 0.04, 0.85]
        hf.forecast_head_gain = 0.01
        hf.forecast_head_fp32 = True
        hf.forecast_input_mean = [0.0, 0.0, 0.03, 0.04, 0.85]
        hf.forecast_input_std = [0.1, 0.1, 0.05, 0.05, 0.3]
    config = rwkv7_ohlcv_config(
        hf,
        data=tmp_path,
        output=tmp_path,
        steps=2,
        micro_batch_size=1,
        global_batch_size=1,
        learning_rate=1e-3,
        warmup_steps=0,
        save_interval=1,
        eval_interval=1,
        eval_iters=0,
        seed=2026,
    ).model
    model = RWKV7ForecastModel(config, None).cuda().train()
    if initialization != "linear":
        silent = torch.zeros(1, 16, 5, device="cuda")
        targets = torch.tensor(hf.forecast_target_mean, device="cuda").expand_as(silent)
        initial_loss = model(silent, labels=targets).mean()
        initial_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert initial_loss < 1e-3 and norm < 1
        model.zero_grad(set_to_none=True)
    bars = torch.randn(1, 17, 5, device="cuda") * 0.1
    bars[..., 2:] = bars[..., 2:].abs()
    before = model.input_projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(bars[:, :-1], labels=bars[:, 1:]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.input_projection.weight.grad.abs().sum() > 0
    assert model.output_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.input_projection.weight)
    model.eval()
    with torch.inference_mode():
        cached = model.generate(bars[:, :-1], horizon=3)
        replay = bars[:, :-1]
        expected = []
        for _ in range(3):
            predicted = model(replay)[:, -1:]
            expected.append(predicted)
            replay = torch.cat((replay, predicted), dim=1)
        torch.testing.assert_close(cached, torch.cat(expected, dim=1), rtol=0.03, atol=0.03)
        torch.testing.assert_close(model.generate(bars[:, :-1], horizon=3), cached, rtol=0, atol=0)
        shorter = model.generate(bars[:, :7], horizon=3)
        assert shorter.shape == (1, 3, 5) and torch.isfinite(shorter).all()
    checkpoint = tmp_path / "forecast.pt"
    torch.save(
        {
            "version": 1,
            "step": 1,
            "run": {"hf_config": json.loads(hf.to_json_string())},
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        },
        checkpoint,
    )
    instrument = tmp_path / "000001.parquet"
    times = np.datetime64("2026-01-05T09:30", "us") + np.arange(20) * np.timedelta64(1, "m")
    pq.write_table(
        pa.table(
            {
                "exchange": ["SZ"] * 20,
                "symbol": ["000001"] * 20,
                "timestamp": times,
                "open": [10.0] * 20,
                "high": [10.1] * 20,
                "low": [9.9] * 20,
                "close": [10.0] * 20,
                "volume": [1000] * 20,
            }
        ),
        instrument,
    )
    raw_history = np.tile([10.0, 10.1, 9.9, 10.0, 1000.0], (17, 1))
    context = torch.from_numpy(encode_bars(raw_history)).unsqueeze(0).cuda()
    original_forecast = decode_bars(model.generate(context, horizon=3)[0].cpu().numpy(), previous_close=10.0)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "prediction.csv"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "examples/models/rwkv7/forecast.py",
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--instrument",
            str(instrument),
            "--output",
            str(output),
            "--horizon",
            "3",
        ],
        cwd=root,
        check=True,
        timeout=300,
    )
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and all(row["symbol"] == "000001" for row in rows)
    loaded_forecast = np.array(
        [[float(row[key]) for key in ("open", "high", "low", "close", "volume")] for row in rows]
    )
    np.testing.assert_allclose(loaded_forecast, original_forecast, rtol=1e-6, atol=1e-6)
    for row in rows:
        opening, high, low, close, volume = (float(row[key]) for key in ("open", "high", "low", "close", "volume"))
        assert high >= max(opening, close) >= min(opening, close) >= low > 0
        assert volume >= 0


def test_direct_quantile_backward_state_reset_and_fresh_reload(tmp_path):
    """Exercise actual recurrent kernels and independently reload persisted weights."""
    torch.manual_seed(2026)
    hf = RwkvConfig(hidden_size=64, num_hidden_layers=2, vocab_size=1, context_length=16)
    hf.forecast_task = "direct_quantile"
    hf.forecast_horizon = 3
    hf.forecast_quantiles = [0.1, 0.5, 0.9]
    config = rwkv7_ohlcv_config(
        hf,
        data=tmp_path,
        output=tmp_path,
        steps=2,
        micro_batch_size=1,
        global_batch_size=1,
        learning_rate=1e-3,
        warmup_steps=0,
        save_interval=1,
        eval_interval=1,
        eval_iters=0,
        seed=2026,
    ).model
    model = RWKV7ForecastModel(config, None).cuda().train()
    inputs = torch.randn(2, 16, 5, device="cuda") * 0.1
    labels = torch.randn(2, 3, device="cuda") * 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.output_projection.weight.detach().clone()
    loss = model(inputs, labels).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for prefix in ("hf.layers", "input_projection", "output_projection"):
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.float().abs().sum() for g in gradients) > 0
    optimizer.step()
    assert not torch.equal(before, model.output_projection.weight)
    model.eval()
    prediction = model.predict_quantiles(inputs)
    model.predict_quantiles(torch.randn_like(inputs))
    torch.testing.assert_close(model.predict_quantiles(inputs), prediction, atol=0, rtol=0)
    payload = {
        "version": 1,
        "step": 1,
        "run": {"hf_config": json.loads(hf.to_json_string())},
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, tmp_path / "forecast.pt")
    torch.save(inputs.cpu(), tmp_path / "inputs.pt")
    script = (
        "import runpy,sys,torch; from pathlib import Path; "
        "entry=runpy.run_path('examples/models/rwkv7/forecast.py'); "
        "root=Path(sys.argv[1]); model,_=entry['load_forecaster'](root/'forecast.pt'); "
        "x=torch.load(root/'inputs.pt',weights_only=True).cuda(); "
        "torch.save(model.predict_quantiles(x).cpu(),root/'predicted.pt')"
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=300,
    )
    torch.testing.assert_close(
        torch.load(tmp_path / "predicted.pt", weights_only=True), prediction.cpu(), atol=0, rtol=0
    )
