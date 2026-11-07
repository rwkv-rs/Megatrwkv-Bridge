"""Validate forecast CLI inputs before loading the CUDA training runtime."""

import runpy
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ENTRY = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast.py"


@pytest.mark.parametrize("command", [[], ["prepare"], ["train"], ["predict"], ["evaluate"]])
def test_help_without_training_dependencies(monkeypatch, command):
    monkeypatch.setattr("sys.argv", [str(ENTRY), *command, "--help"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ENTRY), run_name="__main__")
    assert result.value.code == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "0"],
        ["--epochs", "0"],
        ["--warmup-steps", "10000"],
        ["--global-batch-size", "7"],
        ["--context", "0"],
        ["--eval-iters", "-1"],
        ["--exit-interval", "0"],
    ],
)
def test_reject_invalid_training_inputs(monkeypatch, arguments):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        "sys.argv", [str(ENTRY), "train", "--config", "config.json", "--data", "data", "--output", "run", *arguments]
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ENTRY), run_name="__main__")
    assert result.value.code == 2


def test_reject_zero_horizon(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            str(ENTRY),
            "predict",
            "--checkpoint",
            "forecast.pt",
            "--instrument",
            "stock.parquet",
            "--output",
            "forecast.csv",
            "--horizon",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ENTRY), run_name="__main__")
    assert result.value.code == 2
