"""Validate forecast CLI inputs before loading the CUDA training runtime."""

import argparse
import runpy
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ENTRY = Path(__file__).resolve().parents[3] / "examples/models/rwkv7/forecast.py"


@pytest.mark.parametrize("command", [[], ["prepare"], ["train"], ["predict"], ["evaluate"], ["pack"]])
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
        ["--eval-interval", "0", "--early-stopping-patience", "0"],
        ["--eval-iters", "-1"],
        ["--early-stopping-patience", "-1"],
        ["--task", "direct_quantile", "--calibration-windows", "1", "--eval-interval", "300"],
        ["--exit-interval", "0"],
        ["--normalize-inputs"],
        ["--task", "direct_quantile"],
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


@pytest.mark.parametrize(
    "arguments",
    [
        ["--task", "next_bar", "--eval-interval", "300"],
        ["--task", "direct_quantile", "--calibration-windows", "1", "--eval-iters", "0", "--eval-interval", "300"],
        [
            "--task",
            "direct_quantile",
            "--calibration-windows",
            "1",
            "--early-stopping-patience",
            "0",
            "--eval-interval",
            "300",
        ],
    ],
)
def test_eval_interval_is_unrestricted_when_early_stopping_is_inactive(monkeypatch, arguments):
    original_parse_args = argparse.ArgumentParser.parse_args
    received = []

    def parse_args(parser):
        args = original_parse_args(parser)
        args.handler = received.append
        return args

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    monkeypatch.setattr(
        "sys.argv", [str(ENTRY), "train", "--config", "config.json", "--data", "data", "--output", "run", *arguments]
    )
    runpy.run_path(str(ENTRY), run_name="__main__")
    (args,) = received
    assert args.eval_interval == 300


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
