"""Causality, fixed-length features and reference VMD behavior."""

import math

import numpy as np
import pytest
import torch

from megatron.bridge.data.datasets.ohlcv_vmd import denoise_history, haar_threshold, vmd_modes
from megatron.bridge.models.rwkv7.modeling_rwkv7_forecast import ForecastFeatureMixer


pytestmark = pytest.mark.unit


def _reference_vmd(signal, modes, alpha, iterations):
    length = len(signal)
    half = length // 2
    extended = np.r_[signal[:half][::-1], signal, signal[half:][::-1]]
    spectrum = np.fft.rfft(extended)[:-1]
    frequencies = np.arange(length) / (2 * length)
    coefficients = np.zeros((modes, length), dtype=np.complex128)
    centers = np.arange(modes) / (2 * modes)
    for _ in range(iterations):
        change = 0.0
        for index in range(modes):
            others = coefficients[np.arange(modes) != index].sum(axis=0)
            updated = (spectrum - others) / (1 + alpha * (frequencies - centers[index]) ** 2)
            change += np.square(np.abs(updated - coefficients[index])).sum()
            power = np.square(np.abs(updated))
            centers[index] = (power @ frequencies) / max(power.sum(), 1e-30)
            coefficients[index] = updated
        if change / (2 * length) <= 1e-7:
            break
    return np.fft.irfft(np.pad(coefficients, ((0, 0), (0, 1))))[:, half : 3 * half]


def test_independent_vmd_matches_scalar_reference():
    time = torch.arange(64, dtype=torch.float32)
    history = torch.stack((torch.sin(time / 6) + 0.25 * torch.cos(time * 2), torch.zeros_like(time)))
    result = vmd_modes(history, modes=3, alpha=1200, iterations=120)
    reference = np.stack([_reference_vmd(row, 3, 1200, 120) for row in history.numpy()])
    torch.testing.assert_close(result, torch.from_numpy(reference).float(), atol=2e-5, rtol=3e-4)
    torch.testing.assert_close(result[:1], vmd_modes(history[:1], modes=3, alpha=1200, iterations=120))


def test_haar_threshold_matches_orthogonal_basis():
    signal = np.random.default_rng(12).normal(size=64)
    approximation = np.eye(64)
    details = []
    for _ in range(3):
        even, odd = approximation[::2], approximation[1::2]
        details.append((even - odd) / math.sqrt(2))
        approximation = (even + odd) / math.sqrt(2)
    finest = details[0] @ signal
    sigma = np.median(np.abs(finest - np.median(finest))) / 0.6744897501960817
    threshold = sigma * math.sqrt(2 * math.log(len(signal)))
    reconstructed = approximation.T @ (approximation @ signal)
    for matrix in details:
        coefficients = matrix @ signal
        reconstructed += matrix.T @ (np.sign(coefficients) * np.maximum(0, np.abs(coefficients) - threshold))
    result = haar_threshold(torch.from_numpy(signal)[None])[0]
    torch.testing.assert_close(result, torch.from_numpy(reconstructed))


def test_future_mutation_and_constant_history():
    series = torch.linspace(100, 102, 94, dtype=torch.float64)
    expected = denoise_history(series[:64][None], modes=2, alpha=1000)
    series[64:] = 200
    torch.testing.assert_close(denoise_history(series[:64][None], modes=2, alpha=1000), expected)
    constant = denoise_history(torch.full((1, 64), 10.0), modes=2, alpha=1000)
    torch.testing.assert_close(constant, torch.zeros_like(constant))


def test_cnn_encoding_is_causal_and_attention_updates_all_parameters():
    module = ForecastFeatureMixer(5, 64)
    inputs = torch.randn(2, 64, 5, dtype=torch.bfloat16)
    original, tokens = module.encode(inputs)
    inputs[:, 20:] = 0
    modified, _ = module.encode(inputs)
    torch.testing.assert_close(original[:, :20], modified[:, :20], rtol=0, atol=0)
    module(original[:, -1], tokens).float().square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in module.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA VMD parity requires a GPU")
def test_fused_vmd_matches_cpu():
    history = torch.randn((16, 64), generator=torch.Generator().manual_seed(11)) * 0.3
    expected = vmd_modes(history, modes=9, alpha=2980, iterations=100)
    actual = vmd_modes(history.cuda(), modes=9, alpha=2980, iterations=100).cpu()
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-3)
