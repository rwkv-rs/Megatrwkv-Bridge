# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent, past-only VMD and Haar thresholding for forecast windows."""

import math

import torch
import torch.nn.functional as F


CONTEXT = 64
VMD_ITERATIONS = 500
VMD_TOLERANCE = 1e-7
PEARSON_THRESHOLD = 0.1


def vmd_modes(history: torch.Tensor, *, modes: int, alpha: float, iterations: int = VMD_ITERATIONS) -> torch.Tensor:
    """Decompose each historical window independently, with mirrored boundaries."""
    half = history.shape[-1] // 2
    extended = torch.cat((history[:, :half].flip(-1), history, history[:, -half:].flip(-1)), dim=-1)
    spectrum = torch.fft.rfft(extended)[:, :-1].contiguous()
    if spectrum.is_cuda:
        from megatron.bridge.data.datasets.ohlcv_vmd_cuda import vmd_spectra

        components = vmd_spectra(spectrum, modes=modes, alpha=alpha, iterations=iterations, tol=VMD_TOLERANCE)
    else:
        components = _vmd_reference(spectrum, modes=modes, alpha=alpha, iterations=iterations)
    return torch.fft.irfft(F.pad(components, (0, 1)), n=extended.shape[-1])[..., half : 3 * half]


def _vmd_reference(spectrum: torch.Tensor, *, modes: int, alpha: float, iterations: int) -> torch.Tensor:
    bins = spectrum.shape[-1]
    frequency = torch.arange(bins, dtype=spectrum.real.dtype, device=spectrum.device) / (2 * bins)
    centers = [(frequency.new_full((len(spectrum), 1), index / (2 * modes))) for index in range(modes)]
    components = [torch.zeros_like(spectrum) for _ in range(modes)]
    active = torch.ones((len(spectrum), 1), dtype=torch.bool, device=spectrum.device)
    for _ in range(iterations):
        residual = spectrum - torch.stack(components).sum(0)
        change = torch.zeros_like(centers[0])
        for index, old in enumerate(components):
            candidate = (residual + old) / (1 + alpha * (frequency - centers[index]).square())
            new = torch.where(active, candidate, old)
            residual = residual + old - new
            change += (new - old).abs().square().sum(-1, keepdim=True)
            power = new.abs().square()
            centers[index] = (power * frequency).sum(-1, keepdim=True) / power.sum(-1, keepdim=True).clamp_min(1e-30)
            components[index] = new
        active &= change / (2 * bins) > VMD_TOLERANCE
        if not active.any():
            break
    return torch.stack(components, dim=1)


def haar_threshold(signal: torch.Tensor) -> torch.Tensor:
    """Apply three levels of orthonormal Haar detail shrinkage with a MAD threshold."""
    scale = math.sqrt(0.5)
    details = []
    approximation = signal
    for _ in range(3):
        even, odd = approximation[..., ::2], approximation[..., 1::2]
        approximation = (even + odd) * scale
        details.append((even - odd) * scale)
    finest = details[0]
    center = _median(finest)
    sigma = _median((finest - center).abs()) / 0.6744897501960817
    threshold = sigma * math.sqrt(2 * math.log(signal.shape[-1]))
    for detail in reversed(details):
        detail = detail.sign() * (detail.abs() - threshold).clamp_min(0)
        approximation = torch.stack((approximation + detail, approximation - detail), dim=-1).flatten(-2) * scale
    return approximation


def _median(values: torch.Tensor) -> torch.Tensor:
    middle = values.shape[-1] // 2
    ordered = values.sort(dim=-1).values
    return (ordered[..., middle - 1 : middle] + ordered[..., middle : middle + 1]) / 2


@torch.inference_mode()
def denoise_history(close: torch.Tensor, *, modes: int, alpha: float) -> torch.Tensor:
    """Return the denoised relative log-price sequence from 64 completed closes."""
    signal = (100 * torch.log(close / close[:, :1])).float()
    components = vmd_modes(signal, modes=modes, alpha=alpha)
    centered = signal - signal.mean(-1, keepdim=True)
    centered_modes = components - components.mean(-1, keepdim=True)
    denominator = centered.norm(dim=-1, keepdim=True) * centered_modes.norm(dim=-1)
    correlation = (centered_modes * centered[:, None]).sum(-1) / denominator.clamp_min(1e-30)
    noise = (components * (correlation < PEARSON_THRESHOLD).unsqueeze(-1)).sum(1)
    # The original signal less the noise group contains the information group and residual.
    return signal - noise + haar_threshold(noise)


def envelope_entropy(components: torch.Tensor) -> torch.Tensor:
    """Average normalized Hilbert-envelope entropy over nonzero modes."""
    length = components.shape[-1]
    analytic = torch.fft.fft(components)
    analytic[..., 1 : length // 2] *= 2
    analytic[..., length // 2 + 1 :] = 0
    envelope = torch.fft.ifft(analytic).abs()
    mass = envelope.sum(-1, keepdim=True)
    valid = mass.squeeze(-1) > 1e-12
    probability = envelope / mass.clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1) / math.log(length)
    return (entropy * valid).sum() / valid.sum().clamp_min(1)
