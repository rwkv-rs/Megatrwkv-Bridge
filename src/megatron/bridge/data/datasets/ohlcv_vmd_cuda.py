# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused CUDA iteration for 64-point historical VMD windows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _iterate_vmd(
    spectrum,
    result,
    modes: tl.constexpr,
    rows: tl.constexpr,
    bins: tl.constexpr,
    iterations: tl.constexpr,
    alpha,
    tolerance: tl.constexpr,
):
    window = tl.program_id(0)
    frequency_index = tl.arange(0, bins)
    mode_index = tl.arange(0, rows)
    frequency = frequency_index.to(tl.float32) / (2 * bins)
    signal_offset = (window * bins + frequency_index) * 2
    signal_real = tl.load(spectrum + signal_offset)
    signal_imag = tl.load(spectrum + signal_offset + 1)
    real = tl.full((rows, bins), 0, tl.float32)
    imag = tl.full((rows, bins), 0, tl.float32)
    centers = mode_index.to(tl.float32) / (2 * modes)

    iteration = 0
    change = float("inf")
    while (iteration < iterations) & (change > tolerance):
        residual_real = signal_real - tl.sum(real, axis=0)
        residual_imag = signal_imag - tl.sum(imag, axis=0)
        change = 0.0
        for mode in range(modes):
            lookup = tl.full((1, bins), mode, tl.int32)
            old_real = tl.gather(real, lookup, 0).reshape((bins,))
            old_imag = tl.gather(imag, lookup, 0).reshape((bins,))
            center = tl.sum(tl.where(mode_index == mode, centers, 0), 0)
            delta = frequency - center
            divisor = 1.0 + alpha * delta * delta
            new_real = (residual_real + old_real) / divisor
            new_imag = (residual_imag + old_imag) / divisor
            step_real = old_real - new_real
            step_imag = old_imag - new_imag
            residual_real += step_real
            residual_imag += step_imag
            change += tl.sum(step_real * step_real + step_imag * step_imag, 0) / (2 * bins)
            power = new_real * new_real + new_imag * new_imag
            center = tl.sum(power * frequency, 0) / tl.maximum(tl.sum(power, 0), 1e-30)
            centers = tl.where(mode_index == mode, center, centers)
            real = tl.where(mode_index[:, None] == mode, new_real[None, :], real)
            imag = tl.where(mode_index[:, None] == mode, new_imag[None, :], imag)
        iteration += 1

    output_offset = ((window * modes + mode_index[:, None]) * bins + frequency_index[None, :]) * 2
    tl.store(result + output_offset, real, mode_index[:, None] < modes)
    tl.store(result + output_offset + 1, imag, mode_index[:, None] < modes)


def vmd_spectra(spectrum: torch.Tensor, *, modes: int, alpha: float, iterations: int, tol: float) -> torch.Tensor:
    """Solve each frequency-domain window in one CUDA program."""
    batch, bins = spectrum.shape
    result = torch.empty((batch, modes, bins, 2), device=spectrum.device, dtype=torch.float32)
    _iterate_vmd[(batch,)](
        torch.view_as_real(spectrum),
        result,
        modes,
        triton.next_power_of_2(modes),
        bins,
        iterations,
        alpha,
        tol,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return torch.view_as_complex(result)
