# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample price paths from marginal quantile forecasts for entry and exit timing.

The forecast head emits one quantile set per horizon. Those marginals describe
where the price is at each minute, not how it travels, so they answer neither
"does it reach +0.5% before -0.5%" nor "when". Both questions need the joint
distribution over the 30 minutes.

This module draws paths whose per-horizon marginals are exactly the predicted
quantiles and whose step-to-step dependence follows a Gaussian copula fitted to
realized minute returns. Touch probabilities, first-touch minutes and running
extremes are then read off the sampled paths.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PathForecast:
    """Path statistics derived from sampled trajectories.

    Attributes:
        paths: ``[batch, samples, horizon]`` cumulative returns in the model's
            scaled log-return units.
        running_max: ``[batch, horizon, quantile]`` quantiles of the running maximum.
        running_min: ``[batch, horizon, quantile]`` quantiles of the running minimum.
    """

    paths: torch.Tensor
    running_max: torch.Tensor
    running_min: torch.Tensor

    def touch_probability(self, level: torch.Tensor | float) -> torch.Tensor:
        """Return the probability of reaching ``level`` within the horizon.

        Args:
            level: Scalar or ``[batch]`` barrier in the same units as the paths.
                A positive level is an upper barrier, a negative one a lower barrier.
        """
        level = torch.as_tensor(level, device=self.paths.device, dtype=self.paths.dtype)
        if level.ndim:
            level = level[:, None, None]
        reached = (self.paths >= level) if float(level.max()) > 0 else (self.paths <= level)
        return reached.any(dim=-1).to(self.paths.dtype).mean(dim=-1)

    def first_touch_minute(self, level: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the median first-touch minute and the probability of touching.

        The minute is one-based over paths that touch; it is ``nan`` where no
        sampled path reaches the level.
        """
        level = torch.as_tensor(level, device=self.paths.device, dtype=self.paths.dtype)
        if level.ndim:
            level = level[:, None, None]
        reached = (self.paths >= level) if float(level.max()) > 0 else (self.paths <= level)
        touched = reached.any(dim=-1)
        minute = reached.float().argmax(dim=-1) + 1
        minute = minute.masked_fill(~touched, 0).float()
        median = torch.where(
            touched.any(dim=-1),
            torch.nanmedian(minute.masked_fill(~touched, float("nan")), dim=-1).values,
            torch.full(touched.shape[:1], float("nan"), device=minute.device),
        )
        return median, touched.to(self.paths.dtype).mean(dim=-1)

    def barrier_race(self, upper: torch.Tensor | float, lower: torch.Tensor | float) -> torch.Tensor:
        """Return the probability that ``upper`` is reached before ``lower``.

        Paths that reach neither barrier are excluded, so the value is the
        conditional probability given that the trade is triggered at all.
        """
        up = torch.as_tensor(upper, device=self.paths.device, dtype=self.paths.dtype)
        down = torch.as_tensor(lower, device=self.paths.device, dtype=self.paths.dtype)
        if up.ndim:
            up, down = up[:, None, None], down[:, None, None]
        hit_up, hit_down = self.paths >= up, self.paths <= down
        horizon = self.paths.shape[-1]
        first_up = torch.where(hit_up.any(-1), hit_up.float().argmax(-1), horizon + 1)
        first_down = torch.where(hit_down.any(-1), hit_down.float().argmax(-1), horizon + 1)
        decided = first_up != first_down
        wins = (first_up < first_down) & decided
        total = decided.sum(dim=-1).clamp_min(1)
        return wins.sum(dim=-1) / total


def minute_return_correlation(horizon: int, autocorrelation: float, device: torch.device) -> torch.Tensor:
    """Build the correlation of cumulative returns from an MA(1) minute process.

    Realized one-minute returns carry a strong negative lag-one correlation,
    largely bid-ask bounce. An MA(1) process reproduces exactly that single-lag
    structure, since its lag-one correlation is ``theta / (1 + theta**2)``.
    The forecast fixes the spread at each horizon, so steps that partly cancel
    must individually be larger: such a path travels further before coming
    back, and reaches a barrier more often than independent steps would.
    """
    if not -0.5 < autocorrelation < 0.5:
        raise ValueError("An MA(1) process can only represent a lag-one correlation in (-0.5, 0.5).")
    # Invert rho = theta / (1 + theta^2), taking the invertible root |theta| < 1.
    if abs(autocorrelation) < 1e-12:
        theta = 0.0
    else:
        rho = autocorrelation
        theta = (1 - (1 - 4 * rho**2) ** 0.5) / (2 * rho)
    # Steps are e_t + theta * e_{t-1} over horizon + 1 independent innovations,
    # so the cumulative sums are a fixed linear map of those innovations.
    weights = torch.zeros(horizon, horizon + 1, device=device, dtype=torch.float32)
    index = torch.arange(horizon, device=device)
    weights[index, index + 1] = 1.0
    weights[index, index] += theta
    loading = torch.cumsum(weights, dim=0)
    covariance = loading @ loading.T
    scale = covariance.diagonal().sqrt()
    return covariance / torch.outer(scale, scale)


def sample_paths(
    quantiles: torch.Tensor,
    levels: tuple[float, ...],
    *,
    samples: int = 512,
    autocorrelation: float = -0.21,
    generator: torch.Generator | None = None,
) -> PathForecast:
    """Draw paths matching per-horizon quantiles under a Gaussian copula.

    Args:
        quantiles: ``[batch, horizon, quantile]`` sorted predicted quantiles.
        levels: The quantile levels, ascending, matching the last axis.
        samples: Paths drawn per window.
        autocorrelation: Lag-one correlation of one-minute returns. The default
            is the value measured on this dataset's 2024 validation split.
        generator: Optional seeded generator for reproducible sampling.

    Returns:
        A :class:`PathForecast` whose marginals interpolate ``quantiles``.
    """
    if quantiles.ndim != 3 or quantiles.shape[-1] != len(levels):
        raise ValueError(f"Expected [batch, horizon, {len(levels)}] quantiles.")
    batch, horizon, _ = quantiles.shape
    device = quantiles.device
    normal = torch.distributions.Normal(
        torch.zeros((), device=device, dtype=torch.float32), torch.ones((), device=device, dtype=torch.float32)
    )
    correlation = minute_return_correlation(horizon, autocorrelation, device)
    factor = torch.linalg.cholesky(correlation + 1e-6 * torch.eye(horizon, device=device))
    noise = torch.randn(batch, samples, horizon, device=device, dtype=torch.float32, generator=generator)
    # Correlated standard normals, then their uniform ranks under the copula.
    uniform = normal.cdf(noise @ factor.T).clamp(1e-6, 1 - 1e-6)
    # Map each uniform through the predicted marginal at its own horizon.
    grid = torch.tensor(levels, device=device, dtype=torch.float32)
    index = torch.searchsorted(grid.expand(1, -1).contiguous(), uniform.reshape(1, -1).contiguous()).reshape(
        batch, samples, horizon
    )
    index = index.clamp(1, len(levels) - 1)
    lower_level, upper_level = grid[index - 1], grid[index]
    values = quantiles.to(torch.float32).permute(0, 2, 1)  # [batch, quantile, horizon]
    step = torch.arange(horizon, device=device).expand(batch, samples, horizon)
    rows = torch.arange(batch, device=device)[:, None, None].expand_as(step)
    lower = values[rows, index - 1, step]
    upper = values[rows, index, step]
    weight = (uniform - lower_level) / (upper_level - lower_level).clamp_min(1e-12)
    paths = lower + weight * (upper - lower)
    # The path must be read in forecast order, so enforce no time reordering here.
    running_max = torch.cummax(paths, dim=-1).values
    running_min = torch.cummin(paths, dim=-1).values
    probs = torch.tensor(levels, device=device, dtype=torch.float32)
    return PathForecast(
        paths=paths,
        running_max=torch.quantile(running_max, probs, dim=1).permute(1, 2, 0),
        running_min=torch.quantile(running_min, probs, dim=1).permute(1, 2, 0),
    )
