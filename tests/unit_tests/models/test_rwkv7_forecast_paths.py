"""Path sampling contracts: marginals, ordering, barriers and timing."""

import numpy as np
import pytest
import torch

from megatron.bridge.models.rwkv7.forecast_paths import minute_return_correlation, sample_paths


pytestmark = pytest.mark.unit
LEVELS = tuple(i / 10 for i in range(1, 10))


def _quantiles(batch=2, horizon=30, scale=1.0):
    """Quantiles of a random walk: standard normal widening with sqrt(h)."""
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(LEVELS))
    spread = torch.arange(1, horizon + 1, dtype=torch.float32).sqrt()[:, None] * z[None, :] * scale
    return spread.expand(batch, horizon, len(LEVELS)).contiguous()


def test_sampled_marginals_reproduce_the_predicted_quantiles():
    quantiles = _quantiles()
    generator = torch.Generator().manual_seed(0)
    forecast = sample_paths(quantiles, LEVELS, samples=20000, generator=generator)
    assert forecast.paths.shape == (2, 20000, 30)
    for h in (0, 14, 29):
        empirical = torch.quantile(forecast.paths[0, :, h], torch.tensor(LEVELS))
        tolerance = 4 * float(forecast.paths[0, :, h].std()) / forecast.paths.shape[1] ** 0.5
        assert (empirical - quantiles[0, h]).abs().max() < tolerance, h


def test_running_extremes_bracket_the_endpoint_and_never_reverse():
    forecast = sample_paths(_quantiles(), LEVELS, samples=4000, generator=torch.Generator().manual_seed(1))
    assert (forecast.running_max >= forecast.running_min).all()
    # A running maximum cannot fall, nor a running minimum rise.
    assert (forecast.running_max.diff(dim=1) >= -1e-5).all()
    assert (forecast.running_min.diff(dim=1) <= 1e-5).all()
    # The band for the extreme over 30 minutes is wider than for the endpoint.
    assert forecast.running_max[0, -1, -1] > _quantiles()[0, -1, -1]


def test_touch_probability_and_first_touch_minute_are_ordered():
    forecast = sample_paths(_quantiles(), LEVELS, samples=4000, generator=torch.Generator().manual_seed(2))
    near, far = forecast.touch_probability(1.0), forecast.touch_probability(4.0)
    assert (near > far).all() and (near <= 1).all() and (far >= 0).all()
    close_minute, _ = forecast.first_touch_minute(1.0)
    distant_minute, _ = forecast.first_touch_minute(3.0)
    # A nearer barrier is reached sooner.
    assert close_minute[0] < distant_minute[0]


def test_barrier_race_is_symmetric_for_a_symmetric_forecast():
    forecast = sample_paths(_quantiles(), LEVELS, samples=20000, generator=torch.Generator().manual_seed(3))
    assert forecast.barrier_race(2.0, -2.0)[0] == pytest.approx(0.5, abs=0.03)
    # A forecast shifted upward reaches the upper barrier first more often.
    shifted = sample_paths(_quantiles() + 2.0, LEVELS, samples=20000, generator=torch.Generator().manual_seed(3))
    assert shifted.barrier_race(2.0, -2.0)[0] > 0.6


def test_negative_autocorrelation_widens_the_travelled_range():
    """Alternating steps reach further intra-path for the same endpoint spread.

    The endpoint marginal is fixed by the forecast, so steps that partly cancel
    must individually be larger, and the path travels further before returning.
    That matters for touch probabilities: ignoring the measured -0.21 would
    understate how often a barrier is reached.
    """
    quantiles = _quantiles(batch=1)
    kwargs = dict(samples=8000, generator=torch.Generator().manual_seed(4))
    bouncy = sample_paths(quantiles, LEVELS, autocorrelation=-0.21, **kwargs)
    independent = sample_paths(quantiles, LEVELS, autocorrelation=0.0, **kwargs)
    span = lambda f: float(f.running_max[0, -1, 4] - f.running_min[0, -1, 4])  # noqa: E731
    assert span(bouncy) > span(independent)
    assert bouncy.touch_probability(4.0)[0] > independent.touch_probability(4.0)[0]
    # Both still match the endpoint marginal they were given, within the
    # sampling error of a quantile estimate from this many paths.
    samples = bouncy.paths.shape[1]
    for forecast in (bouncy, independent):
        empirical = torch.quantile(forecast.paths[0, :, -1], torch.tensor(LEVELS))
        tolerance = 4 * float(forecast.paths[0, :, -1].std()) / samples**0.5
        assert (empirical - quantiles[0, -1]).abs().max() < tolerance


def test_correlation_matrix_is_valid_and_decays_with_distance():
    corr = minute_return_correlation(30, -0.21, torch.device("cpu"))
    assert corr.shape == (30, 30)
    torch.testing.assert_close(corr.diagonal(), torch.ones(30))
    torch.testing.assert_close(corr, corr.T)
    assert torch.linalg.eigvalsh(corr).min() > 0
    # Cumulative sums stay correlated, but less so further apart.
    assert corr[0, 1] > corr[0, 29]


def test_rejects_mismatched_quantile_axis():
    with pytest.raises(ValueError, match="quantiles"):
        sample_paths(torch.zeros(2, 30, 5), LEVELS)


def test_path_sampling_is_reproducible_with_a_seeded_generator():
    quantiles = _quantiles(batch=1)
    first = sample_paths(quantiles, LEVELS, samples=256, generator=torch.Generator().manual_seed(7))
    second = sample_paths(quantiles, LEVELS, samples=256, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(first.paths, second.paths)
    assert not np.isnan(first.paths.numpy()).any()
