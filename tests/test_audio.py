"""Offline checks for the 40 ms frame grid, including unpadded tails."""

import numpy as np
import pytest

from waves_sed.audio import frame_grid


@pytest.mark.parametrize(
    ("num_samples", "expected_frames", "last_start", "last_end"),
    [
        (160_000, 250, 9.96, 10.0),
        (160_160, 251, 10.0, 10.01),
        (1, 1, 0.0, 1.0 / 16_000),
        (320, 1, 0.0, 0.02),
        (640, 1, 0.0, 0.04),
        (641, 2, 0.04, 641.0 / 16_000),
    ],
)
def test_frame_grid_retains_audio_and_excludes_padding(
    num_samples, expected_frames, last_start, last_end
):
    starts, ends = frame_grid(num_samples)

    assert starts.shape == ends.shape == (expected_frames,)
    assert starts.dtype == ends.dtype == np.dtype("float64")
    assert starts[0] == 0.0
    assert starts[-1] == pytest.approx(last_start)
    assert ends[-1] == pytest.approx(last_end)
    assert np.all(ends > starts)
    np.testing.assert_allclose(ends[:-1], starts[1:], atol=1e-12)
    np.testing.assert_allclose(ends[:-1] - starts[:-1], 0.04, atol=1e-12)
    assert float(np.sum(ends - starts)) == pytest.approx(num_samples / 16_000)


def test_frame_grid_uses_explicit_sample_rate_and_frame_size():
    starts, ends = frame_grid(1_500, sample_rate=1_000, frame_samples=400)

    np.testing.assert_allclose(starts, [0.0, 0.4, 0.8, 1.2])
    np.testing.assert_allclose(ends, [0.4, 0.8, 1.2, 1.5])


@pytest.mark.parametrize("num_samples", [0, -1])
def test_frame_grid_rejects_empty_or_negative_length(num_samples):
    with pytest.raises(ValueError):
        frame_grid(num_samples)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": -16_000},
        {"frame_samples": 0},
        {"frame_samples": -640},
    ],
)
def test_frame_grid_rejects_invalid_grid_parameters(kwargs):
    with pytest.raises(ValueError):
        frame_grid(160_000, **kwargs)
