"""Synthetic scores exercise event boundaries without asserting model accuracy."""

import numpy as np
import pytest

from waves_sed.evaluation_config import EventConfig
from waves_sed.events import extract_events, smooth_target
from waves_sed.prediction import FramePrediction


def make_prediction(target, starts=None, ends=None):
    target = np.asarray(target, dtype=np.float64)
    boundaries = np.arange(len(target) + 1, dtype=np.float64) * 0.04
    return FramePrediction(
        probabilities=target[:, None],
        frame_start_seconds=boundaries[:-1] if starts is None else np.asarray(starts),
        frame_end_seconds=boundaries[1:] if ends is None else np.asarray(ends),
        class_ids=("synthetic",),
        class_names=("Synthetic score",),
        metadata={"purpose": "synthetic event extraction test"},
    )


def test_threshold_inclusive_contiguous_grouping_and_final_partial_frame():
    target = np.array([0.4, 0.5, 0.8, 0.1, 0.7])
    prediction = make_prediction(target, ends=[0.04, 0.08, 0.12, 0.16, 0.171])

    events = extract_events(prediction, target, EventConfig())

    assert events == ((0.04, 0.12), (0.16, 0.171))
    np.testing.assert_array_equal(prediction.probabilities[:, 0], target)


def test_events_split_at_any_positive_gap_even_if_all_frames_active():
    target = np.ones(4)
    prediction = make_prediction(
        target, starts=[0.0, 0.04, 0.09, 0.130000000001], ends=[0.04, 0.08, 0.13, 0.17]
    )

    assert extract_events(prediction, target, EventConfig()) == (
        (0.0, 0.08),
        (0.09, 0.13),
        (0.130000000001, 0.17),
    )


def test_median_suppresses_isolated_spike_and_fills_one_frame_dropout():
    target = np.array([0.0, 0.9, 0.0, 0.0, 0.8, 0.1, 0.7, 0.0, 0.0])
    prediction = make_prediction(target)

    smoothed = smooth_target(prediction, target, 3)

    np.testing.assert_allclose(smoothed, [0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.1, 0.0, 0.0])
    assert extract_events(prediction, target, EventConfig(median_window_frames=3)) == ((0.2, 0.24),)
    np.testing.assert_array_equal(prediction.probabilities[:, 0], target)


def test_median_uses_edge_padding_separately_for_each_contiguous_run():
    target = np.array([0.0, 1.0, 0.0, 0.0])
    prediction = make_prediction(
        target, starts=[0.0, 0.04, 0.1, 0.14], ends=[0.04, 0.08, 0.14, 0.18]
    )

    np.testing.assert_array_equal(smooth_target(prediction, target, 3), target)
    np.testing.assert_array_equal(smooth_target(prediction, target, 9), target)
    assert extract_events(prediction, target, EventConfig(median_window_frames=3)) == (
        (0.04, 0.08),
    )


def test_window_one_returns_independent_copy():
    target = np.array([0.2, 0.8], dtype=np.float32)
    prediction = make_prediction(target)
    result = smooth_target(prediction, target, 1)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, target)
    result[0] = 1.0
    assert target[0] == np.float32(0.2)


def test_minimum_duration_uses_real_support_and_is_inclusive():
    target = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    prediction = make_prediction(target, ends=[0.04, 0.08, 0.12, 0.16, 0.179])

    # The event at 0.08 has a binary subtraction result just below 0.04.
    events = extract_events(prediction, target, EventConfig(min_duration_seconds=0.04))
    assert events == ((0.0, 0.04), (0.08, 0.12))
    assert extract_events(prediction, target, EventConfig(min_duration_seconds=0.040001)) == ()


def test_minimum_duration_applies_after_smoothing_and_merging():
    target = np.array([0.0, 0.8, 0.8, 0.1, 0.8, 0.8, 0.0])
    prediction = make_prediction(target)
    assert extract_events(
        prediction, target, EventConfig(median_window_frames=3, min_duration_seconds=0.19)
    ) == ((0.04, 0.24),)


@pytest.mark.parametrize(
    ("target", "threshold", "expected"),
    [
        ([0.0, 0.0], 0.5, ()),
        ([1.0, 1.0], 1.0, ((0.0, 0.08),)),
        ([0.0, 0.0], 0.0, ((0.0, 0.08),)),
        ([0.7], 0.5, ((0.0, 0.04),)),
    ],
)
def test_empty_full_single_frame_and_threshold_extremes(target, threshold, expected):
    prediction = make_prediction(target)
    assert (
        extract_events(prediction, np.array(target), EventConfig(threshold=threshold)) == expected
    )


@pytest.mark.parametrize(
    "target",
    [
        np.array([0.5]),
        np.array([[0.5], [0.5]]),
        np.array([float("nan"), 0.5]),
        np.array([float("inf"), 0.5]),
        np.array([-0.1, 0.5]),
        np.array([1.1, 0.5]),
        np.array([0.2j, 0.5]),
        np.array([True, False]),
        np.array(["0.2", "0.5"]),
    ],
)
def test_invalid_target_probabilities_rejected(target):
    prediction = make_prediction([0.2, 0.5])
    with pytest.raises(ValueError, match="Target probabilities"):
        extract_events(prediction, target, EventConfig())


@pytest.mark.parametrize("window", [0, 2, -1, True, 3.0])
def test_smooth_target_validates_window(window):
    prediction = make_prediction([0.2, 0.5])
    with pytest.raises(ValueError, match="median_window_frames"):
        smooth_target(prediction, np.array([0.2, 0.5]), window)


def test_invalid_cache_frame_support_rejected_before_extraction():
    prediction = make_prediction([0.2, 0.5], starts=[0.0, 0.03], ends=[0.04, 0.08])
    with pytest.raises(ValueError, match="Frame supports"):
        extract_events(prediction, np.array([0.2, 0.5]), EventConfig())
