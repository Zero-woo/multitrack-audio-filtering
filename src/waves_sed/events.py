"""Turn target probabilities into half-open events without mutating raw predictions."""

import math

import numpy as np

from waves_sed.evaluation_config import EventConfig
from waves_sed.prediction import FramePrediction


def _validated_target(prediction: FramePrediction, target_probability: np.ndarray) -> np.ndarray:
    prediction.validate()
    target = np.asarray(target_probability)
    if (
        target.shape != (len(prediction.probabilities),)
        or not np.issubdtype(target.dtype, np.number)
        or np.iscomplexobj(target)
        or not np.isfinite(target).all()
        or (target < 0).any()
        or (target > 1).any()
    ):
        raise ValueError("Target probabilities must be real, finite [0, 1] values, one per frame")
    return target.astype(np.float64, copy=True)


def _contiguous_runs(prediction: FramePrediction) -> list[tuple[int, int]]:
    start = np.asarray(prediction.frame_start_seconds)
    end = np.asarray(prediction.frame_end_seconds)
    boundaries = np.concatenate(([0], np.flatnonzero(start[1:] > end[:-1]) + 1, [len(start)]))
    return list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=True))


def smooth_target(
    prediction: FramePrediction,
    target_probability: np.ndarray,
    median_window_frames: int = 1,
) -> np.ndarray:
    """Apply an odd median window to probabilities, edge-padding each contiguous run.

    Frame counts determine the window, including a shorter final frame. Any positive
    temporal gap starts a new run; no interpolation or smoothing crosses missing time.
    A window of one returns an independent float64 copy of the raw target values.
    """
    EventConfig(median_window_frames=median_window_frames)
    target = _validated_target(prediction, target_probability)
    if median_window_frames == 1:
        return target
    half = median_window_frames // 2
    for first, stop in _contiguous_runs(prediction):
        padded = np.pad(target[first:stop], (half, half), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, median_window_frames)
        target[first:stop] = np.median(windows, axis=-1)
    return target


def extract_events(
    prediction: FramePrediction,
    target_probability: np.ndarray,
    config: EventConfig,
) -> tuple[tuple[float, float], ...]:
    """Smooth, activate at ``probability >= threshold``, then filter event duration.

    Events use actual half-open frame supports [start, end), including partial final
    bins. Adjacent active bins merge only when they touch. Minimum duration is inclusive;
    an absolute 1e-12 second allowance absorbs floating-point subtraction at the boundary.
    """
    if not isinstance(config, EventConfig):
        raise ValueError("config must be an EventConfig")
    config.validate()
    smoothed = smooth_target(prediction, target_probability, config.median_window_frames)
    active = smoothed >= config.threshold
    starts = np.asarray(prediction.frame_start_seconds)
    ends = np.asarray(prediction.frame_end_seconds)
    intervals = []
    for first, stop in _contiguous_runs(prediction):
        run = active[first:stop]
        transitions = np.diff(np.concatenate(([False], run, [False])).astype(np.int8))
        event_starts = np.flatnonzero(transitions == 1) + first
        event_stops = np.flatnonzero(transitions == -1) + first
        for onset_index, stop_index in zip(event_starts, event_stops, strict=True):
            onset, offset = float(starts[onset_index]), float(ends[stop_index - 1])
            duration = offset - onset
            if duration >= config.min_duration_seconds or math.isclose(
                duration, config.min_duration_seconds, rel_tol=0.0, abs_tol=1e-12
            ):
                intervals.append((onset, offset))
    return tuple(intervals)
