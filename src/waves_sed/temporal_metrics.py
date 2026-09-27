"""Role-specific measurements against explicit temporal references.

All intervals are half-open ``[start, end)`` in seconds. Duration measurements
use interval unions, so overlaps never count twice. Event matching keeps each
input interval as a separate event and uses its start time. Empty denominators
and unavailable boundary errors produce ``None`` rather than invented zeros.
These measurements do not assign quality decisions or assert that a reference
is ground truth.
"""

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np


def validate_intervals(intervals) -> tuple[tuple[float, float], ...]:
    """Validate finite, nonnegative, positive-duration intervals, keeping order."""
    if isinstance(intervals, (str, bytes, Mapping)):
        raise ValueError("Intervals must be an iterable of start/end pairs")
    try:
        values = tuple(intervals)
    except TypeError as error:
        raise ValueError("Intervals must be an iterable of start/end pairs") from error
    result = []
    for interval in values:
        if isinstance(interval, (str, bytes, Mapping)):
            raise ValueError("Each interval must contain exactly two real numbers")
        try:
            start, end = interval
        except (TypeError, ValueError) as error:
            raise ValueError("Each interval must contain exactly two real numbers") from error
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in (start, end)
        ):
            raise ValueError("Interval boundaries must be real numbers")
        try:
            start, end = float(start), float(end)
        except OverflowError as error:
            raise ValueError("Interval boundaries must be finite") from error
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Intervals must have finite boundaries with 0 <= start < end")
        result.append((start, end))
    return tuple(result)


def union_intervals(intervals) -> tuple[tuple[float, float], ...]:
    """Return sorted, disjoint intervals, merging both overlaps and touching bins."""
    merged = []
    for start, end in sorted(validate_intervals(intervals)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def interval_duration(intervals) -> float:
    """Measure the union duration in seconds (zero for an empty set)."""
    return math.fsum(end - start for start, end in union_intervals(intervals))


def intersection_duration(left, right) -> float:
    """Measure intersecting union supports without double counting overlaps."""
    left, right = union_intervals(left), union_intervals(right)
    i = j = 0
    overlaps = []
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            overlaps.append(end - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return math.fsum(overlaps)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def onset_metrics(expected, detected, tolerance_seconds: float) -> dict:
    """Maximize match count, then minimize total absolute onset error.

    Ordered dynamic programming yields an optimal one-to-one assignment for
    absolute onset distance on a line. Sorting is internal; returned indices
    refer to the caller's original interval ordering. Durations do not affect
    onset matching. Tolerance is inclusive; a relative 1e-12 comparison absorbs
    floating-point rounding at a positive tolerance boundary (zero is exact).
    Mean and median errors are absolute milliseconds, while each match also
    reports the signed error ``detected - expected``.
    """
    expected, detected = validate_intervals(expected), validate_intervals(detected)
    if (
        isinstance(tolerance_seconds, (bool, np.bool_))
        or not isinstance(tolerance_seconds, Real)
        or tolerance_seconds < 0
    ):
        raise ValueError("Onset tolerance must be a finite nonnegative real number")
    try:
        tolerance_seconds = float(tolerance_seconds)
    except OverflowError as error:
        raise ValueError("Onset tolerance must be finite") from error
    if not math.isfinite(tolerance_seconds):
        raise ValueError("Onset tolerance must be finite")
    expected_order = sorted(range(len(expected)), key=lambda index: expected[index][0])
    detected_order = sorted(range(len(detected)), key=lambda index: detected[index][0])
    shape = (len(expected) + 1, len(detected) + 1)
    counts = np.zeros(shape, dtype=np.int64)
    errors = np.zeros(shape, dtype=np.float64)
    actions = np.zeros(shape, dtype=np.uint8)
    for i, expected_index in enumerate(expected_order, start=1):
        for j, detected_index in enumerate(detected_order, start=1):
            # Skipping an expected event wins exact ties between skip actions.
            best_count, best_error, action = counts[i - 1, j], errors[i - 1, j], 1
            if counts[i, j - 1] > best_count or (
                counts[i, j - 1] == best_count and errors[i, j - 1] < best_error
            ):
                best_count, best_error, action = counts[i, j - 1], errors[i, j - 1], 2
            error = abs(detected[detected_index][0] - expected[expected_index][0])
            if error <= tolerance_seconds or math.isclose(
                error, tolerance_seconds, rel_tol=1e-12, abs_tol=0.0
            ):
                matched_count = counts[i - 1, j - 1] + 1
                matched_error = errors[i - 1, j - 1] + error
                if matched_count > best_count or (
                    matched_count == best_count and matched_error <= best_error
                ):
                    best_count, best_error, action = matched_count, matched_error, 3
            counts[i, j], errors[i, j], actions[i, j] = best_count, best_error, action

    matches = []
    i, j = len(expected), len(detected)
    while i and j:
        action = actions[i, j]
        if action == 3:
            expected_index, detected_index = expected_order[i - 1], detected_order[j - 1]
            signed_error = (detected[detected_index][0] - expected[expected_index][0]) * 1000
            matches.append(
                {
                    "expected_index": expected_index,
                    "detected_index": detected_index,
                    "signed_onset_error_ms": signed_error,
                    "absolute_onset_error_ms": abs(signed_error),
                }
            )
            i, j = i - 1, j - 1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    absolute_errors = [match["absolute_onset_error_ms"] for match in matches]
    return {
        "expected_event_count": len(expected),
        "detected_event_count": len(detected),
        "matched_event_count": len(matches),
        "missing_event_count": len(expected) - len(matches),
        "extra_event_count": len(detected) - len(matches),
        "event_recall": _ratio(len(matches), len(expected)),
        "event_precision": _ratio(len(matches), len(detected)),
        "mean_onset_error_ms": (math.fsum(absolute_errors) / len(matches) if matches else None),
        "median_onset_error_ms": float(np.median(absolute_errors)) if matches else None,
        "matches": matches,
    }


def span_metrics(expected, detected) -> dict:
    """Compare support unions, using signed outer-boundary errors in seconds.

    ``onset_error`` is the first detected start minus the first expected start;
    ``offset_error`` is the last detected end minus the last expected end.
    They describe the outer envelope, not individual interval matching.
    ``out_of_window_activation`` is detected duration outside expected support,
    in seconds. Empty supports have zero duration and undefined ratios where
    their denominator is zero; boundary errors require both supports.
    """
    expected, detected = union_intervals(expected), union_intervals(detected)
    expected_duration = interval_duration(expected)
    detected_duration = interval_duration(detected)
    intersection = intersection_duration(expected, detected)
    combined_duration = interval_duration((*expected, *detected))
    return {
        "temporal_iou": _ratio(intersection, combined_duration),
        "expected_coverage": _ratio(intersection, expected_duration),
        "detected_precision": _ratio(intersection, detected_duration),
        "onset_error": detected[0][0] - expected[0][0] if expected and detected else None,
        "offset_error": detected[-1][1] - expected[-1][1] if expected and detected else None,
        "out_of_window_activation": max(0.0, detected_duration - intersection),
        "expected_duration_seconds": expected_duration,
        "detected_duration_seconds": detected_duration,
        "intersection_duration_seconds": intersection,
        "union_duration_seconds": combined_duration,
    }


def _validate_frames(frame_start_seconds, frame_end_seconds, target_probability):
    arrays = tuple(
        np.asarray(values)
        for values in (frame_start_seconds, frame_end_seconds, target_probability)
    )
    if any(values.ndim != 1 for values in arrays) or len({v.shape for v in arrays}) != 1:
        raise ValueError("Frame boundaries and probabilities must be matching 1D arrays")
    if any(
        not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.isfinite(values).all()
        for values in arrays
    ):
        raise ValueError("Frame boundaries and probabilities must be finite real numbers")
    starts, ends, scores = (values.astype(np.float64, copy=False) for values in arrays)
    if (starts < 0).any() or (ends <= starts).any() or (starts[1:] < ends[:-1]).any():
        raise ValueError("Frame supports must be nonnegative, positive, ordered and nonoverlapping")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("Target probabilities must be in [0, 1]")
    return starts, ends, scores


def ambience_metrics(
    expected, detected, frame_start_seconds, frame_end_seconds, target_probability
) -> dict:
    """Measure occupancy and raw confidence over the expected support union.

    Confidence is the integral of raw target probability divided by expected
    duration. Partial frame overlap contributes its actual duration, and
    overlapping reference intervals count only once. Missing observations are
    rejected, including gaps inside expected support. Each reference interval
    must be fully contained in a contiguous observed block. Empty support gives
    ``None`` for occupancy and confidence. ``out_of_window_activity`` is seconds
    of detected activity outside expected support, without an onset constraint.
    """
    expected, detected = union_intervals(expected), union_intervals(detected)
    starts, ends, scores = _validate_frames(
        frame_start_seconds, frame_end_seconds, target_probability
    )
    expected_duration = interval_duration(expected)
    detected_duration = interval_duration(detected)
    intersection = intersection_duration(expected, detected)
    observed = union_intervals(zip(starts, ends))
    observed_expected_duration = intersection_duration(expected, observed)
    observed_index = 0
    for start, end in expected:
        while observed_index < len(observed) and observed[observed_index][1] <= start:
            observed_index += 1
        if (
            observed_index == len(observed)
            or observed[observed_index][0] > start
            or observed[observed_index][1] < end
        ):
            raise ValueError("Expected support is not fully observed in the prediction frame bins")
    contributions = []
    i = j = 0
    while i < len(expected) and j < len(starts):
        overlap = min(expected[i][1], ends[j]) - max(expected[i][0], starts[j])
        if overlap > 0:
            contributions.append(float(overlap * scores[j]))
        if expected[i][1] <= ends[j]:
            i += 1
        else:
            j += 1
    return {
        "occupancy_in_expected_span": _ratio(intersection, expected_duration),
        "mean_target_confidence": _ratio(math.fsum(contributions), expected_duration),
        "out_of_window_activity": max(0.0, detected_duration - intersection),
        "expected_duration_seconds": expected_duration,
        "detected_duration_seconds": detected_duration,
        "intersection_duration_seconds": intersection,
        "observed_expected_duration_seconds": observed_expected_duration,
    }
