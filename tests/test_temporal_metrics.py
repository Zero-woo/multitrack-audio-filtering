"""Small, independently checkable temporal measurement examples."""

import itertools
import math

import numpy as np
import pytest

from waves_sed.temporal_metrics import (
    ambience_metrics,
    intersection_duration,
    interval_duration,
    onset_metrics,
    span_metrics,
    union_intervals,
    validate_intervals,
)


def events(*starts):
    return [(start, start + 0.01) for start in starts]


def test_validation_preserves_unsorted_overlapping_event_identities():
    intervals = [(3, 4), (0, 2), (1, 3), (1, 3)]
    assert validate_intervals(intervals) == tuple(intervals)
    assert union_intervals(intervals) == ((0, 4),)
    assert interval_duration(intervals) == 4


@pytest.mark.parametrize(
    "intervals",
    [
        None,
        "",
        {},
        [{0: "start", 1: "end"}],
        [1],
        [(0,)],
        [(0, 1, 2)],
        [(0, 0)],
        [(1, 0)],
        [(-1, 1)],
        [(False, 1)],
        [(0, True)],
        [("0", 1)],
        [(0, float("inf"))],
        [(float("nan"), 1)],
        [(0, 1j)],
        [(0, 10**400)],
    ],
)
def test_invalid_intervals_raise_value_error(intervals):
    with pytest.raises(ValueError):
        validate_intervals(intervals)


def test_union_and_intersection_use_half_open_sets_without_double_counting():
    left = [(4, 5), (0, 1), (1, 3), (2, 4)]
    right = [(4, 6), (1, 2), (1.5, 2.5)]
    assert union_intervals(left) == ((0, 5),)
    assert union_intervals(right) == ((1, 2.5), (4, 6))
    assert intersection_duration(left, right) == 2.5
    assert intersection_duration([(0, 1)], [(1, 2)]) == 0
    assert interval_duration([]) == 0
    assert intersection_duration([], right) == 0


def test_onset_assignment_avoids_nearest_first_greedy_failure():
    result = onset_metrics(events(1, 2), events(0.4, 1.1), 1)
    assert result["matched_event_count"] == 2
    assert result["missing_event_count"] == result["extra_event_count"] == 0
    assert result["event_recall"] == result["event_precision"] == 1
    assert result["mean_onset_error_ms"] == pytest.approx(750)
    assert result["median_onset_error_ms"] == pytest.approx(750)
    assert [(m["expected_index"], m["detected_index"]) for m in result["matches"]] == [
        (0, 0),
        (1, 1),
    ]


def test_match_cardinality_has_priority_over_a_lower_total_error():
    result = onset_metrics(events(1, 2), events(1, 1.01), 1)
    assert result["matched_event_count"] == 2
    assert result["mean_onset_error_ms"] == pytest.approx(495)


def test_equal_cardinality_chooses_minimum_error_and_preserves_original_indices():
    result = onset_metrics(events(3, 1), events(3.2, 0.8, 1.1), 0.5)
    assert result["matched_event_count"] == 2
    assert result["extra_event_count"] == 1
    assert result["mean_onset_error_ms"] == pytest.approx(150)
    assert result["median_onset_error_ms"] == pytest.approx(150)
    assert [(m["expected_index"], m["detected_index"]) for m in result["matches"]] == [
        (1, 2),
        (0, 0),
    ]
    assert [m["signed_onset_error_ms"] for m in result["matches"]] == pytest.approx([100, 200])


def test_onset_tolerance_is_inclusive_and_errors_are_signed_and_absolute():
    result = onset_metrics(events(0.3), events(0.4), 0.1)
    assert result["matched_event_count"] == 1
    early = onset_metrics(events(2), events(1.75), 0.25)["matches"][0]
    assert early["signed_onset_error_ms"] == -250
    assert early["absolute_onset_error_ms"] == 250
    assert onset_metrics(events(2), events(1.749), 0.25)["matched_event_count"] == 0
    assert onset_metrics(events(1), events(1 + 1e-13), 0)["matched_event_count"] == 0
    assert onset_metrics(events(1), events(1), 0)["matched_event_count"] == 1


def test_onset_events_with_equal_starts_keep_separate_one_to_one_identities():
    result = onset_metrics(events(1, 1, 1), events(1, 1), 0)
    assert result["expected_event_count"] == 3
    assert result["matched_event_count"] == 2
    assert result["missing_event_count"] == 1
    assert result["mean_onset_error_ms"] == 0
    assert len({match["expected_index"] for match in result["matches"]}) == 2
    assert len({match["detected_index"] for match in result["matches"]}) == 2


@pytest.mark.parametrize("tolerance", [-1, float("inf"), float("nan"), True, "1", 1j, 10**400])
def test_invalid_onset_tolerance(tolerance):
    with pytest.raises(ValueError):
        onset_metrics([], [], tolerance)


@pytest.mark.parametrize(
    "expected,detected,recall,precision,missing,extra",
    [
        ([], [], None, None, 0, 0),
        (events(1), [], 0, None, 1, 0),
        ([], events(1), None, 0, 0, 1),
        (events(1), events(2), 0, 0, 1, 1),
    ],
)
def test_unmatched_onset_cases_do_not_invent_error_measurements(
    expected, detected, recall, precision, missing, extra
):
    result = onset_metrics(expected, detected, 0)
    assert result["event_recall"] == recall
    assert result["event_precision"] == precision
    assert result["missing_event_count"] == missing
    assert result["extra_event_count"] == extra
    assert result["mean_onset_error_ms"] is None
    assert result["median_onset_error_ms"] is None
    assert result["matches"] == []


def exhaustive_objective(expected, detected, tolerance):
    # Deliberately permit crossing assignments in this independent small oracle.
    best = (0, 0.0)
    for count in range(1, min(len(expected), len(detected)) + 1):
        for expected_indices in itertools.combinations(range(len(expected)), count):
            for detected_indices in itertools.permutations(range(len(detected)), count):
                errors = [
                    abs(expected[i] - detected[j])
                    for i, j in zip(expected_indices, detected_indices)
                ]
                if all(error <= tolerance for error in errors):
                    candidate = (-count, math.fsum(errors))
                    best = min(best, candidate)
    return -best[0], best[1]


def test_ordered_dynamic_program_matches_exhaustive_assignment_oracle():
    rng = np.random.default_rng(913)
    for _ in range(120):
        expected = rng.integers(0, 24, size=int(rng.integers(0, 5))) / 8
        detected = rng.integers(0, 24, size=int(rng.integers(0, 5))) / 8
        tolerance = float(rng.integers(0, 8)) / 8
        count, cost = exhaustive_objective(expected, detected, tolerance)
        result = onset_metrics(events(*expected), events(*detected), tolerance)
        assert result["matched_event_count"] == count
        assert sum(m["absolute_onset_error_ms"] for m in result["matches"]) == pytest.approx(
            cost * 1000
        )
        assert len({m["expected_index"] for m in result["matches"]}) == count
        assert len({m["detected_index"] for m in result["matches"]}) == count


def test_span_metrics_measure_unions_and_signed_outer_boundaries():
    result = span_metrics([(1, 3), (2, 4), (6, 8)], [(0, 2), (3, 7), (6, 9)])
    assert result["expected_duration_seconds"] == 5
    assert result["detected_duration_seconds"] == 8
    assert result["intersection_duration_seconds"] == 4
    assert result["union_duration_seconds"] == 9
    assert result["temporal_iou"] == pytest.approx(4 / 9)
    assert result["expected_coverage"] == pytest.approx(4 / 5)
    assert result["detected_precision"] == 0.5
    assert result["onset_error"] == -1
    assert result["offset_error"] == 1
    assert result["out_of_window_activation"] == 4


@pytest.mark.parametrize(
    "expected,detected,iou,coverage,precision,outside",
    [([], [], None, None, None, 0), ([(1, 2)], [], 0, 0, None, 0), ([], [(1, 2)], 0, None, 0, 1)],
)
def test_empty_span_supports(expected, detected, iou, coverage, precision, outside):
    result = span_metrics(expected, detected)
    assert result["temporal_iou"] == iou
    assert result["expected_coverage"] == coverage
    assert result["detected_precision"] == precision
    assert result["out_of_window_activation"] == outside
    assert result["onset_error"] is None
    assert result["offset_error"] is None


def test_ambience_confidence_weights_raw_scores_by_partial_frame_overlap():
    # Expected union is [0.25, 2.5): weighted integral = .75*.2 + 1*.8 + .5*.4.
    result = ambience_metrics(
        [(0.25, 1.5), (1, 2.5)],
        [(0, 0.5), (1, 2), (2.25, 3)],
        [0, 1, 2],
        [1, 2, 3],
        [0.2, 0.8, 0.4],
    )
    assert result["expected_duration_seconds"] == 2.25
    assert result["observed_expected_duration_seconds"] == 2.25
    assert result["detected_duration_seconds"] == 2.25
    assert result["intersection_duration_seconds"] == 1.5
    assert result["occupancy_in_expected_span"] == pytest.approx(1.5 / 2.25)
    assert result["mean_target_confidence"] == pytest.approx(1.15 / 2.25)
    assert result["out_of_window_activity"] == 0.75


def test_ambience_disconnected_reference_support_can_avoid_observation_gaps():
    result = ambience_metrics([(0, 1), (2.5, 3)], [], [0, 2], [1, 3], [0.2, 0.8])
    assert result["mean_target_confidence"] == pytest.approx(0.6 / 1.5)
    assert result["occupancy_in_expected_span"] == 0


@pytest.mark.parametrize(
    "expected,starts,ends,scores",
    [
        ([(0, 3)], [0, 2], [1, 3], [0.2, 0.8]),
        ([(0, 3)], [0], [2], [0.2]),
        ([(0, 3)], [1], [3], [0.2]),
        ([(0, 1)], [], [], []),
        ([(0, 1e-13)], [], [], []),
        ([(0, 1)], [0, 0.5 + 1e-13], [0.5, 1], [0.2, 0.8]),
    ],
)
def test_ambience_rejects_unobserved_reference_support(expected, starts, ends, scores):
    with pytest.raises(ValueError, match="not fully observed"):
        ambience_metrics(expected, [], starts, ends, scores)


@pytest.mark.parametrize(
    "starts,ends,scores",
    [
        ([0, 1], [1], [0.5]),
        ([[0]], [[1]], [[0.5]]),
        ([0], [1], [float("nan")]),
        ([0], [1], [1.1]),
        ([0], [1], [-0.1]),
        ([0], [1], [True]),
        ([-1], [0], [0.5]),
        ([1], [1], [0.5]),
        ([0, 0.5], [1, 2], [0.5, 0.5]),
        ([1, 0], [2, 1], [0.5, 0.5]),
        ([0], [float("inf")], [0.5]),
        ([0], [1], [0.5j]),
        (["0"], [1], [0.5]),
    ],
)
def test_ambience_validates_frame_support_and_probability(starts, ends, scores):
    with pytest.raises(ValueError):
        ambience_metrics([], [], starts, ends, scores)


def test_empty_ambience_reference_is_explicitly_undefined():
    result = ambience_metrics([], [(0, 1)], [0], [1], [0.9])
    assert result["occupancy_in_expected_span"] is None
    assert result["mean_target_confidence"] is None
    assert result["out_of_window_activity"] == 1
    assert result["expected_duration_seconds"] == 0
    empty = ambience_metrics([], [], [], [], [])
    assert empty["mean_target_confidence"] is None
    assert empty["out_of_window_activity"] == 0
