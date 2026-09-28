"""Sample-accurate controlled edits; no inference, audio decoder or heavy dependencies."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from waves_sed.corruption import CorruptionSpec, apply_corruption, load_corruption_specs


def mono(values, dtype=np.float64):
    return np.asarray(values, dtype=dtype)[:, None]


def apply(values, operation, **kwargs):
    return apply_corruption(mono(values), 10, CorruptionSpec("edit", operation, **kwargs))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_whole_clip_shift_preserves_channels_dtype_and_input(dtype):
    samples = np.array([[1, 0], [0, -2], [3, 4], [0, 0], [5, 6]], dtype=dtype)
    original = samples.copy()
    digest = hashlib.sha256(samples.tobytes()).hexdigest()
    samples.flags.writeable = False

    actual, report = apply_corruption(
        samples, 10, CorruptionSpec("shift", "shift", shift_seconds=0.2)
    )

    np.testing.assert_array_equal(actual, [[0, 0], [0, 0], [1, 0], [0, -2], [3, 4]])
    np.testing.assert_array_equal(samples, original)
    assert hashlib.sha256(samples.tobytes()).hexdigest() == digest
    assert actual.dtype == dtype
    assert not np.shares_memory(actual, samples)
    assert report["cropped_source_sample_frames"] == 2
    assert report["additive_overlap_sample_frames"] == 0
    assert report["realized_shift_seconds"] == 0.2
    assert report["audio_changed"]
    assert report["input_frames"] == report["output_frames"] == 5
    assert report["channels"] == 2
    assert report["peak_before"] == 6
    assert report["peak_after"] == 4
    json.dumps(report, allow_nan=False)


def test_negative_shift_crops_front_and_zero_pads_end():
    actual, report = apply([1, 2, 3, 4, 5], "shift", shift_seconds=-0.2)
    np.testing.assert_array_equal(actual[:, 0], [3, 4, 5, 0, 0])
    assert report["destination_start_sample"] == -2
    assert report["destination_end_sample"] == 3
    assert report["visible_destination_start_sample"] == 0
    assert report["visible_destination_end_sample"] == 3
    assert report["cropped_source_sample_frames"] == 2


@pytest.mark.parametrize("shift,expected", [(0.05, [0, 1, 2]), (-0.05, [2, 3, 0])])
def test_signed_half_sample_rounding_is_away_from_zero(shift, expected):
    actual, report = apply([1, 2, 3], "shift", shift_seconds=shift)
    np.testing.assert_array_equal(actual[:, 0], expected)
    assert report["shift_samples"] == (1 if shift > 0 else -1)
    assert report["requested_shift_seconds"] == shift
    assert abs(report["realized_shift_seconds"]) == 0.1


def test_decimal_rounding_does_not_use_binary_float_or_bankers_rounding():
    samples = mono(np.arange(60))
    actual, report = apply_corruption(
        samples, 100, CorruptionSpec("edit", "remove", (0.145, 0.285))
    )
    assert report["source_start_sample"] == 15
    assert report["source_end_sample"] == 29
    np.testing.assert_array_equal(actual[:15], samples[:15])
    np.testing.assert_array_equal(actual[15:29], 0)
    np.testing.assert_array_equal(actual[29:], samples[29:])


def test_local_shift_copies_source_before_clearing_and_adds_overlapping_audio():
    actual, report = apply([1, 2, 3, 4, 5, 6], "shift", interval=(0.1, 0.4), shift_seconds=0.2)
    np.testing.assert_array_equal(actual[:, 0], [1, 0, 0, 2, 8, 10])
    assert report["additive_overlap_sample_frames"] == 2
    assert report["affected_output_intervals_samples"] == [[1, 6]]
    assert not report["clipping_applied"]
    assert report["peak_after"] == 10


def test_local_negative_shift_preserves_unrelated_tail():
    actual, report = apply([1, 2, 3, 4, 5, 6], "shift", interval=(0.2, 0.5), shift_seconds=-0.3)
    np.testing.assert_array_equal(actual[:, 0], [5, 7, 0, 0, 0, 6])
    assert report["cropped_source_sample_frames"] == 1


def test_remove_only_changes_half_open_interval():
    actual, report = apply([1, 2, 3, 4, 5], "remove", interval=(0.1, 0.3))
    np.testing.assert_array_equal(actual[:, 0], [1, 0, 0, 4, 5])
    assert report["affected_output_intervals_samples"] == [[1, 3]]
    assert report["destination_start_sample"] is None
    assert report["realized_shift_seconds"] is None
    assert report["cropped_source_sample_frames"] == 0


def test_duplicate_is_additive_and_keeps_original():
    actual, report = apply(
        [1, 2, 3, 4, 5], "duplicate", interval=(0.1, 0.3), destination_seconds=0.2
    )
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 5, 7, 5])
    assert report["additive_overlap_sample_frames"] == 2
    assert report["affected_output_intervals_samples"] == [[2, 4]]


def test_duplicate_at_same_location_doubles_source():
    actual, _ = apply([0, 0.8, -0.9, 0], "duplicate", interval=(0.1, 0.3), destination_seconds=0.1)
    np.testing.assert_array_equal(actual[:, 0], [0, 1.6, -1.8, 0])


def test_duplicate_destination_rounding_and_crop():
    actual, report = apply(
        [1, 2, 3, 4, 5], "duplicate", interval=(0, 0.3), destination_seconds=0.35
    )
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3, 4, 6])
    assert report["destination_start_sample"] == 4
    assert report["destination_end_sample"] == 7
    assert report["cropped_source_sample_frames"] == 2
    assert report["requested_destination_seconds"] == 0.35
    assert report["realized_destination_seconds"] == 0.4


def test_overlap_counts_only_frames_with_same_channel_nonzero_audio():
    samples = np.array([[1, 0], [0, 2], [0, 3], [4, 0]], dtype=np.float32)
    actual, report = apply_corruption(
        samples, 10, CorruptionSpec("edit", "duplicate", (0, 0.2), destination_seconds=0.2)
    )
    np.testing.assert_array_equal(actual, [[1, 0], [0, 2], [1, 3], [4, 2]])
    assert report["additive_overlap_sample_frames"] == 0


def test_shorten_keeps_prefix_and_zeros_tail_without_moving_other_audio():
    actual, report = apply([1, 2, 3, 4, 5, 6], "shorten", interval=(0.1, 0.5), duration_seconds=0.2)
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3, 0, 0, 6])
    assert report["duration_samples"] == 2
    assert report["realized_duration_seconds"] == 0.2
    assert report["affected_output_intervals_samples"] == [[3, 5]]
    assert report["cropped_source_sample_frames"] == 0


def test_extend_repeats_source_and_adds_existing_audio_after_original_segment():
    actual, report = apply([1, 2, 3, 4, 5, 6], "extend", interval=(0.1, 0.3), duration_seconds=0.4)
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3, 6, 8, 6])
    assert report["additive_overlap_sample_frames"] == 2
    assert report["destination_start_sample"] == 1
    assert report["destination_end_sample"] == 5
    assert report["affected_output_intervals_samples"] == [[1, 5]]


def test_extend_crop_counts_repeated_frames():
    actual, report = apply([1, 2, 3, 4, 5], "extend", interval=(0.3, 0.5), duration_seconds=0.5)
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3, 4, 5])
    assert report["cropped_source_sample_frames"] == 3
    assert report["audio_changed"] is False


@pytest.mark.parametrize("shift", [1e300, -1e300])
def test_huge_shift_is_bounded_by_clip_length(shift):
    actual, report = apply([1, 2, 3], "shift", shift_seconds=shift)
    np.testing.assert_array_equal(actual, 0)
    assert report["cropped_source_sample_frames"] == 3
    assert report["realized_shift_seconds"] == shift
    json.dumps(report, allow_nan=False)


def test_huge_extension_does_not_allocate_requested_duration():
    actual, report = apply([1, 2, 3, 4, 5], "extend", interval=(0.1, 0.3), duration_seconds=1e300)
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3, 6, 8])
    assert report["cropped_source_sample_frames"] == report["duration_samples"] - 4
    json.dumps(report, allow_nan=False)


def test_huge_duplicate_destination_changes_nothing():
    actual, report = apply([1, 2, 3], "duplicate", interval=(0, 0.2), destination_seconds=1e300)
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3])
    assert report["cropped_source_sample_frames"] == 2
    assert report["audio_changed"] is False
    assert report["affected_output_intervals_samples"] == []


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("shift", {"shift_seconds": 0.01}),
        ("shorten", {"interval": (0.1, 0.3), "duration_seconds": 0.19}),
        ("extend", {"interval": (0.1, 0.3), "duration_seconds": 0.21}),
    ],
)
def test_subsample_request_can_record_no_signal_change(operation, kwargs):
    actual, report = apply([1, 2, 3, 4], operation, **kwargs)
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3, 4])
    assert report["audio_changed"] is False


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("shift", {"shift_seconds": 0.1}),
        ("remove", {"interval": (0, 0.2)}),
        ("duplicate", {"interval": (0, 0.2), "destination_seconds": 0.1}),
        ("shorten", {"interval": (0, 0.2), "duration_seconds": 0.1}),
        ("extend", {"interval": (0, 0.2), "duration_seconds": 0.3}),
    ],
)
def test_silent_source_has_no_audio_change(operation, kwargs):
    actual, report = apply([0, 0, 0, 0], operation, **kwargs)
    np.testing.assert_array_equal(actual, 0)
    assert report["audio_changed"] is False
    assert report["peak_before"] == report["peak_after"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": ""},
        {"id": " "},
        {"id": "control"},
        {"id": 3},
        {"operation": "unknown"},
        {"operation": []},
        {"shift_seconds": 0},
        {"shift_seconds": True},
        {"shift_seconds": "0.1"},
        {"shift_seconds": float("nan")},
        {"shift_seconds": float("inf")},
        {"shift_seconds": 10**400},
        {"duration_seconds": 0.1},
        {"interval": (0, 0)},
        {"interval": (-1, 1)},
        {"interval": (1, 0)},
        {"interval": (False, 1)},
        {"interval": (0, float("inf"))},
        {"interval": [0]},
        {"interval": "01"},
    ],
)
def test_invalid_shift_specs(kwargs):
    values = {"id": "x", "operation": "shift", "shift_seconds": 0.1} | kwargs
    with pytest.raises(ValueError):
        CorruptionSpec(**values)


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("remove", {}),
        ("duplicate", {}),
        ("shorten", {}),
        ("extend", {}),
        ("remove", {"interval": (0, 1), "shift_seconds": 1}),
        ("duplicate", {"interval": (0, 1)}),
        ("duplicate", {"interval": (0, 1), "destination_seconds": -0.1}),
        ("shorten", {"interval": (0, 1), "duration_seconds": 0}),
        ("shorten", {"interval": (0, 1), "duration_seconds": 1}),
        ("shorten", {"interval": (0, 1), "duration_seconds": 2}),
        ("extend", {"interval": (0, 1), "duration_seconds": 0.5}),
        ("extend", {"interval": (0, 1), "duration_seconds": 1}),
        ("extend", {"interval": (0, 1), "duration_seconds": 2, "destination_seconds": 0}),
    ],
)
def test_invalid_other_specs(operation, kwargs):
    with pytest.raises(ValueError):
        CorruptionSpec("x", operation, **kwargs)


@pytest.mark.parametrize("operation", ["shorten", "extend"])
@pytest.mark.parametrize("interval,duration", [((0.1, 0.3), 0.2), ((0.1, 0.4), 0.3)])
def test_equal_decimal_duration_is_neither_shorten_nor_extend(operation, interval, duration):
    with pytest.raises(ValueError, match="duration"):
        CorruptionSpec("x", operation, interval, duration_seconds=duration)


@pytest.mark.parametrize(
    "samples",
    [
        [1.0],
        np.ones(2),
        np.ones((2, 2, 1)),
        np.ones((0, 1)),
        np.ones((1, 0)),
        np.array([[1]]),
        np.array([[True]]),
        np.array([[1 + 2j]]),
        np.array([[float("nan")]]),
        np.array([[float("inf")]]),
    ],
)
def test_invalid_samples(samples):
    with pytest.raises(ValueError, match="samples"):
        apply_corruption(samples, 10, CorruptionSpec("x", "shift", shift_seconds=0.1))


@pytest.mark.parametrize("rate", [True, 0, -1, 1.5, "10", None])
def test_invalid_sample_rate(rate):
    with pytest.raises(ValueError, match="sample_rate"):
        apply_corruption(mono([1]), rate, CorruptionSpec("x", "shift", shift_seconds=0.1))


def test_numpy_integer_sample_rate_and_numeric_spec_are_json_serializable():
    spec = CorruptionSpec("x", "shift", shift_seconds=np.float32(0.2))
    _, report = apply_corruption(mono([1, 2, 3]), np.int64(10), spec)
    json.dumps(spec.to_dict(), allow_nan=False)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("interval", [(0, 1), (0.1, 0.401), (0, 0.01), (0.11, 0.12)])
def test_invalid_source_bounds_and_zero_quantized_length(interval):
    with pytest.raises(ValueError, match="interval"):
        apply([1, 2, 3, 4], "remove", interval=interval)


def test_float_duration_representation_at_boundary_is_accepted():
    _, report = apply_corruption(
        mono([1, 2, 3, 4, 5, 6, 7]), 3, CorruptionSpec("x", "remove", (0, 7 / 3))
    )
    assert report["source_end_sample"] == 7


def test_zero_quantized_duration_is_rejected():
    with pytest.raises(ValueError, match="at least one sample"):
        apply([1, 2, 3], "shorten", interval=(0, 0.2), duration_seconds=0.01)


def test_overflow_fails_without_mutating_input():
    samples = np.full((3, 2), np.finfo(np.float32).max, dtype=np.float32)
    before = samples.copy()
    with pytest.raises(ValueError, match="overflow"):
        apply_corruption(
            samples, 10, CorruptionSpec("x", "duplicate", (0, 0.2), destination_seconds=0)
        )
    np.testing.assert_array_equal(samples, before)


def test_to_dict_omits_inapplicable_fields_and_freezes_interval():
    interval = [0, 1]
    spec = CorruptionSpec("remove", "remove", interval)
    interval[0] = 0.5
    assert spec.interval == (0, 1)
    assert spec.to_dict() == {"id": "remove", "operation": "remove", "interval": [0.0, 1.0]}
    assert CorruptionSpec("shift", "shift", shift_seconds=-0.5).to_dict() == {
        "id": "shift",
        "operation": "shift",
        "shift_seconds": -0.5,
    }


def test_load_example_config_contains_prescribed_shifts_and_explicit_edits():
    path = Path(__file__).resolve().parents[1] / "configs" / "corruption.example.json"
    specs = load_corruption_specs(path)
    assert len(specs) == 8
    assert [s.shift_seconds for s in specs[:4]] == [0.1, 0.2, 0.5, 1]
    assert all(s.interval is None for s in specs[:4])
    assert {s.operation for s in specs[4:]} == {"remove", "duplicate", "shorten", "extend"}


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {"schema_version": 1, "variants": []},
        {
            "schema_version": True,
            "variants": [{"id": "x", "operation": "remove", "interval": [0, 1]}],
        },
        {"schema_version": 2, "variants": []},
        {"schema_version": 1, "variants": {}, "extra": 1},
        {"schema_version": 1, "variants": ["bad"]},
        {"schema_version": 1, "variants": [{"operation": "remove", "interval": [0, 1]}]},
        {
            "schema_version": 1,
            "variants": [{"id": "x", "operation": "remove", "interval": [0, 1], "extra": 1}],
        },
        {
            "schema_version": 1,
            "variants": [
                {"id": "x", "operation": "remove", "interval": [0, 1], "shift_seconds": None}
            ],
        },
        {
            "schema_version": 1,
            "variants": [{"id": "x", "operation": "shift", "shift_seconds": float("nan")}],
        },
        {
            "schema_version": 1,
            "variants": [{"id": "x", "operation": "shift", "shift_seconds": float("inf")}],
        },
    ],
)
def test_invalid_configs(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_corruption_specs(path)


@pytest.mark.parametrize(
    "contents",
    [
        '{"schema_version":1,"schema_version":1,"variants":[]}',
        '{"schema_version":1,"variants":[{"id":"x","id":"x","operation":"shift","shift_seconds":0.1}]}',
        '{"schema_version":1,"variants":[{"id":"x","operation":"shift","shift_seconds":0.1},{"id":"x","operation":"shift","shift_seconds":0.2}]}',
    ],
)
def test_duplicate_json_keys_and_variant_ids_rejected(tmp_path, contents):
    path = tmp_path / "config.json"
    path.write_text(contents, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Duplicate"):
        load_corruption_specs(path)


def test_config_roundtrip_with_utf8_bom(tmp_path):
    original = CorruptionSpec("한글 편집", "duplicate", (0.1, 0.3), destination_seconds=0.5)
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"schema_version": 1, "variants": [original.to_dict()]}), encoding="utf-8-sig"
    )
    assert load_corruption_specs(path) == (original,)
