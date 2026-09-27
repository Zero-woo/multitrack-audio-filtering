"""Extraction defaults are explicit, finite and never interpreted as quality gates."""

import json

import pytest

from waves_sed.evaluation_config import EvaluationConfig, EventConfig


def write_config(tmp_path, value):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_config_round_trip_and_all_settings(tmp_path):
    config = EvaluationConfig(
        event=EventConfig(threshold=0.7, median_window_frames=5, min_duration_seconds=0.04),
        onset_tolerance_seconds=0.1,
        allow_ambiguous_reference=True,
        outside_family_threshold=0.8,
        outside_family_top_k=7,
    )

    restored = EvaluationConfig.load(write_config(tmp_path, config.to_dict()))

    assert restored == config
    assert restored.to_dict()["schema_version"] == 1
    assert "decision" not in restored.to_dict()


def test_config_defaults_require_explicit_schema_but_allow_omitted_settings(tmp_path):
    assert (
        EvaluationConfig.load(write_config(tmp_path, {"schema_version": 1})) == EvaluationConfig()
    )
    partial = EvaluationConfig.load(
        write_config(tmp_path, {"schema_version": 1, "event": {"threshold": 0}})
    )
    assert partial.event == EventConfig(threshold=0)


def test_load_accepts_utf8_bom(tmp_path):
    path = tmp_path / "bom.json"
    path.write_text('{"schema_version":1}', encoding="utf-8-sig")
    assert EvaluationConfig.load(path) == EvaluationConfig()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, False, "0.5", None])
def test_invalid_event_threshold(value):
    with pytest.raises(ValueError, match="threshold"):
        EventConfig(threshold=value)


@pytest.mark.parametrize("value", [-1, 0, 2, 1.0, True, False, "3", None])
def test_invalid_median_window(value):
    with pytest.raises(ValueError, match="median_window_frames"):
        EventConfig(median_window_frames=value)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.1", None])
def test_invalid_min_duration(value):
    with pytest.raises(ValueError, match="min_duration_seconds"):
        EventConfig(min_duration_seconds=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event", {}),
        ("event", None),
        ("onset_tolerance_seconds", -1),
        ("onset_tolerance_seconds", float("inf")),
        ("onset_tolerance_seconds", True),
        ("allow_ambiguous_reference", 1),
        ("allow_ambiguous_reference", "false"),
        ("outside_family_threshold", -0.1),
        ("outside_family_threshold", 1.1),
        ("outside_family_threshold", float("nan")),
        ("outside_family_threshold", True),
        ("outside_family_top_k", 0),
        ("outside_family_top_k", -1),
        ("outside_family_top_k", True),
        ("outside_family_top_k", 3.0),
    ],
)
def test_invalid_evaluation_settings(field, value):
    with pytest.raises(ValueError, match=field):
        EvaluationConfig(**{field: value})


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        {},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"schema_version": 1, "threshold": 0.7},
        {"schema_version": 1, "event": None},
        {"schema_version": 1, "event": []},
        {"schema_version": 1, "event": {"unknown": 3}},
        {"schema_version": 1, "event": {"threshold": True}},
    ],
)
def test_load_rejects_unknown_fields_wrong_shapes_and_schema(tmp_path, value):
    with pytest.raises(ValueError):
        EvaluationConfig.load(write_config(tmp_path, value))


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"event":{"threshold":0.2,"threshold":0.3}}',
        '{"schema_version":1,"event":{"threshold":NaN}}',
        '{"schema_version":1,"onset_tolerance_seconds":Infinity}',
        '{"schema_version":1,"onset_tolerance_seconds":-Infinity}',
        '{"schema_version":1,"onset_tolerance_seconds":1e999}',
        '{"schema_version":1,"onset_tolerance_seconds":' + "9" * 400 + "}",
    ],
)
def test_load_rejects_duplicate_keys_and_nonfinite_json(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        EvaluationConfig.load(path)
