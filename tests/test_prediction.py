"""Validate cache integrity without interpreting synthetic scores as model output."""

from dataclasses import replace

import numpy as np
import pytest

from waves_sed.prediction import FramePrediction


@pytest.fixture
def prediction():
    # Deliberately synthetic values: these test serialization, not SED quality.
    return FramePrediction(
        probabilities=np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32),
        frame_start_seconds=np.array([0.0, 0.04], dtype=np.float64),
        frame_end_seconds=np.array([0.04, 0.06], dtype=np.float64),
        class_ids=("/m/test_a", "/m/test_b"),
        class_names=("가상 소리 A", "Synthetic sound B"),
        metadata={
            "purpose": "synthetic serialization test",
            "source_path": "음원/测试/작은 소리.wav",
            "provenance": {
                "checkpoint_sha256": "a" * 64,
                "revision": "test-fixture",
                "sample_rate": 16_000,
                "flags": [True, False, None],
            },
        },
    )


def test_prediction_accepts_valid_probabilities_and_short_final_frame(prediction):
    prediction.validate()


def test_prediction_round_trip_preserves_values_labels_and_provenance(tmp_path, prediction):
    path = tmp_path / "예측 결과.npz"
    prediction.save(path)
    restored = FramePrediction.load(path)

    restored.validate()
    np.testing.assert_array_equal(restored.probabilities, prediction.probabilities)
    np.testing.assert_array_equal(restored.frame_start_seconds, prediction.frame_start_seconds)
    np.testing.assert_array_equal(restored.frame_end_seconds, prediction.frame_end_seconds)
    assert restored.probabilities.dtype == np.dtype("float32")
    assert restored.frame_start_seconds.dtype == np.dtype("float64")
    assert restored.frame_end_seconds.dtype == np.dtype("float64")
    assert restored.class_ids == prediction.class_ids
    assert restored.class_names == prediction.class_names
    assert restored.metadata == prediction.metadata

    # Opening every array with pickle disabled detects accidental object storage.
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays.files
        assert all(not arrays[key].dtype.hasobject for key in arrays.files)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([0.2, 0.3], dtype=np.float32),
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((2, 0), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[np.inf, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[-0.01, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[1.01, 0.0], [0.0, 0.0]], dtype=np.float32),
    ],
)
def test_prediction_rejects_invalid_probabilities(prediction, probabilities):
    with pytest.raises(ValueError):
        replace(prediction, probabilities=probabilities).validate()


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("frame_start_seconds", [0.0]),
        ("frame_end_seconds", [0.04]),
        ("frame_start_seconds", [[0.0, 0.04]]),
        ("frame_end_seconds", [[0.04, 0.06]]),
        ("frame_start_seconds", [np.nan, 0.04]),
        ("frame_end_seconds", [0.04, np.inf]),
        ("frame_start_seconds", [-0.01, 0.04]),
        ("frame_end_seconds", [0.0, 0.06]),
        ("frame_end_seconds", [0.04, 0.03]),
        ("frame_start_seconds", [0.0, 0.0]),
        ("frame_start_seconds", [0.0, 0.02]),
    ],
)
def test_prediction_rejects_malformed_frame_times(prediction, field, values):
    invalid = replace(prediction, **{field: np.array(values, dtype=np.float64)})
    with pytest.raises(ValueError):
        invalid.validate()


@pytest.mark.parametrize(
    "updates",
    [
        {"class_ids": ("/m/test_a",)},
        {"class_names": ("Only one name",)},
        {"class_ids": ("/m/test_a", "/m/test_a")},
    ],
)
def test_prediction_rejects_inconsistent_class_metadata(prediction, updates):
    with pytest.raises(ValueError):
        replace(prediction, **updates).validate()


def test_invalid_prediction_is_rejected_before_saving(tmp_path, prediction):
    invalid = replace(
        prediction,
        probabilities=np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError):
        invalid.save(tmp_path / "invalid.npz")


@pytest.mark.parametrize(
    "metadata",
    [[], {"invalid": np.nan}, {"invalid": np.inf}, {"invalid": object()}],
)
def test_prediction_rejects_non_json_metadata(prediction, metadata):
    with pytest.raises(ValueError):
        replace(prediction, metadata=metadata).validate()


@pytest.mark.parametrize(
    ("field", "corrupted"),
    [
        ("schema_version", np.array(999, dtype=np.int64)),
        ("metadata_json", np.array("{not-json}")),
        ("metadata_json", np.array("[]")),
        ("metadata_json", np.array('{"bad": NaN}')),
        ("probabilities", np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32)),
        ("class_ids", np.array(["/m/test_a", "/m/test_a"])),
        ("class_names", np.array(["A", "B"], dtype=object)),
    ],
)
def test_loading_cache_rejects_corrupt_data(tmp_path, prediction, field, corrupted):
    path = tmp_path / "corrupt.npz"
    prediction.save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays[field] = corrupted
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError):
        FramePrediction.load(path)


def test_loading_cache_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, schema_version=np.array(1, dtype=np.int64))
    with pytest.raises((KeyError, ValueError)):
        FramePrediction.load(path)
