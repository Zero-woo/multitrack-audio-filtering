"""Manual semantic mappings must stay explicit and preserve backend ID alignment."""

import json

import numpy as np
import pytest

from waves_sed.labels import load_labels
from waves_sed.mapping import ManualMapping, aggregate_target
from waves_sed.ontology import AudioSetOntology
from waves_sed.prediction import FramePrediction


@pytest.fixture
def ontology(tmp_path):
    # Shared has two parents: traversal must deduplicate the diamond.
    children = {
        "root": ["left", "right"],
        "left": ["shared"],
        "right": ["shared"],
        "shared": [],
        "unrelated": [],
    }
    path = tmp_path / "ontology.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": key,
                    "name": key.title(),
                    "description": "Synthetic test node",
                    "child_ids": values,
                    "restrictions": [],
                }
                for key, values in children.items()
            ]
        ),
        encoding="utf-8",
    )
    return AudioSetOntology.load(path)


def write_config(tmp_path, mappings, **overrides):
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps({"schema_version": 1, "mappings": mappings, **overrides}),
        encoding="utf-8",
    )
    return path


def rule(**overrides):
    return {"descriptions": ["Steady rain"], "allowed_classes": ["root"], **overrides}


def test_description_matching_normalizes_only_case_and_whitespace(tmp_path, ontology):
    mapping = ManualMapping.load(write_config(tmp_path, {"rain": rule()}), ontology)

    match = mapping.resolve("  STEADY\n\t Rain  ", ("root",))
    assert match.status == "supported"
    assert match.mapping_key == "rain"
    assert match.target_class_ids == ("root",)

    for description in ["rain", "Steady rainfall", "Steady rain in the forest", "Steady-rain"]:
        unsupported = mapping.resolve(description, ("root",))
        assert unsupported.status == "unsupported_mapping"
        assert unsupported.reason == "no_explicit_mapping"
        assert unsupported.target_class_ids == ()


@pytest.mark.parametrize("description", [None, "", " \t\n"])
def test_missing_description_has_explicit_unsupported_reason(tmp_path, ontology, description):
    mapping = ManualMapping.load(write_config(tmp_path, {"rain": rule()}), ontology)

    resolution = mapping.resolve(description, ("root",))

    assert resolution.status == "unsupported_mapping"
    assert resolution.reason == "missing_source_description"
    assert resolution.mapping_key is None
    assert resolution.allowed_class_ids == ()
    assert resolution.target_class_ids == ()


@pytest.mark.parametrize(
    "updates",
    [
        {"descriptions": []},
        {"descriptions": [""]},
        {"descriptions": ["  "]},
        {"descriptions": [None]},
        {"descriptions": "Steady rain"},
        {"descriptions": ["Steady rain", "  STEADY rain  "]},
        {"allowed_classes": []},
        {"allowed_classes": ["root", "root"]},
        {"allowed_classes": ["unknown"]},
        {"foreign_classes": ["unknown"]},
        {"foreign_classes": ["unrelated", "unrelated"]},
        {"allowed_class": ["root"]},
        {"include_descendants": "true"},
        {"notes": ["not a string"]},
    ],
)
def test_invalid_or_ambiguous_rules_are_rejected(tmp_path, ontology, updates):
    path = write_config(tmp_path, {"rain": rule(**updates)})
    with pytest.raises(ValueError):
        ManualMapping.load(path, ontology)


def test_normalized_alias_cannot_belong_to_two_rules(tmp_path, ontology):
    path = write_config(
        tmp_path,
        {
            "rain": rule(),
            "other": rule(descriptions=["  steady RAIN"], allowed_classes=["unrelated"]),
        },
    )
    with pytest.raises(ValueError, match="Duplicate|ambiguous"):
        ManualMapping.load(path, ontology)


@pytest.mark.parametrize(
    "overrides", [{"schema_version": True}, {"schema_version": 2}, {"unexpected": "field"}]
)
def test_invalid_config_schema_is_rejected(tmp_path, ontology, overrides):
    with pytest.raises(ValueError):
        ManualMapping.load(write_config(tmp_path, {"rain": rule()}, **overrides), ontology)


def test_descendant_expansion_requires_explicit_opt_in(tmp_path, ontology):
    path = write_config(
        tmp_path,
        {
            "exact": rule(descriptions=["Exact parent"]),
            "family": rule(descriptions=["Family"], include_descendants=True),
        },
    )
    mapping = ManualMapping.load(path, ontology)
    backend_ids = ("unrelated", "shared", "left", "right")

    exact = mapping.resolve("Exact parent", backend_ids)
    assert exact.status == "unsupported_mapping"
    assert exact.reason == "no_allowed_classes_in_backend"
    assert exact.allowed_class_ids == ("root",)
    assert exact.unavailable_class_ids == ("root",)

    expanded = mapping.resolve("Family", backend_ids)
    assert expanded.status == "supported"
    assert expanded.allowed_class_ids == ("left", "right", "root", "shared")
    assert expanded.target_class_ids == ("shared", "left", "right")
    assert expanded.unavailable_class_ids == ("root",)


@pytest.mark.parametrize("include_descendants, foreign", [(False, "root"), (True, "shared")])
def test_foreign_classes_cannot_overlap_allowed_family(
    tmp_path, ontology, include_descendants, foreign
):
    path = write_config(
        tmp_path, {"rain": rule(include_descendants=include_descendants, foreign_classes=[foreign])}
    )
    with pytest.raises(ValueError, match="overlap"):
        ManualMapping.load(path, ontology)


def test_missing_backend_classes_are_reported_without_discarding_supported_targets(
    tmp_path, ontology
):
    mapping = ManualMapping.load(
        write_config(
            tmp_path,
            {"rain": rule(allowed_classes=["left", "right"], foreign_classes=["unrelated"])},
        ),
        ontology,
    )

    partial = mapping.resolve("Steady rain", ("left", "unrelated"))
    assert partial.status == "supported"
    assert partial.target_class_ids == ("left",)
    assert partial.unavailable_class_ids == ("right",)
    assert partial.foreign_class_ids == ("unrelated",)
    assert partial.to_dict()["target_class_ids"] == ["left"]
    assert partial.to_dict()["unavailable_class_ids"] == ["right"]

    absent = mapping.resolve("Steady rain", ("unrelated",))
    assert absent.status == "unsupported_mapping"
    assert absent.mapping_key == "rain"
    assert absent.reason == "no_allowed_classes_in_backend"
    assert absent.unavailable_class_ids == ("left", "right")


def test_duplicate_backend_class_ids_are_rejected(tmp_path, ontology):
    mapping = ManualMapping.load(write_config(tmp_path, {"rain": rule()}), ontology)
    with pytest.raises(ValueError):
        mapping.resolve("Steady rain", ("root", "root"))


def test_unrelated_new_backend_classes_do_not_prevent_explicit_mapping(tmp_path, ontology):
    mapping = ManualMapping.load(write_config(tmp_path, {"rain": rule()}), ontology)
    resolution = mapping.resolve("Steady rain", ("future_backend_class", "root"))
    assert resolution.status == "supported"
    assert resolution.target_class_ids == ("root",)


@pytest.fixture
def prediction():
    # The distracting foreign class has the largest score on every frame.
    return FramePrediction(
        probabilities=np.array(
            [[0.95, 0.1, 0.6], [0.9, 0.7, 0.2], [0.8, 0.0, 0.0]], dtype=np.float32
        ),
        frame_start_seconds=np.array([0.0, 0.04, 0.08]),
        frame_end_seconds=np.array([0.04, 0.08, 0.12]),
        class_ids=("unrelated", "right", "left"),
        class_names=("Unrelated", "Right", "Left"),
        metadata={"purpose": "synthetic aggregation test"},
    )


def test_target_max_uses_ids_across_scrambled_columns(tmp_path, ontology, prediction):
    mapping = ManualMapping.load(
        write_config(
            tmp_path,
            {"rain": rule(allowed_classes=["left", "right"], foreign_classes=["unrelated"])},
        ),
        ontology,
    )
    # Resolution and prediction have different class orderings on purpose.
    resolution = mapping.resolve("Steady rain", ("left", "unrelated", "right"))

    actual = aggregate_target(prediction, resolution)

    np.testing.assert_array_equal(actual, np.array([0.6, 0.7, 0.0], dtype=np.float32))
    assert actual.shape == (3,)


def test_unsupported_mapping_cannot_become_an_all_zero_score(tmp_path, ontology, prediction):
    mapping = ManualMapping.load(write_config(tmp_path, {"rain": rule()}), ontology)
    for resolution in [
        mapping.resolve(None, prediction.class_ids),
        mapping.resolve("Steady rain", prediction.class_ids),
    ]:
        with pytest.raises(ValueError, match="unsupported"):
            aggregate_target(prediction, resolution)


def test_aggregation_rejects_prediction_missing_a_resolved_target(tmp_path, ontology, prediction):
    mapping = ManualMapping.load(write_config(tmp_path, {"rain": rule()}), ontology)
    resolution = mapping.resolve("Steady rain", ("root",))
    with pytest.raises(ValueError, match="missing mapped classes"):
        aggregate_target(prediction, resolution)


def test_official_model_only_id_can_be_mapped_exactly_and_reports_ontology_gap(tmp_path):
    ontology = AudioSetOntology.load()
    model_ids, _ = load_labels()
    model_only = next(class_id for class_id in model_ids if class_id not in ontology.nodes)
    path = write_config(tmp_path, {"explicit": rule(allowed_classes=[model_only])})
    mapping = ManualMapping.load(path, ontology)

    resolution = mapping.resolve("Steady rain")

    assert resolution.status == "supported"
    assert resolution.target_class_ids == (model_only,)
    assert model_only in resolution.ontology_missing_class_ids
    assert model_only in resolution.to_dict()["ontology_missing_class_ids"]
    assert resolution.unavailable_class_ids == ()


def test_model_only_id_cannot_request_unknown_descendants(tmp_path):
    ontology = AudioSetOntology.load()
    model_ids, _ = load_labels()
    model_only = next(class_id for class_id in model_ids if class_id not in ontology.nodes)
    path = write_config(
        tmp_path, {"explicit": rule(allowed_classes=[model_only], include_descendants=True)}
    )
    with pytest.raises(ValueError):
        ManualMapping.load(path, ontology)


def test_default_model_vocabulary_accepts_archived_ontology_gaps(tmp_path):
    ontology = AudioSetOntology.load()
    model_ids, _ = load_labels()
    known = next(class_id for class_id in model_ids if class_id in ontology.nodes)
    path = write_config(tmp_path, {"explicit": rule(allowed_classes=[known])})
    mapping = ManualMapping.load(path)

    resolution = mapping.resolve("Steady rain")

    assert resolution.status == "supported"
    assert resolution.target_class_ids == (known,)


def test_duplicate_mapping_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"mappings":{},"mappings":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate mapping JSON key"):
        ManualMapping.load(path)


def test_invalid_description_and_prediction_id_types_are_explicit_errors(tmp_path, ontology):
    mapping = ManualMapping.load(write_config(tmp_path, {"rain": rule()}), ontology)
    with pytest.raises(ValueError, match="Source description"):
        mapping.resolve(123)
    with pytest.raises(ValueError, match="Prediction class IDs"):
        mapping.resolve("Steady rain", (123,))
