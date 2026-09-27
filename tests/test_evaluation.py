"""Evidence availability, source identity and semantic reports without inference."""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig, EventConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import StemMetadata, load_stems
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256

BARK = "/m/05tny_"
DOG = "/m/0bt9lr"
SPEECH = "/m/09x0r"
KNIFE = "/m/04ctx"


@pytest.fixture
def evidence(tmp_path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"Synthetic identity fixture, not audio requiring decoding")
    config = tmp_path / "mappings.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mappings": {
                    "bark": {
                        "descriptions": ["dog barking"],
                        "allowed_classes": [BARK],
                        "foreign_classes": [SPEECH],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    mapping = ManualMapping.load(config)
    stem = StemMetadata(
        stem_id="clip::dog",
        candidate_id="dog",
        clip_key="clip",
        audio_path=audio,
        source_description="dog barking",
        role="onset",
        expected_intervals=((0.0, 0.25), (0.5, 0.75)),
        provenance={
            "reference_origin": "waves_pass1_planned",
            "reference_status": "planned",
            "raw_final_stem": {"sha256": sha256(audio)},
        },
    )
    prediction = FramePrediction(
        np.array(
            [
                [0.8, 0.9, 0.1, 0.95],
                [0.1, 0.9, 0.7, 0.95],
                [0.9, 0.9, 0.1, 0.95],
                [0.1, 0.9, 0.1, 0.95],
            ],
            dtype=np.float32,
        ),
        np.array([0.0, 0.25, 0.5, 0.75]),
        np.array([0.25, 0.5, 0.75, 1.0]),
        (BARK, DOG, SPEECH, KNIFE),
        ("Bark", "Dog", "Speech", "Knife"),
        {
            "audio_path": str(audio),
            "audio_sha256": sha256(audio),
            "backend": "synthetic_test",
            "duration_seconds": 1.0,
        },
    )
    return stem, prediction, mapping


def evaluate(evidence, **kwargs):
    stem, prediction, mapping = evidence
    return evaluate_stem(stem, prediction, mapping, EvaluationConfig(), **kwargs)


def test_onset_evaluation_and_full_provenance(evidence, tmp_path):
    stem, prediction, mapping = evidence
    path = tmp_path / "raw.npz"
    prediction.save(path)
    before = path.read_bytes()
    report = evaluate(evidence, prediction_path=path)
    assert report["evaluation_status"] == "evaluated"
    assert report["detection_status"] == "available"
    assert report["metrics"]["matched_event_count"] == 2
    assert report["metrics"]["missing_event_count"] == 0
    assert report["metrics"]["mean_onset_error_ms"] == 0
    assert report["detected_intervals"] == [[0, 0.25], [0.5, 0.75]]
    assert report["cache_validation"]["status"] == "verified_current_audio"
    assert report["cache_validation"]["source_file_verified"]
    assert report["decision"] is None
    assert report["provenance"]["prediction"]["sha256"] == sha256(path)
    assert report["provenance"]["mapping"] == mapping.provenance
    assert report["provenance"]["stem"] == stem.to_dict()
    assert path.read_bytes() == before
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "role,metric", [("span", "temporal_iou"), ("ambience", "occupancy_in_expected_span")]
)
def test_role_specific_metrics(evidence, role, metric):
    stem, prediction, mapping = evidence
    report = evaluate((replace(stem, role=role), prediction, mapping))
    assert report["metrics"][metric] == 1
    assert "matched_event_count" not in report["metrics"]
    if role == "ambience":
        assert report["metrics"]["mean_target_confidence"] == pytest.approx(0.85)
        assert "onset_error" not in report["metrics"]


def test_semantics_related_classes_are_not_automatically_foreign(evidence):
    report = evaluate(evidence)
    items = {
        item["class_id"]: item for item in report["semantic_evidence"]["outside_allowed_family"]
    }
    assert "ancestor" in items[DOG]["relations_to_allowed"]
    assert not items[DOG]["explicit_foreign"]
    assert items[SPEECH]["explicit_foreign"]
    assert items[KNIFE]["relations_to_allowed"] == ["unknown_hierarchy"]
    assert items[KNIFE]["ontology_name"] is None
    assert BARK not in items
    assert report["decision"] is None


def test_explicit_foreign_reporting_is_not_lost_to_top_k(evidence):
    stem, prediction, mapping = evidence
    report = evaluate_stem(stem, prediction, mapping, EvaluationConfig(outside_family_top_k=1))
    semantic = report["semantic_evidence"]
    assert len(semantic["outside_allowed_family"]) == 1
    assert semantic["explicit_foreign_classes"][0]["class_id"] == SPEECH
    assert semantic["explicit_foreign_classes"][0]["activation_duration_seconds"] == 0.25


def test_cache_identity_uses_bytes_not_path(evidence, tmp_path):
    stem, prediction, mapping = evidence
    relocated = tmp_path / "relocated.wav"
    relocated.write_bytes(stem.audio_path.read_bytes())
    report = evaluate((replace(stem, audio_path=relocated), prediction, mapping))
    assert report["evaluation_status"] == "evaluated"


def test_verification_script_protects_recorded_source_when_audio_was_relocated(
    evidence, tmp_path, monkeypatch
):
    stem, prediction, _ = evidence
    original = stem.audio_path.read_bytes()
    relocated = tmp_path / "relocated.wav"
    relocated.write_bytes(original)
    cache = tmp_path / "raw.npz"
    prediction.save(cache)
    output = tmp_path / "demo"
    first_output = output / "threshold-0.2"
    first_output.mkdir(parents=True)
    try:
        (first_output / "dataset.json").hardlink_to(stem.audio_path)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_cached_evaluation.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--prediction",
            str(cache),
            "--audio",
            str(relocated),
            "--class-id",
            BARK,
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        runpy.run_path(str(script), run_name="__main__")
    assert stem.audio_path.read_bytes() == original
    assert not (first_output / "stems").exists()


def test_changed_source_is_not_evaluated(evidence):
    stem, _, _ = evidence
    stem.audio_path.write_bytes(b"Different audio bytes")
    report = evaluate(evidence)
    assert report["metrics"] is None
    assert report["detected_intervals"] is None
    assert {"source_audio_manifest_mismatch", "prediction_audio_mismatch"}.issubset(
        report["reason_codes"]
    )


def test_cache_from_other_audio_is_not_evaluated(evidence):
    evidence[1].metadata["audio_sha256"] = "a" * 64
    report = evaluate(evidence)
    assert report["evaluation_status"] == "unavailable"
    assert report["reason_codes"] == ["prediction_audio_mismatch"]


def test_missing_original_can_use_explicit_manifest_digest(evidence):
    evidence[0].audio_path.unlink()
    report = evaluate(evidence)
    assert report["evaluation_status"] == "evaluated"
    assert report["cache_validation"]["status"] == "verified_manifest_hash"
    assert not report["cache_validation"]["source_file_verified"]


def test_missing_original_without_digest_is_not_matched_by_path(evidence):
    stem, prediction, mapping = evidence
    stem.audio_path.unlink()
    provenance = {**stem.provenance, "raw_final_stem": {}}
    report = evaluate((replace(stem, provenance=provenance), prediction, mapping))
    assert report["metrics"] is None
    assert "audio_identity_unverified" in report["reason_codes"]


@pytest.mark.parametrize("bad_hash", [None, "", "incorrect", True])
def test_missing_or_invalid_cache_digest_is_unavailable(evidence, bad_hash):
    evidence[1].metadata["audio_sha256"] = bad_hash
    report = evaluate(evidence)
    assert report["metrics"] is None
    assert "missing_or_invalid_cache_audio_sha256" in report["reason_codes"]


def test_unsupported_mapping_does_not_become_zero_events(evidence):
    stem, prediction, mapping = evidence
    report = evaluate((replace(stem, source_description="unmapped source"), prediction, mapping))
    assert report["mapping_status"] == "unsupported_mapping"
    assert report["metrics"] is None
    assert report["detected_intervals"] is None


def test_missing_prediction_retains_reference_without_fake_metrics(evidence):
    stem, _, mapping = evidence
    report = evaluate((stem, None, mapping))
    assert "missing_prediction" in report["reason_codes"]
    assert report["metrics"] is None
    assert report["expected_intervals"] == [list(item) for item in stem.expected_intervals]


def test_missing_prediction_keeps_requested_path_for_audit(evidence, tmp_path):
    stem, _, mapping = evidence
    path = tmp_path / "missing.npz"
    report = evaluate((stem, None, mapping), prediction_path=path)
    assert report["provenance"]["prediction"] is None
    assert report["provenance"]["prediction_request"]["path"] == str(path.resolve())


def test_ambiguous_reference_is_only_used_on_explicit_opt_in(evidence):
    stem, prediction, mapping = evidence
    stem = replace(stem, provenance={**stem.provenance, "reference_status": "ambiguous"})
    report = evaluate((stem, prediction, mapping))
    assert report["detection_status"] == "available"
    assert report["metrics"] is None
    assert report["reference"]["status"] == "ambiguous"
    allowed = evaluate_stem(
        stem, prediction, mapping, EvaluationConfig(allow_ambiguous_reference=True)
    )
    assert allowed["evaluation_status"] == "evaluated"
    assert allowed["reference"]["status"] == "ambiguous"
    assert allowed["provenance"]["evaluation_config"]["allow_ambiguous_reference"] is True


@pytest.mark.parametrize("intervals", [None, ()])
def test_missing_or_empty_reference_keeps_detection_but_no_metrics(evidence, intervals):
    stem, prediction, mapping = evidence
    report = evaluate((replace(stem, expected_intervals=intervals), prediction, mapping))
    assert report["metrics"] is None
    assert report["detection_status"] == "available"


def test_reference_outside_audio_is_not_clipped(evidence):
    stem, prediction, mapping = evidence
    report = evaluate((replace(stem, expected_intervals=((0, 1.01),)), prediction, mapping))
    assert report["metrics"] is None
    assert "reference_outside_prediction" in report["reason_codes"]
    assert report["expected_intervals"] == [[0, 1.01]]


@pytest.mark.parametrize("duration", [None, True, 0, -1, 2])
def test_incomplete_or_undocumented_prediction_duration_is_unavailable(evidence, duration):
    evidence[1].metadata["duration_seconds"] = duration
    report = evaluate(evidence)
    assert report["metrics"] is None
    assert report["detection_status"] == "unavailable"


def test_internal_prediction_gap_does_not_create_silent_time(evidence):
    evidence[1].frame_start_seconds[1] = 0.3
    report = evaluate(evidence)
    assert report["metrics"] is None
    assert "incomplete_prediction_timeline" in report["reason_codes"]


def test_threshold_re_evaluation_preserves_raw_prediction(evidence):
    stem, prediction, mapping = evidence
    before = prediction.probabilities.copy()
    report = evaluate_stem(
        stem, prediction, mapping, EvaluationConfig(event=EventConfig(threshold=1.0))
    )
    assert report["metrics"]["missing_event_count"] == 2
    assert report["metrics"]["event_precision"] is None
    assert report["metrics"]["mean_onset_error_ms"] is None
    np.testing.assert_array_equal(prediction.probabilities, before)


def test_reference_interface_can_be_replaced(evidence):
    from waves_sed.temporal_reference import TemporalSupport

    class FixtureExternalReference:
        def resolve(self, stem, *, allow_ambiguous=False):
            return TemporalSupport(
                "fixture_external_annotation",
                "available",
                ((0.25, 0.5),),
                True,
                None,
                {"fixture": True},
            )

    report = evaluate(evidence, reference=FixtureExternalReference())
    assert report["evaluation_status"] == "evaluated"
    assert report["reference"]["origin"] == "fixture_external_annotation"
    assert report["metrics"]["missing_event_count"] == 1
    assert report["metrics"]["extra_event_count"] == 2


def test_normalized_manifest_roundtrip_and_relative_paths(evidence, tmp_path):
    stem, _, _ = evidence
    row = stem.to_dict()
    row["audio_path"] = "source.wav"
    row["mapping"] = {"status": "obsolete_hint"}
    path = tmp_path / "stems.json"
    path.write_text(json.dumps({"schema_version": 1, "stems": [row]}), encoding="utf-8")
    assert load_stems(path) == [stem]


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(schema_version=True),
        lambda value: value["stems"].append(value["stems"][0]),
        lambda value: value["stems"][0].update(expected_intervals=[[1, 0]]),
        lambda value: value["stems"][0].update(expected_intervals=[[1, 2], [0, 1]]),
        lambda value: value["stems"][0].update(provenance={"source_media_paths": "not a list"}),
        lambda value: value["stems"][0].update(provenance={"inputs": {"dsp": None}}),
        lambda value: value["stems"][0].update(provenance={"raw_final_stem": []}),
        lambda value: value["stems"][0].update(role=True),
        lambda value: value["stems"][0].update(stem_id=""),
        lambda value: value["stems"][0].update(extra="typo"),
    ],
)
def test_malformed_normalized_metadata_is_rejected(evidence, tmp_path, change):
    value = {"schema_version": 1, "stems": [evidence[0].to_dict()]}
    change(value)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_stems(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"stems":[],"stems":[]}',
        '{"schema_version":1,"stems":[],"summary":{"nan":NaN}}',
    ],
)
def test_duplicate_or_nonfinite_manifest_is_rejected(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_stems(path)
