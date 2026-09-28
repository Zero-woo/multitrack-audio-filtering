"""Synthetic probabilities test paired measurements independently of SED accuracy."""

import csv
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from waves_sed.corruption_comparison import compare_corruptions, write_comparison
from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import StemMetadata
from waves_sed.prediction import FramePrediction


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def make_experiment(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mappings": {
                    "bark": {"descriptions": ["dog barking"], "allowed_classes": ["/m/05tny_"]}
                },
            }
        ),
        encoding="utf-8",
    )
    mapping = ManualMapping.load(mapping_path)

    def build(variants, *, role="onset", expected=((1.0, 1.2),), control_detected=None):
        source = StemMetadata(
            stem_id="original",
            candidate_id="candidate",
            clip_key="clip",
            role=role,
            source_description="dog barking",
            audio_path=tmp_path / "source.wav",
            expected_intervals=expected,
            provenance={
                "reference_origin": "waves_pass1_planned",
                "reference_status": "planned",
                "support_policy": "selected_parent_only",
            },
        ).to_dict()
        source_audio = {
            "path": source["audio_path"],
            "sha256": digest("source"),
            "sample_rate": 100,
            "frames": 600,
            "channels": 1,
        }

        def entry(variant_id, operation, support, *, application=None, status="generated"):
            stem = deepcopy(source)
            stem.update(
                stem_id=variant_id,
                candidate_id=variant_id,
                audio_path=source["audio_path"]
                if operation == "control"
                else str(tmp_path / f"{variant_id}.wav"),
            )
            audio_sha = source_audio["sha256"] if operation == "control" else digest(variant_id)
            spec = {"id": variant_id, "operation": operation}
            app = {"audio_changed": True, **(application or {})}
            stem["provenance"]["raw_final_stem"] = {"sha256": audio_sha}
            stem["provenance"]["corruption"] = {
                "experiment_id": "experiment",
                "kind": "control" if operation == "control" else "variant",
                "variant_id": variant_id,
                "parent_stem_id": "original",
                "control_stem_id": "control",
                "operation": operation,
                "source_audio": {key: source_audio[key] for key in ("path", "sha256")},
                "spec": None if operation == "control" else spec,
                "application": None if operation == "control" else app,
                "source_stem": deepcopy(source),
            }
            result = {
                "id": variant_id,
                "stem_id": variant_id,
                "audio_path": stem["audio_path"],
                "audio_sha256": audio_sha,
                "operation": operation,
                "spec": spec,
                "application": app,
                "status": status,
                "error": None,
                "stem": stem,
            }
            if status != "generated":
                result.update(
                    audio_path=None,
                    audio_sha256=None,
                    stem=None,
                    application=None,
                    error="controlled failure",
                )
                return result, None
            edges = np.arange(601, dtype=np.float64) / 100
            scores = np.zeros((600, 1), dtype=np.float32)
            for start, end in support:
                scores[int(round(start * 100)) : int(round(end * 100)), 0] = 0.9
            prediction = FramePrediction(
                probabilities=scores,
                frame_start_seconds=edges[:-1],
                frame_end_seconds=edges[1:],
                class_ids=("/m/05tny_",),
                class_names=("Bark",),
                metadata={
                    "backend": "synthetic_oracle",
                    "audio_sha256": audio_sha,
                    "duration_seconds": 6,
                    "preprocessing": "synthetic-10ms-v1",
                },
            )
            report = evaluate_stem(
                StemMetadata.from_dict(stem), prediction, mapping, EvaluationConfig()
            )
            assert report["evaluation_status"] == "evaluated"
            return result, report

        control, control_report = entry(
            "control", "control", expected if control_detected is None else control_detected
        )
        control = {key: control[key] for key in ("stem_id", "audio_path", "audio_sha256", "stem")}
        experiment = {
            "schema_version": 1,
            "experiment_id": "experiment",
            "source_stem": source,
            "source_audio": source_audio,
            "config": {"path": "config.json", "sha256": digest("config")},
            "control": control,
            "variants": [],
            "reference_policy": "unchanged_from_source",
        }
        dataset = {
            "schema_version": 1,
            "stems": [control_report],
            "provenance": {"synthetic": True},
        }
        for index, variant in enumerate(variants):
            variant = dict(variant)
            support = variant.pop("support")
            variant_id = variant.pop("id", f"variant{index}")
            item, report = entry(variant_id, support=support, **variant)
            experiment["variants"].append(item)
            if report is not None:
                dataset["stems"].append(report)
        return experiment, dataset

    return build


@pytest.mark.parametrize("shift", [0.1, 0.2, 0.5, 1.0])
def test_known_shift_observed_displacement_separate_from_tolerance_metrics(make_experiment, shift):
    experiment, dataset = make_experiment(
        [
            {
                "operation": "shift",
                "support": [(1 + shift, 1.2 + shift)],
                "application": {"realized_shift_seconds": shift},
            }
        ]
    )
    original = deepcopy((experiment, dataset))
    result = compare_corruptions(experiment, dataset)
    row = result["comparisons"][0]
    assert row["comparison_status"] == "compared", row["reason_codes"]
    assert row["detection_changes"]["single_event_onset_displacement_ms"] == pytest.approx(
        shift * 1000
    )
    assert row["detection_changes"]["single_event_shift_error_ms"] == pytest.approx(0, abs=1e-10)
    if shift <= 0.2:
        assert row["metric_deltas"]["mean_onset_error_ms"] == pytest.approx(shift * 1000)
        assert row["metric_deltas"]["missing_event_count"] == 0
    else:
        assert row["metric_deltas"]["mean_onset_error_ms"] is None
        assert row["metric_deltas"]["missing_event_count"] == 1
        assert row["metric_deltas"]["extra_event_count"] == 1
    assert "matches" not in row["metric_deltas"]
    assert result["decision"] is None
    assert (experiment, dataset) == original


def test_missing_and_extra_are_counted_relative_to_control(make_experiment):
    experiment, dataset = make_experiment(
        [
            {"operation": "remove", "support": []},
            {"operation": "duplicate", "support": [(1, 1.2), (3, 3.2)]},
        ]
    )
    result = compare_corruptions(experiment, dataset)
    missing, extra = result["comparisons"]
    assert missing["metric_deltas"]["missing_event_count"] == 1
    assert missing["detection_changes"]["detection_count_delta"] == -1
    assert missing["detection_changes"]["single_event_onset_displacement_ms"] is None
    assert extra["metric_deltas"]["extra_event_count"] == 1
    assert extra["detection_changes"]["detection_count_delta"] == 1
    assert extra["detection_changes"]["detected_support_iou"] == pytest.approx(0.5)
    assert extra["detection_changes"]["single_event_onset_displacement_ms"] is None


def test_shortened_extended_spans_and_ambience_use_role_metrics(make_experiment):
    experiment, dataset = make_experiment(
        [
            {"operation": "shorten", "support": [(1, 2)]},
            {"operation": "extend", "support": [(1, 4)]},
        ],
        role="span",
        expected=((1, 3),),
    )
    result = compare_corruptions(experiment, dataset)
    shortened, extended = result["comparisons"]
    assert shortened["metric_deltas"]["expected_coverage"] == -0.5
    assert shortened["metric_deltas"]["offset_error"] == -1
    assert extended["metric_deltas"]["offset_error"] == 1
    assert extended["metric_deltas"]["out_of_window_activation"] == 1
    experiment, dataset = make_experiment(
        [
            {"operation": "shorten", "support": [(1, 2)]},
        ],
        role="ambience",
        expected=((1, 3),),
    )
    row = compare_corruptions(experiment, dataset)["comparisons"][0]
    assert row["metric_deltas"]["occupancy_in_expected_span"] == -0.5
    assert row["metric_deltas"]["mean_target_confidence"] == pytest.approx(-0.45)
    assert "offset_error" not in row["metric_deltas"]


def test_missing_reports_failed_generation_and_unsupported_never_create_deltas(make_experiment):
    experiment, dataset = make_experiment(
        [
            {"operation": "shift", "support": [(1.1, 1.3)]},
            {"operation": "remove", "support": [], "status": "failed"},
        ]
    )
    dataset["stems"].pop(0)
    result = compare_corruptions(experiment, dataset)
    assert result["summary"]["compared_count"] == 0
    assert all("control_report_missing" in row["reason_codes"] for row in result["comparisons"])
    assert result["comparisons"][0]["variant_metrics"] is not None
    assert "variant_generation_failed" in result["comparisons"][1]["reason_codes"]
    assert all(row["metric_deltas"] is None for row in result["comparisons"])
    assert result["summary"]["metric_deltas"]["missing_event_count"] == {
        "mean": None,
        "contributing_count": 0,
    }
    experiment, dataset = make_experiment([{"operation": "remove", "support": []}])
    dataset["stems"][1].update(
        evaluation_status="unavailable", mapping_status="unsupported_mapping", metrics=None
    )
    row = compare_corruptions(experiment, dataset)["comparisons"][0]
    assert "variant_evaluation_unavailable" in row["reason_codes"]
    assert row["metric_deltas"] is None
    dataset["stems"].pop()
    assert (
        "variant_report_missing"
        in compare_corruptions(experiment, dataset)["comparisons"][0]["reason_codes"]
    )


def test_no_change_pair_excluded_from_sensitivity_means(make_experiment):
    experiment, dataset = make_experiment(
        [
            {
                "operation": "shift",
                "support": [(1.1, 1.3)],
                "application": {"realized_shift_seconds": 0.1},
            },
            {"operation": "shift", "support": [(1, 1.2)], "application": {"audio_changed": False}},
            {
                "operation": "shift",
                "support": [(2, 2.2)],
                "application": {"realized_shift_seconds": 1.0},
            },
        ]
    )
    unrelated = deepcopy(dataset["stems"][0])
    unrelated["stem_id"] = "unrelated"
    dataset["stems"].append(unrelated)
    result = compare_corruptions(experiment, dataset)
    assert result["summary"]["compared_count"] == 2
    assert result["summary"]["unavailable_count"] == 1
    assert result["summary"]["ignored_dataset_stem_count"] == 1
    assert result["comparisons"][1]["reason_codes"] == ["no_waveform_change"]
    aggregate = result["summary"]["by_operation"]["shift"]["metric_deltas"]["mean_onset_error_ms"]
    assert aggregate["mean"] == pytest.approx(100)
    assert aggregate["contributing_count"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "config",
        "metric_version",
        "conventions",
        "mapping",
        "resolution",
        "backend",
        "checkpoint",
        "preprocessing",
        "class_ids",
        "class_names",
        "provider",
        "reference_status",
        "reference_intervals",
        "audio_hash",
        "stem",
        "source_description",
        "role",
        "corruption",
        "manifest_reference",
    ],
)
def test_mismatched_pair_becomes_unavailable(make_experiment, field):
    experiment, dataset = make_experiment([{"operation": "shift", "support": [(1.1, 1.3)]}])
    row = dataset["stems"][1]
    provenance = row["provenance"]
    if field == "config":
        provenance["evaluation_config"]["event"]["threshold"] = 0.7
    elif field == "metric_version":
        provenance["metric_implementation_version"] = 99
    elif field == "conventions":
        provenance["metric_conventions"] = {"onset": "different"}
    elif field == "mapping":
        provenance["mapping"] = {"sha256": digest("other")}
    elif field == "resolution":
        row["mapping"]["target_class_ids"] = ["/m/09x0r"]
    elif field in {"backend", "checkpoint", "preprocessing", "audio_hash"}:
        key = {"checkpoint": "checkpoint_sha256", "audio_hash": "audio_sha256"}.get(field, field)
        provenance["prediction"]["metadata"][key] = "other"
    elif field in {"class_ids", "class_names"}:
        provenance["prediction"][field] = ["other"]
    elif field == "provider":
        row["reference"]["provenance"]["provider"] = "other"
    elif field == "reference_status":
        row["reference"]["status"] = "ambiguous"
    elif field == "reference_intervals":
        row["reference"]["intervals"] = [[0, 1]]
    elif field == "stem":
        provenance["stem"]["candidate_id"] = "other"
    elif field in {"source_description", "role"}:
        row[field] = "other"
    elif field == "corruption":
        experiment["variants"][0]["stem"]["provenance"]["corruption"]["parent_stem_id"] = "other"
    else:
        experiment["variants"][0]["stem"]["provenance"]["reference_status"] = "ambiguous"
    result = compare_corruptions(experiment, dataset)
    assert result["comparisons"][0]["comparison_status"] == "unavailable"
    assert result["comparisons"][0]["reason_codes"]
    assert result["comparisons"][0]["metric_deltas"] is None
    assert result["summary"]["compared_count"] == 0


@pytest.mark.parametrize(
    "change", ["duplicate_report", "duplicate_variant", "nan", "schema", "policy", "missing_flag"]
)
def test_structural_errors_fail_explicitly(make_experiment, change):
    experiment, dataset = make_experiment([{"operation": "shift", "support": [(1.1, 1.3)]}])
    if change == "duplicate_report":
        dataset["stems"].append(dataset["stems"][0])
    elif change == "duplicate_variant":
        experiment["variants"].append(experiment["variants"][0])
    elif change == "nan":
        dataset["stems"][0]["metrics"]["event_recall"] = float("nan")
    elif change == "schema":
        experiment["schema_version"] = True
    elif change == "policy":
        experiment["reference_policy"] = "shift_expected_with_audio"
    else:
        del experiment["variants"][0]["application"]["audio_changed"]
    with pytest.raises(ValueError):
        compare_corruptions(experiment, dataset)


def test_empty_detection_support_has_null_iou_and_no_event_identity_claim(make_experiment):
    experiment, dataset = make_experiment(
        [
            {"operation": "shift", "support": [], "application": {"realized_shift_seconds": 0.5}},
        ],
        control_detected=[],
    )
    row = compare_corruptions(experiment, dataset)["comparisons"][0]
    assert row["comparison_status"] == "compared"
    assert row["detection_changes"] == {
        "detection_count_delta": 0,
        "detected_support_iou": None,
        "single_event_onset_displacement_ms": None,
        "single_event_shift_error_ms": None,
    }


def test_written_csv_scalars_nulls_json_and_source_protection(make_experiment, tmp_path):
    experiment, dataset = make_experiment(
        [
            {"operation": "shift", "support": [(1.5, 1.7)]},
            {"operation": "remove", "support": [], "status": "failed"},
        ]
    )
    result = compare_corruptions(experiment, dataset)
    output = tmp_path / "report"
    write_comparison(result, output, protected_inputs=[])
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == result
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["delta_missing_event_count"] == "1"
    assert rows[0]["delta_mean_onset_error_ms"] == ""
    assert rows[1]["delta_missing_event_count"] == ""
    assert json.loads(rows[1]["reason_codes"]) == ["variant_generation_failed"]
    snapshot = (output / "comparison.csv").read_bytes()
    with pytest.raises(ValueError, match="overwrite input"):
        write_comparison(result, output, protected_inputs=[output / "comparison.json"])
    assert (output / "comparison.csv").read_bytes() == snapshot


def test_hardlink_alias_preflight_happens_before_any_write(make_experiment, tmp_path):
    experiment, dataset = make_experiment([{"operation": "remove", "support": []}])
    result = compare_corruptions(experiment, dataset)
    output = tmp_path / "protected"
    output.mkdir()
    source = tmp_path / "source.json"
    source.write_text("preserve bytes", encoding="utf-8")
    (output / "comparison.json").hardlink_to(source)
    with pytest.raises(ValueError, match="overwrite input"):
        write_comparison(result, output, protected_inputs=[source])
    assert source.read_text(encoding="utf-8") == "preserve bytes"
    assert not (output / "comparison.csv").exists()


@pytest.mark.parametrize("field", ["path", "sha256"])
def test_control_must_retain_original_audio_identity(make_experiment, field):
    experiment, dataset = make_experiment([{"operation": "remove", "support": []}])
    experiment["source_audio"][field] = (
        digest("different") if field == "sha256" else "elsewhere.wav"
    )
    row = compare_corruptions(experiment, dataset)["comparisons"][0]
    assert row["comparison_status"] == "unavailable"
    assert "control_original_audio_mismatch" in row["reason_codes"]
