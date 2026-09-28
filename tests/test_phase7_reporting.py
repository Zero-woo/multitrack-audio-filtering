"""Quality labels are audited stem counts, independent from metric contributors."""

import copy
import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from waves_sed.decision import FilterConfig, decide
from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import StemMetadata
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256
from waves_sed.reporting import summarize_records, write_reports


@pytest.fixture
def report_factory(tmp_path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"Identity-only synthetic fixture; no waveform decoding")
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mappings": {
                    "bark": {
                        "descriptions": ["dog barking"],
                        "allowed_classes": ["/m/05tny_"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    mapping = ManualMapping.load(mapping_file)
    stem = StemMetadata(
        stem_id="clip::dog",
        candidate_id="dog",
        clip_key="clip",
        audio_path=audio,
        source_description="dog barking",
        role="span",
        expected_intervals=((0.0, 0.25),),
        provenance={
            "reference_origin": "waves_pass1_planned",
            "reference_status": "planned",
            "raw_final_stem": {"sha256": sha256(audio)},
        },
    )
    prediction = FramePrediction(
        probabilities=np.array([[0.8], [0.1], [0.1], [0.1]], dtype=np.float32),
        frame_start_seconds=np.array([0.0, 0.25, 0.5, 0.75]),
        frame_end_seconds=np.array([0.25, 0.5, 0.75, 1.0]),
        class_ids=("/m/05tny_",),
        class_names=("Bark",),
        metadata={
            "audio_path": str(audio),
            "audio_sha256": sha256(audio),
            "backend": "synthetic_test",
            "duration_seconds": 1.0,
        },
    )

    def make(stem_id="one", role="span", available=True, clip="clip"):
        selected = replace(stem, stem_id=stem_id, candidate_id=stem_id, role=role, clip_key=clip)
        return evaluate_stem(
            selected, prediction if available else None, mapping, EvaluationConfig()
        )

    return make


def attach_decision(report, config):
    report = copy.deepcopy(report)
    audit = decide(report, config)
    report["decision"] = audit["decision"]
    report["decision_details"] = audit
    report["provenance"]["filter_config"] = config.to_dict()
    report["provenance"]["filter_config_provenance"] = config.provenance
    return report


def span_policy(pass_threshold=0.8, fail_threshold=0.5):
    return FilterConfig.from_dict(
        {
            "schema_version": 1,
            "filter": {"span": {"min_tiou": {"pass": pass_threshold, "fail": fail_threshold}}},
        }
    )


def test_audited_counts_keep_all_labels_and_unassigned_separate_from_measurements(report_factory):
    config = span_policy()
    rows = []
    for stem_id, score in (("pass", 0.9), ("review", 0.6), ("fail", 0.4), ("null", None)):
        row = report_factory(stem_id)
        row["metrics"]["temporal_iou"] = score
        rows.append(attach_decision(row, config))
    rows.extend(
        [
            attach_decision(report_factory("missing", available=False), config),
            attach_decision(report_factory("unsupported", role=None), config),
            attach_decision(report_factory("not-configured", role="onset"), config),
        ]
    )
    summary = summarize_records(rows)
    assert summary["decision_summary"]["counts"] == {
        "PASS": 1,
        "REVIEW": 3,
        "FAIL": 1,
        "UNSUPPORTED": 1,
        "unassigned": 1,
    }
    assert summary["decision_summary"]["statuses"] == {"decided": 6, "not_configured": 1}
    assert summary["roles"]["span"]["decision_summary"]["counts"] == {
        "PASS": 1,
        "REVIEW": 3,
        "FAIL": 1,
        "UNSUPPORTED": 0,
        "unassigned": 0,
    }
    assert summary["roles"]["span"]["metrics"]["temporal_iou"] == {
        "mean": pytest.approx(1.9 / 3),
        "contributing_count": 3,
    }
    assert summary["roles"]["onset"]["decision_summary"]["counts"]["unassigned"] == 1
    assert summary["roles"]["unspecified"]["decision_summary"]["counts"]["UNSUPPORTED"] == 1
    assert summary["decision"] is None


@pytest.mark.parametrize("other", ["different", "disabled", "legacy"])
def test_assigned_decisions_cannot_mix_policy_contexts(report_factory, other):
    first = attach_decision(report_factory("first"), span_policy())
    second = report_factory("second")
    if other != "legacy":
        second = attach_decision(
            second, span_policy(0.9, 0.7) if other == "different" else FilterConfig()
        )
    with pytest.raises(ValueError, match="Cannot aggregate assigned decisions"):
        summarize_records([first, second])


def test_changed_measurement_invalidates_existing_audit(report_factory):
    row = attach_decision(report_factory(), span_policy())
    row["metrics"]["temporal_iou"] = 0.1
    with pytest.raises(ValueError, match="does not match the configured decision engine"):
        summarize_records([row])


def test_semantically_identical_policies_can_have_different_file_provenance(
    report_factory, tmp_path
):
    value = {"schema_version": 1, "filter": {"span": {"min_tiou": 0.8}}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(value), encoding="utf-8")
    second.write_text(json.dumps(value, indent=2), encoding="utf-8")
    first_config, second_config = FilterConfig.load(first), FilterConfig.load(second)
    assert first_config.provenance != second_config.provenance
    rows = [
        attach_decision(report_factory("first"), first_config),
        attach_decision(report_factory("second"), second_config),
    ]
    assert summarize_records(rows)["decision_summary"]["counts"]["PASS"] == 2


def test_clip_json_csv_and_dataset_counts_agree_without_aggregate_quality_label(
    report_factory, tmp_path
):
    config = span_policy()
    rows = [
        attach_decision(report_factory("one", clip="first"), config),
        attach_decision(report_factory("two", available=False, clip="second"), config),
    ]
    dataset = write_reports(rows, tmp_path / "reports", protected_inputs=[], provenance={})
    assert dataset["summary"]["decision_summary"]["counts"]["PASS"] == 1
    assert dataset["summary"]["decision_summary"]["counts"]["REVIEW"] == 1
    for clip in dataset["clips"]:
        clip_path = tmp_path / "reports" / dataset["files"]["clips"][clip["clip_key"]]
        value = json.loads(clip_path.read_text(encoding="utf-8"))
        assert value["summary"] == clip["summary"]
        assert value["decision"] is None
    with (tmp_path / "reports" / "clips.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert all(row["decision"] == "" for row in csv_rows)
    assert json.loads(csv_rows[0]["decision_counts"])["PASS"] == 1
    assert json.loads(csv_rows[1]["decision_counts"])["REVIEW"] == 1
    assert json.loads(csv_rows[1]["decision_reason_codes"])["evaluation_unavailable"] == 1


def test_legacy_reports_preserve_exact_unconfigured_shape(report_factory, tmp_path):
    row = report_factory()
    dataset = write_reports([row], tmp_path / "reports", protected_inputs=[], provenance={})
    assert "decision_summary" not in dataset["summary"]
    assert "decision_summary" not in dataset["summary"]["roles"]["span"]
    assert "decisions" not in dataset["aggregation_policy"]
    with (tmp_path / "reports" / "stems.csv").open(encoding="utf-8", newline="") as stream:
        csv_row = next(csv.DictReader(stream))
    assert csv_row["decision"] == ""
    assert "decision_status" not in csv_row


def test_disabled_policy_counts_unassigned_and_preserves_metric_contributors(report_factory):
    config = FilterConfig()
    rows = [
        attach_decision(report_factory("ready"), config),
        attach_decision(report_factory("missing", available=False), config),
    ]
    result = summarize_records(rows)
    assert result["decision"] is None
    assert result["decision_summary"]["counts"] == {
        "PASS": 0,
        "REVIEW": 0,
        "FAIL": 0,
        "UNSUPPORTED": 0,
        "unassigned": 2,
    }
    assert result["decision_summary"]["statuses"] == {"disabled": 2}
    assert result["decision_summary"]["config"] == config.to_dict()
    assert result["roles"]["span"]["decision_summary"] == result["decision_summary"]
    assert result["roles"]["span"]["metrics"]["temporal_iou"] == {
        "mean": 1,
        "contributing_count": 1,
    }


def test_disabled_audit_is_exported_to_csv_with_null_label(report_factory, tmp_path):
    row = attach_decision(report_factory(), FilterConfig())
    dataset = write_reports([row], tmp_path, protected_inputs=[], provenance={})
    with (tmp_path / "stems.csv").open(encoding="utf-8", newline="") as stream:
        csv_row = next(csv.DictReader(stream))
    assert csv_row["decision"] == ""
    assert csv_row["decision_status"] == "disabled"
    assert json.loads(csv_row["decision_reason_codes"]) == row["decision_details"]["reason_codes"]
    assert json.loads(csv_row["decision_checks"]) == row["decision_details"]["checks"]
    assert json.loads(csv_row["decision_config"]) == row["decision_details"]["config"]
    assert csv_row["decision_implementation_version"] == "1"
    with (tmp_path / "clips.csv").open(encoding="utf-8", newline="") as stream:
        clip_row = next(csv.DictReader(stream))
    assert clip_row["decision"] == ""
    assert json.loads(clip_row["decision_counts"])["unassigned"] == 1
    assert json.loads(clip_row["span_decision_counts"])["unassigned"] == 1
    assert dataset["decision"] is None
    assert dataset["clips"][0]["summary"]["decision"] is None


@pytest.mark.parametrize("label", ["PASS", "UNSUPPORTED", "made up", True, 1])
def test_arbitrary_labels_cannot_enter_counts_without_audit(report_factory, label):
    row = report_factory()
    row["decision"] = label
    with pytest.raises(ValueError, match="requires decision_details"):
        summarize_records([row])


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision", "PASS"),
        ("status", "decided"),
        ("reason_codes", ["invented"]),
        ("checks", [{"fake": True}]),
        ("implementation_version", True),
    ],
)
def test_tampered_audits_are_rejected_before_any_output(report_factory, tmp_path, field, value):
    row = attach_decision(report_factory(), FilterConfig())
    row["decision_details"][field] = value
    output = tmp_path / "reports"
    with pytest.raises(ValueError, match="Decision"):
        write_reports([row], output, protected_inputs=[], provenance={})
    assert not output.exists()


@pytest.mark.parametrize("kind", ["label", "policy", "file_provenance", "missing_audit"])
def test_audit_must_bind_to_report_decision_and_policy_provenance(report_factory, kind):
    row = attach_decision(report_factory(), FilterConfig())
    if kind == "label":
        row["decision"] = "PASS"
    elif kind == "policy":
        row["provenance"]["filter_config"] = {}
    elif kind == "file_provenance":
        row["provenance"]["filter_config_provenance"] = {"sha256": "other"}
    else:
        del row["decision_details"]
    with pytest.raises(ValueError, match="decision|Decision"):
        summarize_records([row])


def test_disabled_and_legacy_rows_remain_unassigned_without_fake_policy_context(report_factory):
    rows = [report_factory("legacy"), attach_decision(report_factory("disabled"), FilterConfig())]
    result = summarize_records(rows)["decision_summary"]
    assert result["counts"]["unassigned"] == 2
    assert result["config"] is None
    assert result["implementation_version"] is None
    assert result["statuses"] == {"disabled": 1, "legacy_unconfigured": 1}


def test_archived_policy_file_is_not_reopened_when_reading_audits(report_factory, tmp_path):
    path = tmp_path / "filter.json"
    path.write_text(json.dumps({"schema_version": 1, "filter": {}}), encoding="utf-8")
    config = FilterConfig.load(path)
    row = attach_decision(report_factory(), config)
    path.unlink()
    summary = summarize_records([row])["decision_summary"]
    assert summary["config"] == config.to_dict()
    assert summary["counts"]["unassigned"] == 1
