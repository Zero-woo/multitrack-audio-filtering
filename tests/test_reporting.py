"""Unavailable rows cannot improve aggregate scores or masquerade as zero errors."""

import csv
import hashlib
import json

import pytest

from waves_sed.reporting import summarize_records, write_reports


def record(stem_id, role="onset", metrics=None, evaluated=True, clip="clip:one"):
    return {
        "schema_version": 1,
        "stem_id": stem_id,
        "candidate_id": stem_id,
        "clip_key": clip,
        "source_description": "test source",
        "role": role,
        "sed_backend": "synthetic",
        "mapping_status": "supported",
        "reference": {"origin": "waves_pass1_planned", "status": "planned", "usable": True},
        "expected_intervals": [[0, 1]],
        "detected_intervals": [[0, 1]] if evaluated else None,
        "evaluation_status": "evaluated" if evaluated else "unavailable",
        "detection_status": "available" if evaluated else "unavailable",
        "reason_codes": [] if evaluated else ["missing_prediction"],
        "metrics": metrics,
        "decision": None,
    }


def onset(expected, detected, matched, error):
    return {
        "expected_event_count": expected,
        "detected_event_count": detected,
        "matched_event_count": matched,
        "missing_event_count": expected - matched,
        "extra_event_count": detected - matched,
        "event_recall": matched / expected if expected else None,
        "event_precision": matched / detected if detected else None,
        "mean_onset_error_ms": error,
        "median_onset_error_ms": error,
        "matches": [],
    }


def test_summary_separates_roles_uses_numeric_denominators_and_pools_onset_counts():
    rows = [
        record("one", metrics=onset(1, 1, 1, 20)),
        record("two", metrics=onset(3, 1, 0, None)),
        record("missing", evaluated=False),
        record("span", role="span", metrics={"temporal_iou": 0.75}),
    ]
    result = summarize_records(rows)
    assert result["stem_count"] == 4
    assert result["evaluation_statuses"] == {"evaluated": 3, "unavailable": 1}
    role = result["roles"]["onset"]
    assert role["evaluated_count"] == 2
    assert role["unavailable_count"] == 1
    assert role["metrics"]["mean_onset_error_ms"] == {"mean": 20, "contributing_count": 1}
    assert role["metrics"]["event_recall"] == {"mean": 0.5, "contributing_count": 2}
    assert "matches" not in role["metrics"]
    assert "temporal_iou" not in role["metrics"]
    assert role["onset_micro"]["expected_event_count"] == 4
    assert role["onset_micro"]["matched_event_count"] == 1
    assert role["onset_micro"]["event_recall"] == 0.25
    assert role["onset_micro"]["event_precision"] == 0.5
    assert result["roles"]["span"]["metrics"]["temporal_iou"]["mean"] == 0.75
    assert result["decision"] is None


def test_all_unavailable_keeps_metric_means_and_micro_counts_null():
    result = summarize_records([record("missing", evaluated=False)])
    role = result["roles"]["onset"]
    assert role["metrics"]["missing_event_count"] == {"mean": None, "contributing_count": 0}
    assert role["onset_micro"]["expected_event_count"] is None
    assert role["onset_micro"]["missing_event_count"] is None
    assert role["onset_micro"]["event_recall"] is None
    assert role["onset_micro"]["contributing_count"] == 0


def test_missing_status_is_counted_without_sorting_null_against_strings():
    missing = record("missing", evaluated=False)
    missing["reference"]["status"] = None
    result = summarize_records([record("ready", metrics=onset(1, 1, 1, 0)), missing])
    assert result["reference_statuses"] == {"planned": 1, "unspecified": 1}


@pytest.mark.parametrize(
    "field", ["config", "version", "backend", "reference", "mapping", "checkpoint"]
)
def test_aggregates_reject_different_evaluation_contexts(field):
    first = record("first", metrics=onset(1, 1, 1, 0))
    second = record("second", metrics=onset(1, 1, 1, 0))
    second["provenance"] = {}
    if field == "config":
        second["provenance"]["evaluation_config"] = {"event": {"threshold": 0.9}}
    elif field == "version":
        second["provenance"]["metric_implementation_version"] = 2
    elif field == "backend":
        second["sed_backend"] = "different backend"
    elif field == "reference":
        second["reference"]["origin"] = "external video reference"
    elif field == "mapping":
        second["provenance"]["mapping"] = {"sha256": "different mapping"}
    else:
        second["provenance"]["prediction"] = {"metadata": {"checkpoint_sha256": "different"}}
    with pytest.raises(ValueError, match="Cannot aggregate"):
        summarize_records([first, second])


def test_unavailable_records_do_not_create_aggregate_context_conflicts():
    first = record("first", metrics=onset(1, 1, 1, 0))
    missing = record("missing", evaluated=False)
    missing["sed_backend"] = "different backend"
    result = summarize_records([first, missing])
    assert result["aggregation_context"]["sed_backend"] == "synthetic"
    assert result["roles"]["onset"]["evaluated_count"] == 1


def test_genuinely_empty_evaluated_onsets_have_zero_counts_and_null_ratios():
    role = summarize_records([record("empty", metrics=onset(0, 0, 0, None))])["roles"]["onset"]
    assert role["onset_micro"]["expected_event_count"] == 0
    assert role["onset_micro"]["event_recall"] is None
    assert role["onset_micro"]["event_precision"] is None
    assert role["onset_micro"]["contributing_count"] == 1


def test_reports_include_portable_hash_indexes_and_consistent_json_csv(tmp_path):
    rows = [record("clip:one::source/with\\invalid?chars", metrics=onset(2, 1, 1, 0))]
    dataset = write_reports(rows, tmp_path, protected_inputs=[], provenance={"inference": False})
    stem_id = rows[0]["stem_id"]
    expected_file = f"stems/{hashlib.sha256(stem_id.encode()).hexdigest()}.json"
    assert dataset["files"]["stems"][stem_id] == expected_file
    assert json.loads((tmp_path / expected_file).read_text(encoding="utf-8")) == rows[0]
    assert json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8")) == dataset
    clip_file = dataset["files"]["clips"][rows[0]["clip_key"]]
    assert json.loads((tmp_path / clip_file).read_text(encoding="utf-8"))["stems"] == rows
    with (tmp_path / "stems.csv").open(encoding="utf-8", newline="") as stream:
        csv_row = next(csv.DictReader(stream))
    assert csv_row["stem_id"] == stem_id
    assert csv_row["metric_matched_event_count"] == "1"
    assert csv_row["decision"] == ""
    with (tmp_path / "clips.csv").open(encoding="utf-8", newline="") as stream:
        clip_row = next(csv.DictReader(stream))
    assert clip_row["onset_micro_event_recall"] == "0.5"


def test_preflight_all_outputs_before_any_report_mutation(tmp_path):
    protected = tmp_path / "clips.csv"
    protected.write_bytes(b"original source")
    with pytest.raises(ValueError, match="must not overwrite"):
        write_reports([record("one")], tmp_path, protected_inputs=[protected], provenance={})
    assert protected.read_bytes() == b"original source"
    assert not (tmp_path / "dataset.json").exists()
    assert not (tmp_path / "stems").exists()


def test_preflight_detects_hardlinks_between_two_output_files(tmp_path):
    first = tmp_path / "stems.csv"
    second = tmp_path / "clips.csv"
    first.write_bytes(b"original existing report")
    try:
        second.hardlink_to(first)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    with pytest.raises(ValueError, match="must not alias"):
        write_reports([record("one")], tmp_path, protected_inputs=[], provenance={})
    assert first.read_bytes() == second.read_bytes() == b"original existing report"
    assert not (tmp_path / "stems").exists()


def test_rerun_keeps_stale_files_but_index_only_lists_current_records(tmp_path):
    first = write_reports([record("old")], tmp_path, protected_inputs=[], provenance={})
    old_file = tmp_path / first["files"]["stems"]["old"]
    second = write_reports([], tmp_path, protected_inputs=[], provenance={})
    assert old_file.exists()
    assert second["files"]["stems"] == {}
    assert second["summary"]["stem_count"] == 0
    assert (tmp_path / "stems.csv").read_text().startswith("stem_id,clip_key,role")
