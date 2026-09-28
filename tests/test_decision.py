"""Explicit policy boundaries never convert unknown evidence into quality failures."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from waves_sed.decision import FilterConfig, FilterDecision, decide


def config(role="span", rule="min_tiou", value=None):
    return FilterConfig.from_dict({"schema_version": 1, "filter": {role: {rule: value}}})


def report(role="span", metrics=None):
    return {
        "role": role,
        "mapping_status": "supported",
        "reference": {
            "origin": "waves_pass1_planned",
            "status": "planned",
            "intervals": [[0, 1]],
            "usable": True,
            "reason": None,
            "provenance": {},
        },
        "evaluation_status": "evaluated",
        "detection_status": "available",
        "cache_validation": {"verified": True},
        "metrics": {"temporal_iou": 1.0} if metrics is None else metrics,
    }


def test_disabled_default_and_example_do_not_invent_thresholds():
    default = FilterConfig()
    example = FilterConfig.load(Path(__file__).parents[1] / "configs/filter.example.json")
    assert not default.enabled
    assert default.to_dict() == example.to_dict()
    assert all(
        bound is None for role in default.to_dict()["filter"].values() for bound in role.values()
    )
    for row in ({}, report(), report(role="unknown")):
        audit = decide(row, default)
        assert audit["decision"] is None
        assert audit["status"] == "disabled"
        assert audit["checks"] == []
        assert audit["implementation_version"] == 1
    assert {member.value for member in FilterDecision} == {"PASS", "REVIEW", "FAIL", "UNSUPPORTED"}


def test_partial_config_normalizes_and_does_not_share_mutable_state():
    value = {"schema_version": 1, "filter": {"span": {"min_tiou": 0.6}}}
    loaded = FilterConfig.from_dict(value)
    value["filter"]["span"]["min_tiou"] = 0.2
    normalized = loaded.to_dict()
    assert normalized["filter"]["span"]["min_tiou"] == {"pass": 0.6, "fail": 0.6}
    assert normalized["filter"]["onset"]["min_event_recall"] is None
    normalized["filter"]["span"]["min_tiou"]["pass"] = 1
    assert loaded.to_dict()["filter"]["span"]["min_tiou"]["pass"] == 0.6
    assert loaded.provenance == {}
    assert FilterConfig.from_dict(loaded.to_dict()).to_dict() == loaded.to_dict()


def test_load_binds_exact_bytes_and_normalizes_bom(tmp_path):
    path = tmp_path / "policy.json"
    payload = '\ufeff{"schema_version":1,"filter":{"span":{"min_tiou":0.6}}}'.encode("utf-8")
    path.write_bytes(payload)
    loaded = FilterConfig.load(path)
    assert loaded.provenance == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    provenance = loaded.provenance
    provenance["sha256"] = "changed"
    assert loaded.provenance["sha256"] == hashlib.sha256(payload).hexdigest()
    audit = decide(report(), loaded)
    assert audit["config_provenance"] == loaded.provenance
    audit["config_provenance"]["path"] = "changed"
    assert loaded.provenance["path"] == str(path.resolve())


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "FAIL"), (0.499, "FAIL"), (0.5, "REVIEW"), (0.799, "REVIEW"), (0.8, "PASS"), (1, "PASS")],
)
def test_minimum_band_boundaries(value, expected):
    audit = decide(
        report(metrics={"temporal_iou": value}), config(value={"pass": 0.8, "fail": 0.5})
    )
    assert audit["decision"] == expected
    assert audit["status"] == "decided"
    assert audit["checks"][0]["decision"] == expected
    assert audit["checks"][0]["status"] == "checked"
    assert audit["checks"][0]["direction"] == "min"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "PASS"), (100, "PASS"), (100.001, "REVIEW"), (200, "REVIEW"), (200.001, "FAIL")],
)
def test_maximum_band_boundaries(value, expected):
    policy = config("onset", "max_mean_onset_error_ms", {"pass": 100, "fail": 200})
    audit = decide(report("onset", {"mean_onset_error_ms": value}), policy)
    assert audit["decision"] == expected
    assert audit["checks"][0]["direction"] == "max"


@pytest.mark.parametrize(
    ("role", "rule", "metric", "threshold", "value", "expected"),
    [
        ("onset", "min_event_recall", "event_recall", 0.8, 0.8, "PASS"),
        ("onset", "min_event_recall", "event_recall", 0.8, 0.799, "FAIL"),
        ("onset", "max_mean_onset_error_ms", "mean_onset_error_ms", 100, 100, "PASS"),
        ("onset", "max_mean_onset_error_ms", "mean_onset_error_ms", 100, 100.001, "FAIL"),
        ("span", "min_tiou", "temporal_iou", 0.5, 0.5, "PASS"),
        ("span", "min_tiou", "temporal_iou", 0.5, 0.499, "FAIL"),
        ("ambience", "min_occupancy", "occupancy_in_expected_span", 0.7, 0.7, "PASS"),
        ("ambience", "min_occupancy", "occupancy_in_expected_span", 0.7, 0.699, "FAIL"),
    ],
)
def test_single_bound_rules_and_metric_mapping(role, rule, metric, threshold, value, expected):
    audit = decide(report(role, {metric: value}), config(role, rule, threshold))
    assert audit["decision"] == expected
    assert audit["checks"][0]["metric"] == metric


def test_unconfigured_known_role_is_distinct_from_unsupported_role():
    policy = config(value=0.8)
    inactive = decide(report("ambience"), policy)
    assert inactive["status"] == "not_configured"
    assert inactive["decision"] is None
    assert inactive["checks"] == []
    for role in (None, "impact", "SPAN", [], 1):
        unsupported = decide(report(role), policy)
        assert unsupported["decision"] == "UNSUPPORTED"
        assert unsupported["reason_codes"] == ["unsupported_role"]


@pytest.mark.parametrize("status", ["unsupported", "missing", None, "ambiguous"])
def test_unsupported_mapping_does_not_produce_fail(status):
    row = report(metrics={"temporal_iou": 0})
    row["mapping_status"] = status
    audit = decide(row, config(value=0.8))
    assert audit["decision"] == "UNSUPPORTED"
    assert audit["reason_codes"] == ["unsupported_mapping"]
    assert audit["checks"][0]["status"] == "blocked"


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("reference",), None, "missing_reference"),
        (("reference", "origin"), None, "missing_reference"),
        (("reference", "origin"), " ", "missing_reference"),
        (("reference", "status"), None, "missing_reference"),
        (("reference", "status"), [], "missing_reference"),
        (("reference", "intervals"), None, "missing_reference"),
        (("reference", "usable"), False, "unusable_reference"),
        (("reference", "usable"), 1, "unusable_reference"),
        (("reference", "reason"), "inconsistent_reference_status", "unusable_reference"),
        (("reference", "status"), "ambiguous", "ambiguous_reference"),
        (("reference", "status"), "missing", "unusable_reference"),
        (("reference", "provenance"), {"ambiguity_signals": ["relabelled"]}, "ambiguous_reference"),
        (("evaluation_status",), "unavailable", "evaluation_unavailable"),
        (("detection_status",), "unavailable", "detection_unavailable"),
        (("cache_validation",), None, "unverified_cache"),
        (("cache_validation", "verified"), False, "unverified_cache"),
        (("cache_validation", "verified"), 1, "unverified_cache"),
    ],
)
def test_evidence_guards_block_even_otherwise_certain_fail(path, value, reason):
    row = report(metrics={"temporal_iou": 0})
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    audit = decide(row, config(value=0.8))
    assert audit["decision"] == "REVIEW"
    assert reason in audit["reason_codes"]
    assert audit["checks"][0]["status"] == "blocked"


def test_external_reference_vocabulary_and_explicit_empty_support_are_preserved():
    row = report()
    row["reference"].update(origin="independent_video_annotation", status="annotated")
    assert decide(row, config(value=0.8))["decision"] == "PASS"
    row["reference"]["intervals"] = []
    row["metrics"]["temporal_iou"] = None
    audit = decide(row, config(value=0.8))
    assert audit["decision"] == "REVIEW"
    assert audit["checks"][0]["status"] == "unavailable"


@pytest.mark.parametrize(
    "value", [None, True, False, "0.9", -0.1, 1.1, float("inf"), float("nan"), 10**400]
)
def test_bad_or_null_metric_is_review_and_audit_is_serializable(value):
    audit = decide(report(metrics={"temporal_iou": value}), config(value=0.8))
    assert audit["decision"] == "REVIEW"
    assert audit["checks"][0]["status"] == "unavailable"
    json.dumps(audit, allow_nan=False)


def test_missing_metric_is_not_zero_or_pass():
    audit = decide(report(metrics={}), config(value=0.0))
    assert audit["decision"] == "REVIEW"
    assert audit["checks"][0]["value"] is None
    assert audit["checks"][0]["reason_codes"] == ["missing_metric"]


@pytest.mark.parametrize(
    ("recall", "error", "expected"),
    [
        (1, 10, "PASS"),
        (0.7, 10, "REVIEW"),
        (1, None, "REVIEW"),
        (0, None, "FAIL"),
        (0.7, 300, "FAIL"),
    ],
)
def test_multi_rule_precedence_with_unavailable_match_error(recall, error, expected):
    policy = FilterConfig.from_dict(
        {
            "schema_version": 1,
            "filter": {
                "onset": {
                    "min_event_recall": {"pass": 0.8, "fail": 0.5},
                    "max_mean_onset_error_ms": {"pass": 100, "fail": 200},
                }
            },
        }
    )
    row = report("onset", {"event_recall": recall, "mean_onset_error_ms": error})
    audit = decide(row, policy)
    assert audit["decision"] == expected
    assert len(audit["checks"]) == 2


def test_no_report_mutation_and_semantic_evidence_has_no_automatic_effect():
    row = report()
    row["semantic_evidence"] = {"outside_allowed_family": [{"max_probability": 1.0}]}
    original = deepcopy(row)
    audit = decide(row, config(value=0.8))
    assert audit["decision"] == "PASS"
    assert row == original
    audit["checks"][0]["value"] = 0.0
    assert row == original


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"schema_version": 1},
        {"schema_version": True, "filter": {}},
        {"schema_version": 1.0, "filter": {}},
        {"schema_version": 2, "filter": {}},
        {"schema_version": 1, "filter": {}, "extra": 1},
        {"schema_version": 1, "filter": None},
        {"schema_version": 1, "filter": {"impact": {}}},
        {"schema_version": 1, "filter": {"span": None}},
        {"schema_version": 1, "filter": {"span": {"min_occupancy": 0.5}}},
    ],
)
def test_rejects_invalid_config_shapes(value):
    with pytest.raises(ValueError):
        FilterConfig.from_dict(value)


@pytest.mark.parametrize(
    "value", [True, False, "0.8", [], -0.1, 1.1, float("nan"), float("inf"), 10**400]
)
def test_rejects_invalid_ratio_thresholds(value):
    with pytest.raises(ValueError):
        config(value=value)


@pytest.mark.parametrize("value", [-1, True, "100", float("nan"), float("inf")])
def test_rejects_invalid_error_thresholds(value):
    with pytest.raises(ValueError):
        config("onset", "max_mean_onset_error_ms", value)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"pass": 0.8},
        {"fail": 0.5},
        {"pass": 0.8, "fail": 0.5, "review": 0.6},
        {"pass": None, "fail": 0.5},
        {"pass": 0.8, "fail": True},
        {"pass": 0.5, "fail": 0.8},
    ],
)
def test_rejects_incomplete_or_reversed_min_bands(value):
    with pytest.raises(ValueError):
        config(value=value)


def test_rejects_reversed_maximum_band():
    with pytest.raises(ValueError, match="pass <= fail"):
        config("onset", "max_mean_onset_error_ms", {"pass": 200, "fail": 100})


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1,"filter":{}}',
        '{"schema_version":1,"filter":{"span":{},"span":{}}}',
        '{"schema_version":1,"filter":{"span":{"min_tiou":0.1,"min_tiou":0.2}}}',
        '{"schema_version":1,"filter":{"span":{"min_tiou":{"pass":1,"pass":1,"fail":0}}}}',
        '{"schema_version":1,"filter":{"span":{"min_tiou":NaN}}}',
        '{"schema_version":1,"filter":{"span":{"min_tiou":Infinity}}}',
        '{"schema_version":1,"filter":{"span":{"min_tiou":-Infinity}}}',
        '{"schema_version":1,"filter":{"span":{"min_tiou":1e999}}}',
    ],
)
def test_load_rejects_duplicate_and_nonfinite_json(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        FilterConfig.load(path)
