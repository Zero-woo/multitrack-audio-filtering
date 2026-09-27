"""WAVES plans retain their limitations and cannot silently become trusted labels."""

from dataclasses import replace

import pytest

from waves_sed.metadata import StemMetadata
from waves_sed.temporal_reference import (
    ExternalVideoReference,
    TemporalReference,
    TemporalSupport,
    WavesPlannedReference,
)


@pytest.fixture
def stem():
    return StemMetadata(
        stem_id="clip::candidate",
        candidate_id="candidate",
        clip_key="clip",
        audio_path=None,
        source_description="Synthetic source",
        role="onset",
        expected_intervals=((0.0, 0.4), (0.3, 0.6)),
        provenance={
            "reference_origin": "waves_pass1_planned",
            "reference_status": "planned",
            "support_policy": "selected_parent_only",
            "inputs": {"metadata": {"sha256": "test-only"}},
        },
    )


def test_planned_support_preserves_overlaps_order_and_provenance(stem):
    support = WavesPlannedReference().resolve(stem)

    assert support.usable
    assert support.reason is None
    assert support.origin == "waves_pass1_planned"
    assert support.status == "planned"
    assert support.intervals == stem.expected_intervals
    assert support.to_dict()["intervals"] == [[0.0, 0.4], [0.3, 0.6]]
    assert support.provenance["stem_provenance"] == stem.provenance
    stem.provenance["inputs"]["metadata"]["sha256"] = "changed"
    assert support.provenance["stem_provenance"]["inputs"]["metadata"]["sha256"] == "test-only"


def test_ambiguous_support_requires_opt_in_and_keeps_ambiguous_status(stem):
    ambiguous = replace(
        stem,
        issues=("relabelled", "merged_support_parent_only"),
        provenance={**stem.provenance, "reference_status": "ambiguous"},
    )

    blocked = WavesPlannedReference().resolve(ambiguous)
    allowed = WavesPlannedReference().resolve(ambiguous, allow_ambiguous=True)

    assert not blocked.usable
    assert blocked.reason == "ambiguous_reference"
    assert allowed.usable
    assert allowed.status == "ambiguous"
    assert allowed.intervals == stem.expected_intervals
    assert allowed.provenance["stem_issues"] == ["relabelled", "merged_support_parent_only"]
    assert allowed.provenance["allow_ambiguous_reference"] is True


@pytest.mark.parametrize("allow_ambiguous", [False, True])
@pytest.mark.parametrize("status", ["planned", "ambiguous", "missing"])
def test_missing_support_is_never_usable(stem, status, allow_ambiguous):
    missing = replace(
        stem,
        expected_intervals=None,
        provenance={**stem.provenance, "reference_status": status},
    )
    support = WavesPlannedReference().resolve(missing, allow_ambiguous=allow_ambiguous)
    assert not support.usable
    assert support.reason == "missing_support"
    assert support.intervals is None
    assert support.status == status


def test_status_missing_is_unusable_even_if_intervals_are_present(stem):
    marked_missing = replace(stem, provenance={**stem.provenance, "reference_status": "missing"})
    support = WavesPlannedReference().resolve(marked_missing, allow_ambiguous=True)
    assert not support.usable
    assert support.reason == "missing_support"
    assert support.intervals == stem.expected_intervals


@pytest.mark.parametrize("status", ["planned", "ambiguous"])
def test_empty_support_does_not_mean_expected_absence(stem, status):
    empty = replace(
        stem,
        expected_intervals=(),
        provenance={**stem.provenance, "reference_status": status},
    )
    support = WavesPlannedReference().resolve(empty, allow_ambiguous=True)
    assert not support.usable
    assert support.reason == "empty_support"
    assert support.to_dict()["intervals"] == []


@pytest.mark.parametrize("origin", [None, "", "external_video", "waves"])
def test_unknown_origin_is_preserved_and_unusable(stem, origin):
    support = WavesPlannedReference().resolve(
        replace(stem, provenance={**stem.provenance, "reference_origin": origin}),
        allow_ambiguous=True,
    )
    assert not support.usable
    assert support.reason == "unknown_reference_origin"
    assert support.origin == origin


@pytest.mark.parametrize("status", [None, "", "trusted", "ground_truth"])
def test_unknown_status_is_preserved_and_unusable(stem, status):
    support = WavesPlannedReference().resolve(
        replace(stem, provenance={**stem.provenance, "reference_status": status}),
        allow_ambiguous=True,
    )
    assert not support.usable
    assert support.reason == "unknown_reference_status"
    assert support.status == status


def test_no_reference_is_inferred_from_intervals_alone(stem):
    support = WavesPlannedReference().resolve(replace(stem, provenance={}))
    assert not support.usable
    assert support.origin is None
    assert support.status is None
    assert support.intervals == stem.expected_intervals


@pytest.mark.parametrize(
    "intervals",
    [
        "0,1",
        ((0.1,),),
        ((0.1, 0.2, 0.3),),
        ((0.1, "0.2"),),
        ((True, 2.0),),
        ((0.0, float("inf")),),
        ((float("nan"), 1.0),),
        ((0, 10**400),),
        ((-0.1, 0.2),),
        ((0.2, 0.2),),
        ((0.2, 0.1),),
        ((0.2, 0.4), (0.1, 0.3)),
    ],
)
def test_invalid_intervals_rejected(stem, intervals):
    with pytest.raises(ValueError, match="Expected support"):
        WavesPlannedReference().resolve(replace(stem, expected_intervals=intervals))


@pytest.mark.parametrize("allow", [1, "true", None])
def test_ambiguous_opt_in_must_be_boolean(stem, allow):
    with pytest.raises(ValueError, match="allow_ambiguous"):
        WavesPlannedReference().resolve(stem, allow_ambiguous=allow)


def test_protocol_accepts_provider_and_does_not_instantiate_video_detector():
    assert isinstance(WavesPlannedReference(), TemporalReference)
    with pytest.raises(TypeError):
        ExternalVideoReference()


@pytest.mark.parametrize("allow_ambiguous", [False, True])
@pytest.mark.parametrize(
    "issue", ["relabelled", "role_changed", "merged_support_parent_only", "decision_mismatch"]
)
def test_planned_status_cannot_override_known_ambiguity_issue(stem, issue, allow_ambiguous):
    support = WavesPlannedReference().resolve(
        replace(stem, issues=(issue,)), allow_ambiguous=allow_ambiguous
    )
    assert not support.usable
    assert support.reason == "inconsistent_reference_status"
    assert support.status == "planned"
    assert support.provenance["ambiguity_signals"] == [issue]


@pytest.mark.parametrize(
    "change",
    [
        {"merged_candidate_ids": ("child",)},
        {"planned_role": "span"},
        {"planned_label": "Different source"},
    ],
)
def test_planned_status_cannot_override_ambiguous_metadata_without_issues(stem, change):
    support = WavesPlannedReference().resolve(replace(stem, **change), allow_ambiguous=True)
    assert not support.usable
    assert support.reason == "inconsistent_reference_status"
    assert support.provenance["ambiguity_signals"]


def test_non_temporal_issue_does_not_disable_planned_support(stem):
    support = WavesPlannedReference().resolve(replace(stem, issues=("audio_file_not_found",)))
    assert support.usable
    assert support.provenance["stem_issues"] == ["audio_file_not_found"]


@pytest.mark.parametrize("intervals", [(), ((0.2, 0.5),)])
def test_custom_provider_can_supply_independent_video_annotations(stem, intervals):
    class SyntheticVideoAnnotations:
        def resolve(self, stem, *, allow_ambiguous=False):
            return TemporalSupport(
                origin="external_video",
                status="annotated",
                intervals=intervals,
                usable=True,
                reason=None,
                provenance={"video_id": "synthetic-video", "annotation_revision": "fixture-1"},
            )

    provider = SyntheticVideoAnnotations()
    assert isinstance(provider, TemporalReference)
    result = provider.resolve(stem)
    assert result.origin == "external_video"
    assert result.status == "annotated"
    assert result.usable
    assert result.intervals == intervals


@pytest.mark.parametrize(
    "change",
    [
        {"usable": "yes"},
        {"origin": None},
        {"status": None},
        {"reason": "not_reliable"},
        {"intervals": None},
        {"intervals": ((0.2, 0.1),)},
        {"provenance": None},
        {"provenance": {"confidence": float("nan")}},
    ],
)
def test_custom_support_rejects_inconsistent_or_unserializable_payload(change):
    value = {
        "origin": "external_video",
        "status": "annotated",
        "intervals": ((0.2, 0.5),),
        "usable": True,
        "reason": None,
        "provenance": {},
    }
    with pytest.raises(ValueError):
        TemporalSupport(**{**value, **change})
