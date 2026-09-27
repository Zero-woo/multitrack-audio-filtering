import copy
import hashlib
import json
from pathlib import Path

import pytest

from waves_sed.adapters import load_frozen, load_materialized

FIXTURES = Path(__file__).parent / "fixtures" / "waves"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def frozen(tmp_path):
    finals = json.loads((FIXTURES / "frozen_finals.json").read_text(encoding="utf-8"))
    reports = json.loads((FIXTURES / "frozen_reports.json").read_text(encoding="utf-8"))
    return (
        write_json(tmp_path / "finals.json", finals),
        write_json(tmp_path / "reports.json", reports),
        finals,
        reports,
    )


@pytest.fixture
def materialized(tmp_path):
    # Unchanged final semantics make the planned-support status easy to isolate.
    finals = json.loads((FIXTURES / "frozen_finals.json").read_text(encoding="utf-8"))
    reports = json.loads((FIXTURES / "frozen_reports.json").read_text(encoding="utf-8"))
    final = next(clip for clip in finals if clip["key"] == "fb5k_1354")
    report = next(clip for clip in reports if clip["key"] == final["key"])
    final_stem = final["final_stems"][0]
    metadata = {
        "key": final["key"],
        "title": report["title"],
        "mixed_audio": "unavailable_original.wav",
        "stems": [
            {
                **final_stem,
                "file": "stem_01.wav",
                "sha256": "declared-output-hash",
                "source_attempt_file": "original_attempt.wav",
                "source_attempt_sha256": "declared-source-hash",
            }
        ],
        "clip_summary": final["clip_summary"],
    }
    manifest_clip = copy.deepcopy(report)
    manifest_clip["video_id"] = "explicit-video-id"
    for candidate in manifest_clip["candidates"]:
        candidate.pop("attempts")
    manifest = {"version": 1, "clips": [manifest_clip]}
    metadata_path = write_json(tmp_path / "final" / final["key"] / "metadata.json", metadata)
    manifest_path = write_json(tmp_path / "sam_manifest.json", manifest)
    report_path = write_json(tmp_path / "dsp_reports" / f"{final['key']}.json", report)
    # Adapter checks presence only; inference owns waveform validation.
    (metadata_path.parent / "stem_01.wav").write_bytes(b"audio placeholder")
    return metadata_path, manifest_path, report_path, metadata, manifest, report


def test_frozen_final_semantics_and_no_invented_audio(frozen):
    final_path, report_path, _, _ = frozen
    before = {path: path.read_bytes() for path in (final_path, report_path)}
    rows = load_frozen(str(final_path), str(report_path))
    assert len(rows) == 4
    assert all(row.audio_path is None and "missing_audio_path" in row.issues for row in rows)
    assert all(row.optional_video_id is None for row in rows)
    by_id = {row.candidate_id: row for row in rows}
    relabelled = by_id["fb5k_1229__01"]
    assert relabelled.source_description == "player weapon fire"
    assert relabelled.description_origin == "final_label"
    assert relabelled.planned_label == "weapon fire"
    assert relabelled.planned_description.startswith("The player character")
    assert "relabelled" in relabelled.issues
    assert relabelled.provenance["reference_status"] == "ambiguous"
    assert relabelled.stem_id == "fb5k_1229::fb5k_1229__01"
    unchanged = by_id["fb5k_1354__02"]
    assert unchanged.provenance["reference_status"] == "planned"
    assert unchanged.provenance["reference_origin"] == "waves_pass1_planned"
    assert "relabelled" not in unchanged.issues
    json.dumps([row.to_dict() for row in rows], allow_nan=False)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert (
        unchanged.provenance["inputs"]["finals"]["sha256"]
        == hashlib.sha256(before[final_path]).hexdigest()
    )


@pytest.mark.parametrize("cid", ["fb5k_3156__03", "legacy_16__02"])
def test_merge_support_is_selected_parent_only(frozen, cid):
    final_path, report_path, _, reports = frozen
    row = next(row for row in load_frozen(final_path, report_path) if row.candidate_id == cid)
    candidate = next(c for clip in reports for c in clip["candidates"] if c["candidate_id"] == cid)
    assert row.expected_intervals == tuple(tuple(i) for i in candidate["activity_intervals"])
    assert row.provenance["support_policy"] == "selected_parent_only"
    assert row.provenance["reference_status"] == "ambiguous"
    assert "merged_support_parent_only" in row.issues
    assert set(row.provenance["raw_merged_candidate_sources"]) == set(row.merged_candidate_ids)
    for child in row.provenance["raw_merged_candidate_sources"].values():
        assert child["frozen_reports"]["activity_intervals"]
    if cid == "legacy_16__02":
        assert row.planned_role == "span"
        assert row.role == "onset"
        assert "role_changed" in row.issues


def test_explicit_frozen_audio_mapping_uses_mapping_directory(frozen, tmp_path):
    final_path, report_path, _, _ = frozen
    mapping = write_json(tmp_path / "map" / "paths.json", {"fb5k_1229/01/a1": "recorded.wav"})
    actual = mapping.parent / "recorded.wav"
    actual.write_bytes(b"placeholder")
    rows = load_frozen(final_path, report_path, audio_paths=mapping)
    assert rows[0].audio_path == actual.resolve()
    assert "missing_audio_path" not in rows[0].issues
    assert "audio_file_not_found" not in rows[0].issues
    assert all(row.audio_path is None for row in rows[1:])
    assert rows[0].to_dict()["audio_path"] == str(actual.resolve())


def test_materialized_joins_explicit_sources_and_resolves_final_path(materialized):
    metadata_path, manifest_path, report_path, _, _, _ = materialized
    (row,) = load_materialized(metadata_path, sam_manifest=manifest_path, dsp_report=report_path)
    assert row.audio_path == (metadata_path.parent / "stem_01.wav").resolve()
    assert row.optional_video_id == "explicit-video-id"
    assert not row.issues
    assert row.provenance["reference_status"] == "planned"
    assert set(row.provenance["raw_candidate_sources"]) == {"sam_manifest", "dsp_report"}
    assert row.provenance["raw_final_stem"]["sha256"] == "declared-output-hash"
    assert row.provenance["raw_final_stem"]["source_attempt_file"] == "original_attempt.wav"


def test_materialized_without_auxiliary_does_not_discover_neighbors(materialized):
    metadata_path, *_ = materialized
    (row,) = load_materialized(metadata_path)
    assert row.source_description == "wizard laugh"
    assert row.expected_intervals is None
    assert row.planned_description is None
    assert row.optional_video_id is None
    assert "missing_candidate_metadata" in row.issues
    assert "missing_support" in row.issues
    assert row.provenance["reference_status"] == "missing"


@pytest.mark.parametrize(
    "field,value",
    [
        ("description", "contradictory description"),
        ("role", "onset"),
        ("activity_intervals", [[0.0, 1.0]]),
        ("label", "another label"),
    ],
)
def test_conflicting_candidate_sources_rejected(materialized, field, value):
    metadata_path, manifest_path, report_path, _, _, report = materialized
    report["candidates"][0][field] = value
    write_json(report_path, report)
    with pytest.raises(ValueError, match=f"Conflicting {field}"):
        load_materialized(metadata_path, sam_manifest=manifest_path, dsp_report=report_path)


@pytest.mark.parametrize(
    "intervals",
    [
        [[2, 1]],
        [[-1, 1]],
        [[0, 0]],
        [[0, float("inf")]],
        [[0, float("nan")]],
        [[True, 1]],
        [["0", 1]],
        [[0]],
        [[2, 3], [0, 1]],
    ],
)
def test_invalid_support_rejected_without_repair(frozen, intervals):
    final_path, report_path, _, reports = frozen
    reports[0]["candidates"][0]["activity_intervals"] = intervals
    write_json(report_path, reports)
    with pytest.raises(ValueError):
        load_frozen(final_path, report_path)


def test_overlapping_ordered_support_is_preserved(frozen):
    final_path, report_path, _, reports = frozen
    reports[0]["candidates"][0]["activity_intervals"] = [[0, 2], [1, 3]]
    write_json(report_path, reports)
    assert load_frozen(final_path, report_path)[0].expected_intervals == ((0.0, 2.0), (1.0, 3.0))


@pytest.mark.parametrize(
    "support,expected,issue",
    [
        (None, None, "missing_support"),
        ([], (), "empty_support"),
    ],
)
def test_missing_support_is_distinct_from_empty(frozen, support, expected, issue):
    final_path, report_path, _, reports = frozen
    reports[0]["candidates"][0]["activity_intervals"] = support
    write_json(report_path, reports)
    row = load_frozen(final_path, report_path)[0]
    assert row.expected_intervals == expected
    assert issue in row.issues
    assert row.provenance["reference_status"] == "missing"


@pytest.mark.parametrize("remove", ["selected_attempt", "attempts", "chosen_attempt"])
def test_missing_attempts_are_explicit(frozen, remove):
    final_path, report_path, finals, reports = frozen
    expected = "missing_selected_attempt"
    if remove == "selected_attempt":
        finals[0]["final_stems"][0].pop("selected_attempt")
    elif remove == "attempts":
        reports[0]["candidates"][0].pop("attempts")
        expected = "missing_attempt_metadata"
    else:
        reports[0]["candidates"][0]["attempts"] = []
        expected = "selected_attempt_not_found"
    write_json(final_path, finals)
    write_json(report_path, reports)
    row = load_frozen(final_path, report_path)[0]
    assert expected in row.issues
    assert row.audio_path is None
    assert row.provenance["selected_attempt"] is None


def test_missing_final_label_never_falls_back_to_plan(materialized):
    metadata_path, manifest_path, _, metadata, _, _ = materialized
    metadata["stems"][0].pop("final_label")
    write_json(metadata_path, metadata)
    (row,) = load_materialized(metadata_path, sam_manifest=manifest_path)
    assert row.source_description is None
    assert row.description_origin is None
    assert row.planned_label == "wizard laugh"
    assert "missing_final_label" in row.issues
    assert row.provenance["reference_status"] == "ambiguous"


@pytest.mark.parametrize(
    "where", ["final_clip", "final_stem", "report_clip", "candidate", "attempt"]
)
def test_duplicate_identifiers_rejected(frozen, where):
    final_path, report_path, finals, reports = frozen
    arrays = {
        "final_clip": finals,
        "final_stem": finals[0]["final_stems"],
        "report_clip": reports,
        "candidate": reports[0]["candidates"],
        "attempt": reports[0]["candidates"][0]["attempts"],
    }
    arrays[where].append(copy.deepcopy(arrays[where][0]))
    write_json(final_path, finals)
    write_json(report_path, reports)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_frozen(final_path, report_path)


def test_cross_clip_report_rejected(materialized):
    metadata_path, _, report_path, _, _, report = materialized
    report["key"] = "another_clip"
    write_json(report_path, report)
    with pytest.raises(ValueError, match="Cross-clip"):
        load_materialized(metadata_path, dsp_report=report_path)


def test_cross_clip_candidate_rejected(frozen):
    final_path, report_path, finals, _ = frozen
    finals[0]["final_stems"][0]["candidate_id"] = "fb5k_1354__02"
    write_json(final_path, finals)
    with pytest.raises(ValueError, match="Cross-clip"):
        load_frozen(final_path, report_path)


def test_missing_candidate_and_merged_child_are_explicit(frozen):
    final_path, report_path, _, reports = frozen
    reports[0]["candidates"] = []
    reports[-1]["candidates"] = [
        c for c in reports[-1]["candidates"] if c["candidate_id"] == "legacy_16__02"
    ]
    write_json(report_path, reports)
    rows = load_frozen(final_path, report_path)
    assert "missing_candidate_metadata" in rows[0].issues
    assert rows[0].expected_intervals is None
    assert "missing_merged_candidate_metadata" in rows[-1].issues
    assert rows[-1].expected_intervals is not None


def test_nonexistent_explicit_audio_is_reported(materialized):
    metadata_path, _, _, metadata, _, _ = materialized
    metadata["stems"][0]["file"] = "absent.wav"
    write_json(metadata_path, metadata)
    (row,) = load_materialized(metadata_path)
    assert row.audio_path == (metadata_path.parent / "absent.wav").resolve()
    assert "audio_file_not_found" in row.issues


def test_duplicate_json_keys_are_rejected(frozen):
    final_path, report_path, _, _ = frozen
    final_path.write_text('[{"key":"a","key":"b","final_stems":[]}]')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_frozen(final_path, report_path)


@pytest.mark.parametrize("decisions", [None, {}, "bad", [None], ["bad"]])
def test_malformed_decisions_fail_as_value_error(frozen, decisions):
    final_path, report_path, finals, _ = frozen
    finals[0]["candidate_decisions"] = decisions
    write_json(final_path, finals)
    with pytest.raises(ValueError, match="candidate[ _]decision"):
        load_frozen(final_path, report_path)


def test_final_and_decision_mismatch_is_flagged_without_changing_final(frozen):
    final_path, report_path, finals, _ = frozen
    finals[1]["candidate_decisions"][0]["selected_attempt"] = 3
    write_json(final_path, finals)
    row = load_frozen(final_path, report_path)[1]
    assert row.provenance["raw_final_stem"]["selected_attempt"] == 2
    assert row.provenance["selected_attempt"]["attempt"] == 2
    assert "decision_mismatch" in row.issues
    assert row.provenance["reference_status"] == "ambiguous"


@pytest.mark.parametrize("attempt", [0, 4, True, 1.0, "1"])
def test_invalid_selected_attempt_rejected(frozen, attempt):
    final_path, report_path, finals, _ = frozen
    finals[0]["final_stems"][0]["selected_attempt"] = attempt
    write_json(final_path, finals)
    with pytest.raises(ValueError, match="selected_attempt"):
        load_frozen(final_path, report_path)


def test_multiple_parents_cannot_claim_same_merged_child(frozen):
    final_path, report_path, finals, _ = frozen
    parent = copy.deepcopy(finals[-1]["final_stems"][0])
    parent["candidate_id"] = "legacy_16__03"
    finals[-1]["final_stems"].append(parent)
    write_json(final_path, finals)
    with pytest.raises(ValueError, match="multiple final parents"):
        load_frozen(final_path, report_path)
