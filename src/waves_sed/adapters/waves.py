"""Read WAVES materialized metadata and frozen Pass 2 artifacts.

Input JSON is never rewritten. No path or support is inferred from an audio ID,
and semantic merges retain only the selected parent's planned intervals.
"""

import hashlib
import json
import math
from pathlib import Path

from waves_sed.metadata import StemMetadata

_PLAN_FIELDS = ("label", "description", "role", "activity_intervals")
_ROLES = {"onset", "span", "ambience"}


def _absolute_media_paths(value: object) -> set[str]:
    """Collect explicit pipeline media references for output overwrite protection.

    WAVES prepare_inputs/run_sam_attempts write absolute source paths. Relative
    final stems are resolved separately from metadata; no base is guessed for
    other relative provenance paths.
    """
    paths = set()
    if isinstance(value, dict):
        for field, item in value.items():
            if field in {"file", "source_attempt_file", "mixed_audio", "video"}:
                if isinstance(item, str) and item and Path(item).is_absolute():
                    paths.add(str(Path(item).resolve()))
            paths.update(_absolute_media_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_absolute_media_paths(item))
    return paths


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _read(path: str | Path) -> tuple[object, dict]:
    path = Path(path).resolve()
    data = path.read_bytes()
    try:
        value = json.loads(
            data.decode("utf-8-sig"),
            object_pairs_hook=_object_pairs,
            parse_constant=_invalid_constant,
        )
        # Also catch numbers such as 1e999 which JSON parses as infinity.
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid WAVES JSON {path}: {error}") from error
    return value, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def _object(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _array(value: object, context: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _text(value: object, context: str, *, optional: bool = False) -> str | None:
    if optional and (value is None or value == ""):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _role(value: object, context: str) -> str | None:
    value = _text(value, context, optional=True)
    if value is not None and value not in _ROLES:
        raise ValueError(f"{context} must be onset, span, or ambience")
    return value


def _intervals(value: object, context: str) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    result = []
    previous_start = -1.0
    for interval in _array(value, context):
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{context}: intervals must be [start, end] pairs")
        start, end = interval
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in interval):
            raise ValueError(f"{context}: interval bounds must be real numbers")
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError(f"{context}: require finite 0 <= start < end")
        if start < previous_start:
            raise ValueError(f"{context}: intervals must be ordered by start time")
        result.append((float(start), float(end)))
        previous_start = start
    return tuple(result)


def _candidate_id(value: object, key: str) -> str:
    value = _text(value, f"{key}: candidate_id")
    # WAVES generates <clip_key>__<index>. Check this when the separator is present;
    # IDs without it remain valid opaque IDs and are checked against clip indexes.
    if "__" in value and value.rsplit("__", 1)[0] != key:
        raise ValueError(f"Cross-clip candidate_id {value!r} in clip {key!r}")
    return value


def _candidate_index(clips: list, context: str) -> tuple[dict, dict]:
    index, owners = {}, {}
    for raw_clip in clips:
        clip = _object(raw_clip, context)
        key = _text(clip.get("key"), f"{context}: key")
        if key in index:
            raise ValueError(f"Duplicate clip key {key!r} in {context}")
        candidates = {}
        for raw_candidate in _array(clip.get("candidates", []), f"{key}: candidates"):
            candidate = _object(raw_candidate, f"{key}: candidate")
            cid = _candidate_id(candidate.get("candidate_id"), key)
            if cid in owners:
                raise ValueError(f"Duplicate candidate_id {cid!r} in {context}")
            owners[cid] = key
            for field in ("label", "description"):
                _text(candidate.get(field), f"{cid}: {field}", optional=True)
            _role(candidate.get("role"), f"{cid}: role")
            _intervals(candidate.get("activity_intervals"), f"{cid}: activity_intervals")
            attempts = {}
            for raw_attempt in _array(candidate.get("attempts", []), f"{cid}: attempts"):
                attempt = _object(raw_attempt, f"{cid}: attempt")
                number = attempt.get("attempt")
                if type(number) is not int or not 1 <= number <= 3:
                    raise ValueError(f"{cid}: attempt must be an integer in [1, 3]")
                if number in attempts:
                    raise ValueError(f"{cid}: duplicate attempt {number}")
                attempts[number] = attempt
            candidates[cid] = candidate
        index[key] = {"clip": clip, "candidates": candidates}
    return index, owners


def _plan_sources(key: str, cid: str, sources: dict) -> dict:
    matches = {}
    for name, (index, owners) in sources.items():
        if cid in owners and owners[cid] != key:
            raise ValueError(f"Cross-clip candidate {cid!r}: expected {key!r}, got {owners[cid]!r}")
        candidate = index.get(key, {}).get("candidates", {}).get(cid)
        if candidate is not None:
            matches[name] = candidate
    # Only shared, present fields can contradict each other. Missing fields are
    # filled from another explicitly supplied source and remain visible in raw data.
    for field in _PLAN_FIELDS:
        values = [
            candidate[field] for candidate in matches.values() if candidate.get(field) is not None
        ]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"Conflicting {field} for {key}/{cid} between candidate sources")
    return matches


def _combine_plan(matches: dict) -> dict:
    return {
        field: next((c[field] for c in matches.values() if c.get(field) is not None), None)
        for field in _PLAN_FIELDS
    }


def _selected_attempt(stem: dict, matches: dict, issues: list) -> dict | None:
    selected = stem.get("selected_attempt")
    if selected is None:
        issues.append("missing_selected_attempt")
        return None
    if type(selected) is not int or not 1 <= selected <= 3:
        raise ValueError(f"{stem['candidate_id']}: selected_attempt must be an integer in [1, 3]")
    records = []
    has_attempts = False
    for candidate in matches.values():
        if "attempts" in candidate:
            has_attempts = True
            records.extend(a for a in candidate["attempts"] if a["attempt"] == selected)
    if not records:
        issues.append("selected_attempt_not_found" if has_attempts else "missing_attempt_metadata")
        return None
    result = {}
    for record in records:
        for field, value in record.items():
            if field in result and result[field] != value:
                raise ValueError(f"{stem['candidate_id']}: conflicting selected attempt {field}")
            result[field] = value
    return result


def _video_id(clip: dict, sources: dict) -> str | None:
    values = []
    for candidate_clip in [clip] + [
        index[clip["key"]]["clip"] for index, _ in sources.values() if clip["key"] in index
    ]:
        value = _text(candidate_clip.get("video_id"), "video_id", optional=True)
        if value is not None:
            values.append(value)
    if len(set(values)) > 1:
        raise ValueError(f"Conflicting video_id for {clip['key']}")
    return values[0] if values else None


def _audio_path(value: object, base: Path, issues: list) -> Path | None:
    value = _text(value, "audio path", optional=True)
    if value is None:
        issues.append("missing_audio_path")
        return None
    path = Path(value)
    path = (base / path).resolve()
    if not path.is_file():
        issues.append("audio_file_not_found")
    return path


def _build_stems(
    clip: dict,
    rows: list,
    sources: dict,
    documents: dict,
    *,
    mode: str,
    audio_base: Path,
    audio_mapping: dict | None = None,
) -> list:
    key = _text(clip.get("key"), "final clip key")
    video_id = _video_id(clip, sources)
    media_paths = _absolute_media_paths(clip)
    for index, _ in sources.values():
        if key in index:
            media_paths.update(_absolute_media_paths(index[key]["clip"]))
    decisions = {}
    for value in _array(clip.get("candidate_decisions", []), f"{key}: candidate_decisions"):
        decision = _object(value, f"{key}: candidate decision")
        cid = _candidate_id(decision.get("candidate_id"), key)
        if cid in decisions:
            raise ValueError(f"Duplicate candidate decision {cid!r}")
        decisions[cid] = decision
    results, seen = [], set()
    merge_owners = {}
    for raw_stem in rows:
        stem = _object(raw_stem, f"{key}: final stem")
        cid = _candidate_id(stem.get("candidate_id"), key)
        if cid in seen:
            raise ValueError(f"Duplicate final candidate_id {cid!r}")
        seen.add(cid)
        matches = _plan_sources(key, cid, sources)
        plan = _combine_plan(matches)
        issues = []
        if not matches:
            issues.append("missing_candidate_metadata")
        label = _text(stem.get("final_label"), f"{cid}: final_label", optional=True)
        role = _role(stem.get("final_role"), f"{cid}: final_role")
        for value, issue in [
            (label, "missing_final_label"),
            (role, "missing_final_role"),
            (plan["description"], "missing_planned_description"),
            (plan["label"], "missing_planned_label"),
            (plan["role"], "missing_planned_role"),
            (video_id, "missing_video_id"),
        ]:
            if value is None:
                issues.append(issue)
        intervals = _intervals(plan["activity_intervals"], f"{cid}: activity_intervals")
        if intervals is None:
            issues.append("missing_support")
        elif not intervals:
            issues.append("empty_support")
        if label is not None and plan["label"] is not None and label != plan["label"]:
            issues.append("relabelled")
        if role is not None and plan["role"] is not None and role != plan["role"]:
            issues.append("role_changed")
        merged = tuple(
            _candidate_id(value, key)
            for value in _array(
                stem.get("merged_candidate_ids", []), f"{cid}: merged_candidate_ids"
            )
        )
        if len(set(merged)) != len(merged) or cid in merged:
            raise ValueError(f"{cid}: merged_candidate_ids must be unique and exclude parent")
        children = {}
        for child_id in merged:
            if child_id in merge_owners:
                raise ValueError(f"Merged candidate {child_id!r} belongs to multiple final parents")
            merge_owners[child_id] = cid
            children[child_id] = _plan_sources(key, child_id, sources)
            if not children[child_id]:
                issues.append("missing_merged_candidate_metadata")
        if merged:
            issues.append("merged_support_parent_only")
        decision = decisions.get(cid)
        if decision is not None and (
            decision.get("action") != "keep"
            or any(
                decision.get(field) != stem.get(field)
                for field in ("selected_attempt", "final_label", "final_role")
            )
        ):
            issues.append("decision_mismatch")
        for child_id in merged:
            decision = decisions.get(child_id)
            if decision is not None and (
                decision.get("action") != "merge_into"
                or decision.get("merge_into_candidate_id") != cid
            ):
                issues.append("decision_mismatch")
        selected = _selected_attempt(stem, matches, issues)
        if mode == "materialized":
            path = _audio_path(stem.get("file"), audio_base, issues)
        else:
            audio_id = _text(
                selected.get("audio_id") if selected else None, f"{cid}: audio_id", optional=True
            )
            if audio_id is None:
                issues.append("missing_audio_id")
            path = _audio_path((audio_mapping or {}).get(audio_id), audio_base, issues)
        ambiguous = bool(merged) or any(
            issue in issues
            for issue in (
                "relabelled",
                "role_changed",
                "missing_final_label",
                "missing_final_role",
                "missing_planned_label",
                "missing_planned_role",
                "missing_selected_attempt",
                "selected_attempt_not_found",
                "decision_mismatch",
            )
        )
        status = "missing" if not intervals else ("ambiguous" if ambiguous else "planned")
        provenance = {
            "adapter": f"waves_{mode}",
            "adapter_schema_version": 1,
            "inputs": documents,
            "source_media_paths": sorted(media_paths),
            "description_origin": "final_label" if label else None,
            "reference_origin": "waves_pass1_planned" if intervals is not None else None,
            "reference_status": status,
            "support_policy": "selected_parent_only",
            "raw_final_stem": stem,
            "raw_candidate_sources": matches,
            "raw_merged_candidate_sources": children,
            "selected_attempt": selected,
            "raw_clip_context": {
                k: v
                for k, v in clip.items()
                if k not in ("stems", "final_stems", "candidate_decisions")
            },
        }
        provenance["raw_candidate_decisions"] = [
            decisions[decision_id] for decision_id in (cid, *merged) if decision_id in decisions
        ]
        results.append(
            StemMetadata(
                stem_id=f"{key}::{cid}",
                candidate_id=cid,
                clip_key=key,
                audio_path=path,
                source_description=label,
                role=role,
                expected_intervals=intervals,
                optional_video_id=video_id,
                description_origin="final_label" if label else None,
                planned_description=plan["description"],
                planned_label=plan["label"],
                planned_role=plan["role"],
                merged_candidate_ids=merged,
                issues=tuple(dict.fromkeys(issues)),
                provenance=provenance,
            )
        )
    if set(merge_owners) & seen:
        raise ValueError("A merged candidate cannot also be a selected final parent")
    return results


def load_materialized(
    metadata_path: str | Path,
    *,
    sam_manifest: str | Path | None = None,
    dsp_report: str | Path | None = None,
) -> list[StemMetadata]:
    """Read final/<key>/metadata.json and explicitly supplied candidate sources.

    Relative ``stems[].file`` values resolve from the final metadata directory.
    Missing optional data is flagged; contradictory joins and malformed data raise
    ValueError. Supplying both SAM and DSP sources requires their shared plan fields
    to agree. No neighboring files are discovered automatically.
    """
    value, record = _read(metadata_path)
    clip = _object(value, "materialized metadata")
    key = _text(clip.get("key"), "materialized key")
    sources, documents = {}, {"metadata": record}
    if sam_manifest is not None:
        value, documents["sam_manifest"] = _read(sam_manifest)
        manifest = _object(value, "SAM manifest")
        sources["sam_manifest"] = _candidate_index(
            _array(manifest.get("clips"), "SAM clips"), "SAM manifest"
        )
        if key not in sources["sam_manifest"][0]:
            raise ValueError(f"Clip {key!r} is absent from SAM manifest")
    if dsp_report is not None:
        value, documents["dsp_report"] = _read(dsp_report)
        report = _object(value, "DSP report")
        if report.get("key") != key:
            raise ValueError(f"Cross-clip DSP report: expected {key!r}, got {report.get('key')!r}")
        sources["dsp_report"] = _candidate_index([report], "DSP report")
    return _build_stems(
        clip,
        _array(clip.get("stems"), "stems"),
        sources,
        documents,
        mode="materialized",
        audio_base=Path(record["path"]).parent,
    )


def load_frozen(
    finals_path: str | Path, reports_path: str | Path, *, audio_paths: str | Path | None = None
) -> list[StemMetadata]:
    """Read frozen final/report arrays, optionally joining explicit audio_id paths.

    ``audio_paths`` is a JSON object {audio_id: path}; relative paths resolve from
    that JSON's directory. Frozen IDs are never converted into inferred filenames.
    """
    finals, final_document = _read(finals_path)
    reports, report_document = _read(reports_path)
    sources = {"frozen_reports": _candidate_index(_array(reports, "frozen reports"), "reports")}
    documents = {"finals": final_document, "reports": report_document}
    mapping = {}
    audio_base = Path(final_document["path"]).parent
    if audio_paths is not None:
        value, documents["audio_paths"] = _read(audio_paths)
        mapping = _object(value, "audio_paths mapping")
        for audio_id, path in mapping.items():
            _text(audio_id, "audio_id mapping key")
            _text(path, f"{audio_id}: audio path")
        audio_base = Path(documents["audio_paths"]["path"]).parent
    results, seen_keys, seen_ids = [], set(), set()
    for value in _array(finals, "frozen finals"):
        clip = _object(value, "frozen final clip")
        key = _text(clip.get("key"), "final clip key")
        if key in seen_keys:
            raise ValueError(f"Duplicate final clip key {key!r}")
        seen_keys.add(key)
        stems = _build_stems(
            clip,
            _array(clip.get("final_stems"), "final_stems"),
            sources,
            documents,
            mode="frozen",
            audio_base=audio_base,
            audio_mapping=mapping,
        )
        for stem in stems:
            if stem.candidate_id in seen_ids:
                raise ValueError(f"Duplicate final candidate_id {stem.candidate_id!r}")
            seen_ids.add(stem.candidate_id)
        results.extend(stems)
    return results
