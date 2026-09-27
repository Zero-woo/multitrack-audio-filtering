"""Role-aware cached-evaluation reports; unavailable measurements stay unavailable."""

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

ROLE_METRICS = {
    "onset": (
        "expected_event_count",
        "detected_event_count",
        "matched_event_count",
        "missing_event_count",
        "extra_event_count",
        "event_recall",
        "event_precision",
        "mean_onset_error_ms",
        "median_onset_error_ms",
    ),
    "span": (
        "temporal_iou",
        "expected_coverage",
        "detected_precision",
        "onset_error",
        "offset_error",
        "out_of_window_activation",
        "expected_duration_seconds",
        "detected_duration_seconds",
        "intersection_duration_seconds",
        "union_duration_seconds",
    ),
    "ambience": (
        "occupancy_in_expected_span",
        "mean_target_confidence",
        "out_of_window_activity",
        "expected_duration_seconds",
        "detected_duration_seconds",
        "intersection_duration_seconds",
        "observed_expected_duration_seconds",
    ),
}


def _counts(values) -> dict:
    return dict(
        sorted(Counter("unspecified" if value is None else value for value in values).items())
    )


def _aggregation_context(records: list[dict]) -> dict | None:
    """An aggregate must describe one comparable evaluation setup.

    Missing context is retained as null, and cannot be mixed with known context.
    Individual mapping rules may differ by stem; their configuration must agree.
    """
    contexts = {}
    for row in records:
        if row["evaluation_status"] != "evaluated":
            continue
        provenance = row.get("provenance") or {}
        prediction = provenance.get("prediction") or {}
        metadata = prediction.get("metadata") or {}
        mapping = provenance.get("mapping") or {}
        context = {
            "evaluation_config": provenance.get("evaluation_config"),
            "metric_implementation_version": provenance.get("metric_implementation_version"),
            "sed_backend": row.get("sed_backend"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "preprocessing": metadata.get("preprocessing"),
            "upstream_revision": metadata.get("upstream_revision"),
            "mapping_sha256": mapping.get("sha256"),
            "reference_origin": row["reference"].get("origin"),
        }
        contexts[json.dumps(context, sort_keys=True, allow_nan=False)] = context
    if len(contexts) > 1:
        raise ValueError(
            "Cannot aggregate evaluated records with different evaluation configs, metric "
            "implementations, SED models, mappings, or reference origins; report them separately"
        )
    return next(iter(contexts.values()), None)


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_records(records: list[dict]) -> dict:
    """Means have per-metric denominators; onset micro ratios use pooled counts."""
    context = _aggregation_context(records)
    roles = {}
    for role in sorted({row.get("role") or "unspecified" for row in records}):
        rows = [row for row in records if (row.get("role") or "unspecified") == role]
        evaluated = [row for row in rows if row["evaluation_status"] == "evaluated"]
        metric_keys = sorted(
            set(ROLE_METRICS.get(role, ()))
            | {
                key
                for row in evaluated
                for key, value in (row.get("metrics") or {}).items()
                if _number(value) or value is None
            }
        )
        metrics = {}
        for key in metric_keys:
            values = [
                row["metrics"][key]
                for row in evaluated
                if _number((row.get("metrics") or {}).get(key))
            ]
            metrics[key] = {
                "mean": math.fsum(values) / len(values) if values else None,
                "contributing_count": len(values),
            }
        role_summary = {
            "stem_count": len(rows),
            "evaluated_count": len(evaluated),
            "unavailable_count": len(rows) - len(evaluated),
            "metrics": metrics,
        }
        if role == "onset":
            count_keys = (
                "expected_event_count",
                "detected_event_count",
                "matched_event_count",
                "missing_event_count",
                "extra_event_count",
            )
            counted = [
                row
                for row in evaluated
                if all(_number((row.get("metrics") or {}).get(key)) for key in count_keys)
            ]
            micro = {
                key: sum(row["metrics"][key] for row in counted) if counted else None
                for key in count_keys
            }
            micro["contributing_count"] = len(counted)
            micro["event_recall"] = (
                micro["matched_event_count"] / micro["expected_event_count"]
                if micro["expected_event_count"]
                else None
            )
            micro["event_precision"] = (
                micro["matched_event_count"] / micro["detected_event_count"]
                if micro["detected_event_count"]
                else None
            )
            role_summary["onset_micro"] = micro
        roles[role] = role_summary
    return {
        "stem_count": len(records),
        "clip_count": len({row["clip_key"] for row in records}),
        "evaluation_statuses": _counts(row["evaluation_status"] for row in records),
        "detection_statuses": _counts(row["detection_status"] for row in records),
        "mapping_statuses": _counts(row["mapping_status"] for row in records),
        "reference_statuses": _counts(row["reference"]["status"] for row in records),
        "reason_codes": _counts(reason for row in records for reason in set(row["reason_codes"])),
        "roles": roles,
        "aggregation_context": context,
        "decision": None,
    }


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _csv(rows: list[dict], leading_columns: tuple[str, ...]) -> str:
    columns = [
        *leading_columns,
        *sorted(set().union(*(row.keys() for row in rows)) - set(leading_columns)),
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def _stem_csv_row(record: dict) -> dict:
    keys = (
        "stem_id",
        "candidate_id",
        "clip_key",
        "source_description",
        "role",
        "sed_backend",
        "mapping_status",
        "evaluation_status",
        "detection_status",
        "reason_codes",
        "expected_intervals",
        "detected_intervals",
        "decision",
    )
    result = {key: record.get(key) for key in keys}
    result["reference_origin"] = record["reference"].get("origin")
    result["reference_status"] = record["reference"]["status"]
    result["reference_usable"] = record["reference"].get("usable")
    result.update(
        {
            f"metric_{key}": value
            for key, value in (record.get("metrics") or {}).items()
            if _number(value) or value is None
        }
    )
    return result


def _clip_csv_row(clip: dict) -> dict:
    summary = clip["summary"]
    result = {
        "clip_key": clip["clip_key"],
        "stem_count": summary["stem_count"],
        "evaluation_statuses": summary["evaluation_statuses"],
        "detection_statuses": summary["detection_statuses"],
        "mapping_statuses": summary["mapping_statuses"],
        "reference_statuses": summary["reference_statuses"],
        "reason_codes": summary["reason_codes"],
        "decision": None,
    }
    for role, values in summary["roles"].items():
        for key in ("stem_count", "evaluated_count", "unavailable_count"):
            result[f"{role}_{key}"] = values[key]
        for key, metric in values["metrics"].items():
            result[f"{role}_{key}_mean"] = metric["mean"]
            result[f"{role}_{key}_contributing_count"] = metric["contributing_count"]
        for key, value in values.get("onset_micro", {}).items():
            result[f"{role}_micro_{key}"] = value
    return result


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def preflight_outputs(outputs: list[Path], protected_inputs: list[Path]) -> None:
    """Check the complete write set before touching any report file or directory."""
    for index, output in enumerate(outputs):
        if output.exists() and not output.is_file():
            raise ValueError(f"Report output must be a file: {output}")
        for source in protected_inputs:
            if _same_path(output, source):
                raise ValueError(f"Report output must not overwrite input: {source}")
        for other in outputs[:index]:
            if _same_path(output, other):
                raise ValueError(f"Report output paths must not alias each other: {output}")
        for parent in output.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Report parent must be a directory: {parent}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", newline="", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_reports(
    records: list[dict], output_dir: Path, *, protected_inputs: list[Path], provenance: dict
) -> dict:
    """Write per-stem/per-clip JSON and CSV; dataset.json is the authoritative index.

    Every file is atomically replaced, with dataset.json replaced last. Obsolete files
    from earlier evaluations are retained and are not part of the current index.
    """
    output_dir = Path(output_dir)
    stem_ids = [row["stem_id"] for row in records]
    if len(set(stem_ids)) != len(stem_ids):
        raise ValueError("Report stem IDs must be unique")
    stem_files = {
        stem_id: f"stems/{hashlib.sha256(stem_id.encode('utf-8')).hexdigest()}.json"
        for stem_id in stem_ids
    }
    clips = []
    for clip_key in sorted({row["clip_key"] for row in records}):
        rows = [row for row in records if row["clip_key"] == clip_key]
        clips.append(
            {
                "schema_version": 1,
                "clip_key": clip_key,
                "summary": summarize_records(rows),
                "stems": rows,
                "stem_files": {row["stem_id"]: stem_files[row["stem_id"]] for row in rows},
                "provenance": provenance,
                "decision": None,
            }
        )
    clip_files = {
        clip[
            "clip_key"
        ]: f"clips/{hashlib.sha256(clip['clip_key'].encode('utf-8')).hexdigest()}.json"
        for clip in clips
    }
    dataset = {
        "schema_version": 1,
        "summary": summarize_records(records),
        "stems": records,
        "clips": [{"clip_key": clip["clip_key"], "summary": clip["summary"]} for clip in clips],
        "files": {
            "stems": stem_files,
            "clips": clip_files,
            "stems_csv": "stems.csv",
            "clips_csv": "clips.csv",
        },
        "provenance": provenance,
        "aggregation_policy": {
            "unit": "stem",
            "means": "role-specific arithmetic means over evaluated finite measurements only",
            "missing_values": "null; never treated as zero",
            "onset_micro": "pooled counts; zero-denominator precision/recall are null",
            "csv_nulls": "empty cells",
            "index": "dataset.json; unindexed files from previous runs are not current results",
        },
        "decision": None,
    }
    payloads = [(output_dir / stem_files[row["stem_id"]], _json(row)) for row in records]
    payloads.extend((output_dir / clip_files[clip["clip_key"]], _json(clip)) for clip in clips)
    payloads.extend(
        [
            (
                output_dir / "stems.csv",
                _csv([_stem_csv_row(row) for row in records], ("stem_id", "clip_key", "role")),
            ),
            (
                output_dir / "clips.csv",
                _csv([_clip_csv_row(clip) for clip in clips], ("clip_key",)),
            ),
            (output_dir / "dataset.json", _json(dataset)),
        ]
    )
    preflight_outputs([path for path, _ in payloads], protected_inputs)
    for path, serialized in payloads:
        _atomic_text(path, serialized)
    return dataset
