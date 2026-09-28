"""Paired, descriptive corruption measurements over existing evaluation reports.

The generated control is the baseline. Original timing references stay fixed;
neither a waveform edit nor its intended effect is treated as an observed SED
response. Missing or incompatible evidence never contributes a zero delta.
"""

import json
import math
import re
from copy import deepcopy
from pathlib import Path

from waves_sed.reporting import ROLE_METRICS, _atomic_text, _csv, _json, preflight_outputs
from waves_sed.temporal_metrics import intersection_duration, interval_duration, validate_intervals

_SEMANTIC_FIELDS = (
    "clip_key",
    "source_description",
    "role",
    "expected_intervals",
    "optional_video_id",
    "description_origin",
    "planned_description",
    "planned_label",
    "planned_role",
    "merged_candidate_ids",
    "issues",
)
_DETECTION_FIELDS = (
    "detection_count_delta",
    "detected_support_iou",
    "single_event_onset_displacement_ms",
    "single_event_shift_error_ms",
)


def _number(value) -> bool:
    try:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    except OverflowError:
        return False


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _validate_inputs(experiment: dict, dataset: dict) -> dict:
    for value, label in ((experiment, "Experiment"), (dataset, "Dataset")):
        _object(value, label)
        if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
            raise ValueError(f"{label} requires schema_version 1")
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must contain finite JSON data") from error
    _text(experiment.get("experiment_id"), "experiment_id")
    source = _object(experiment.get("source_stem"), "source_stem")
    _text(source.get("stem_id"), "source_stem.stem_id")
    _object(source.get("provenance"), "source_stem.provenance")
    source_audio = _object(experiment.get("source_audio"), "source_audio")
    _sha(source_audio.get("sha256"), "source_audio.sha256")
    _object(experiment.get("config"), "config")
    if experiment.get("reference_policy") != "unchanged_from_source":
        raise ValueError("Comparison requires reference_policy unchanged_from_source")
    control = _object(experiment.get("control"), "control")
    control_id = _text(control.get("stem_id"), "control.stem_id")
    _sha(control.get("audio_sha256"), "control.audio_sha256")
    _object(control.get("stem"), "control.stem")
    variants = experiment.get("variants")
    if not isinstance(variants, list):
        raise ValueError("variants must be a list")
    ids, stem_ids = {"control"}, {control_id}
    for variant in variants:
        _object(variant, "variant")
        variant_id = _text(variant.get("id"), "variant.id")
        stem_id = _text(variant.get("stem_id"), "variant.stem_id")
        _text(variant.get("operation"), "variant.operation")
        _object(variant.get("spec"), "variant.spec")
        if variant_id in ids or stem_id in stem_ids:
            raise ValueError("Experiment variant and stem IDs must be unique")
        ids.add(variant_id)
        stem_ids.add(stem_id)
        if variant.get("status") not in {"generated", "failed"}:
            raise ValueError("Variant status must be generated or failed")
        if variant["status"] == "generated":
            _sha(variant.get("audio_sha256"), "variant.audio_sha256")
            _object(variant.get("stem"), "variant.stem")
            application = _object(variant.get("application"), "variant.application")
            if not isinstance(application.get("audio_changed"), bool):
                raise ValueError("Generated variant application requires boolean audio_changed")
    rows = dataset.get("stems")
    if not isinstance(rows, list):
        raise ValueError("Dataset stems must be a list")
    indexed = {}
    for row in rows:
        _object(row, "Dataset stem")
        stem_id = _text(row.get("stem_id"), "Dataset stem_id")
        if stem_id in indexed:
            raise ValueError("Duplicate dataset stem_id")
        indexed[stem_id] = row
    return indexed


def _without_paths(value):
    """Equal content hashes may come from different locations."""
    if isinstance(value, dict):
        return {key: _without_paths(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _context(row: dict) -> dict:
    provenance = row.get("provenance") or {}
    prediction = provenance.get("prediction") or {}
    metadata = prediction.get("metadata") or {}
    reference = row.get("reference") or {}
    return {
        "evaluation_config": provenance.get("evaluation_config"),
        "metric_implementation_version": provenance.get("metric_implementation_version"),
        "metric_conventions": provenance.get("metric_conventions"),
        "mapping_provenance": _without_paths(provenance.get("mapping")),
        "mapping_resolution": row.get("mapping"),
        "model": {
            "sed_backend": row.get("sed_backend"),
            **{
                key: metadata.get(key)
                for key in (
                    "backend",
                    "checkpoint_sha256",
                    "upstream_revision",
                    "preprocessing",
                    "package_version",
                    "frame_hop_seconds",
                    "chunk_seconds",
                    "frame_time_convention",
                    "smoothing",
                    "threshold",
                    "model_frozen",
                    "dependency_versions",
                    "device",
                    "inference_sample_rate",
                )
            },
            "class_ids": prediction.get("class_ids"),
            "class_names": prediction.get("class_names"),
        },
        "reference": {
            key: reference.get(key) for key in ("origin", "status", "intervals", "usable", "reason")
        },
        "reference_provider": (reference.get("provenance") or {}).get("provider"),
        "expected_intervals": row.get("expected_intervals"),
        "role": row.get("role"),
        "source_description": row.get("source_description"),
    }


def _binding_reasons(experiment, entry, row, *, kind):
    if row is None:
        return ["report_missing"]
    reasons = []
    stem = entry["stem"]
    source = experiment["source_stem"]
    provenance = _object(row.get("provenance") or {}, "Report provenance")
    stem_provenance = _object(stem.get("provenance") or {}, "Manifest stem provenance")
    if provenance.get("stem") != stem:
        reasons.append("report_stem_mismatch")
    if stem.get("stem_id") != entry["stem_id"] or stem.get("audio_path") != entry.get("audio_path"):
        reasons.append("manifest_stem_mismatch")
    if kind == "control" and (
        entry.get("audio_path") != experiment["source_audio"].get("path")
        or source.get("audio_path") != experiment["source_audio"].get("path")
        or entry["audio_sha256"].lower() != experiment["source_audio"]["sha256"].lower()
    ):
        reasons.append("original_audio_mismatch")
    if any(stem.get(key) != source.get(key) for key in _SEMANTIC_FIELDS):
        reasons.append("source_semantics_mismatch")
    if any(
        stem_provenance.get(key) != source["provenance"].get(key)
        for key in ("reference_origin", "reference_status", "support_policy")
    ):
        reasons.append("source_reference_mismatch")
    corruption = _object(stem_provenance.get("corruption") or {}, "Corruption provenance")
    expected_binding = {
        "experiment_id": experiment["experiment_id"],
        "kind": kind,
        "variant_id": "control" if kind == "control" else entry["id"],
        "parent_stem_id": source["stem_id"],
        "control_stem_id": experiment["control"]["stem_id"],
        "operation": "control" if kind == "control" else entry["operation"],
        "source_stem": source,
        "spec": None if kind == "control" else entry["spec"],
        "application": None if kind == "control" else entry["application"],
        "source_audio": {key: experiment["source_audio"].get(key) for key in ("path", "sha256")},
    }
    if any(corruption.get(key) != expected for key, expected in expected_binding.items()):
        reasons.append("corruption_provenance_mismatch")
    if any(
        row.get(key) != stem.get(key) for key in ("stem_id", "candidate_id", *_SEMANTIC_FIELDS[:4])
    ):
        reasons.append("report_semantics_mismatch")
    if row.get("expected_intervals") != source.get("expected_intervals"):
        reasons.append("report_reference_intervals_mismatch")
    reference = _object(row.get("reference") or {}, "Report reference")
    _object(reference.get("provenance") or {}, "Reference provenance")
    if any(
        reference.get(key) != source["provenance"].get(f"reference_{key}")
        for key in ("origin", "status")
    ) or reference.get("intervals") != source.get("expected_intervals"):
        reasons.append("report_reference_mismatch")
    prediction = provenance.get("prediction")
    if prediction is not None:
        _object(prediction, "Prediction provenance")
        metadata = _object(prediction.get("metadata") or {}, "Prediction metadata")
        digest = metadata.get("audio_sha256")
        if not isinstance(digest, str) or digest.lower() != entry["audio_sha256"].lower():
            reasons.append("prediction_audio_mismatch")
        if metadata.get("backend") != row.get("sed_backend"):
            reasons.append("prediction_backend_mismatch")
    if row.get("evaluation_status") != "evaluated":
        reasons.append("evaluation_unavailable")
    if (
        row.get("mapping_status") != "supported"
        or row.get("detection_status") != "available"
        or reference.get("usable") is not True
        or _object(row.get("cache_validation") or {}, "Cache validation").get("verified")
        is not True
        or not isinstance(row.get("metrics"), dict)
        or prediction is None
    ):
        reasons.append("evaluation_evidence_incomplete")
    if prediction is not None and (
        not isinstance(provenance.get("evaluation_config"), dict)
        or not isinstance(provenance.get("mapping"), dict)
        or provenance.get("metric_implementation_version") is None
        or not isinstance(row.get("mapping"), dict)
        or not prediction.get("class_ids")
        or not prediction.get("class_names")
        or not row.get("sed_backend")
    ):
        reasons.append("comparison_context_incomplete")
    try:
        validate_intervals(row.get("detected_intervals"))
    except ValueError:
        reasons.append("invalid_detected_intervals")
    return reasons


def _deltas(control, variant, role):
    keys = set(ROLE_METRICS.get(role, ())) | {
        key
        for metrics in (control, variant)
        for key, value in metrics.items()
        if _number(value) or value is None
    }
    return {
        key: variant[key] - control[key]
        if _number(control.get(key)) and _number(variant.get(key))
        else None
        for key in sorted(keys)
    }


def _detection_changes(control, variant, entry):
    left, right = control["detected_intervals"], variant["detected_intervals"]
    intersection = intersection_duration(left, right)
    union = interval_duration((*left, *right))
    displacement = (right[0][0] - left[0][0]) * 1000 if len(left) == len(right) == 1 else None
    shift = entry["application"].get("realized_shift_seconds")
    return {
        "detection_count_delta": len(right) - len(left),
        "detected_support_iou": intersection / union if union else None,
        "single_event_onset_displacement_ms": displacement,
        "single_event_shift_error_ms": displacement - shift * 1000
        if entry["operation"] == "shift" and displacement is not None and _number(shift)
        else None,
    }


def _summary(rows):
    compared = [row for row in rows if row["comparison_status"] == "compared"]
    result = {
        "variant_count": len(rows),
        "compared_count": len(compared),
        "unavailable_count": len(rows) - len(compared),
    }
    for field in ("metric_deltas", "detection_changes"):
        keys = sorted(
            {key for row in compared for key in row[field]}
            | (
                {key for row in rows for key in ROLE_METRICS.get(row["role"], ())}
                if field == "metric_deltas"
                else set(_DETECTION_FIELDS)
            )
        )
        result[field] = {}
        for key in keys:
            values = [row[field][key] for row in compared if _number(row[field].get(key))]
            result[field][key] = {
                "mean": math.fsum(values) / len(values) if values else None,
                "contributing_count": len(values),
            }
    return result


def compare_corruptions(experiment: dict, dataset: dict) -> dict:
    """Compare compatible, evaluated, changed variants to their generated control.

    This function reads no files, decodes no audio, and loads no model. It binds
    reports to the supplied manifest hashes and metadata, which the evaluation
    step verified against audio/cache bytes. It does not reverify current files.
    """
    indexed = _validate_inputs(experiment, dataset)
    control_entry = experiment["control"]
    control = indexed.get(control_entry["stem_id"])
    control_reasons = _binding_reasons(experiment, control_entry, control, kind="control")
    comparisons = []
    for entry in experiment["variants"]:
        variant = indexed.get(entry["stem_id"])
        reasons = [f"control_{reason}" for reason in control_reasons]
        variant_reasons = []
        if entry["status"] != "generated":
            variant_reasons.append("generation_failed")
        else:
            variant_reasons = _binding_reasons(experiment, entry, variant, kind="variant")
        reasons.extend(f"variant_{reason}" for reason in variant_reasons)
        if entry["status"] == "generated" and (
            not entry["application"]["audio_changed"]
            or entry["audio_sha256"].lower() == control_entry["audio_sha256"].lower()
        ):
            reasons.append("no_waveform_change")
        if not control_reasons and not variant_reasons:
            left_context, right_context = _context(control), _context(variant)
            reasons.extend(
                f"incompatible_{key}"
                for key in left_context
                if left_context[key] != right_context[key]
            )
        compared = not reasons
        comparisons.append(
            {
                "variant_id": entry["id"],
                "stem_id": entry["stem_id"],
                "operation": entry["operation"],
                "role": experiment["source_stem"].get("role"),
                "comparison_status": "compared" if compared else "unavailable",
                "reason_codes": reasons,
                "spec": entry["spec"],
                "application": entry.get("application"),
                "generation_error": entry.get("error"),
                "control_metrics": control["metrics"] if not control_reasons else None,
                "variant_metrics": variant["metrics"] if not variant_reasons else None,
                "control_evaluation_reasons": control.get("reason_codes", []) if control else [],
                "variant_evaluation_reasons": variant.get("reason_codes", []) if variant else [],
                "metric_deltas": _deltas(control["metrics"], variant["metrics"], control["role"])
                if compared
                else None,
                "detection_changes": _detection_changes(control, variant, entry)
                if compared
                else {key: None for key in _DETECTION_FIELDS},
                "control_audio_sha256": control_entry["audio_sha256"],
                "variant_audio_sha256": entry.get("audio_sha256"),
                "control_context": _context(control) if control else None,
                "variant_context": _context(variant) if variant else None,
                "decision": None,
            }
        )
    summary = _summary(comparisons)
    summary["by_operation"] = {
        operation: _summary([row for row in comparisons if row["operation"] == operation])
        for operation in sorted({row["operation"] for row in comparisons})
    }
    related = {control_entry["stem_id"], *(entry["stem_id"] for entry in experiment["variants"])}
    summary["ignored_dataset_stem_count"] = len(indexed.keys() - related)
    return deepcopy(
        {
            "schema_version": 1,
            "experiment_id": experiment["experiment_id"],
            "control_stem_id": control_entry["stem_id"],
            "comparisons": comparisons,
            "summary": summary,
            "measurement_policy": {
                "metric_deltas": "variant minus generated control; undefined inputs remain null",
                "aggregation": "only compatible evaluated pairs with changed waveforms; per-field finite counts",
                "detected_support_iou": "intersection / union of control and variant detected support; empty union is null",
                "single_event_displacement": "signed variant minus control onset, milliseconds; requires one detection in each",
                "single_event_shift_error": "observed displacement minus realized waveform shift; descriptive, no event identity assertion",
                "reference": "unchanged source reference; WAVES planned support is not video ground truth",
                "identity": "report-to-manifest metadata and hash binding; current files are not reopened",
                "interpretation": "intended waveform changes and observed detector responses are separate; no quality decision",
            },
            "provenance": {
                "experiment": experiment,
                "dataset": dataset.get("provenance"),
                "comparison_implementation_version": 1,
            },
            "decision": None,
        }
    )


def write_comparison(result: dict, output_dir: Path, *, protected_inputs: list[Path]) -> None:
    """Write flat CSV first and authoritative JSON last, preserving input aliases."""
    output_dir = Path(output_dir)
    rows = []
    for comparison in result["comparisons"]:
        row = {
            key: value
            for key, value in comparison.items()
            if key not in {"metric_deltas", "detection_changes"}
        }
        row.update(
            {
                f"delta_{key}": value
                for key, value in (
                    comparison["metric_deltas"]
                    or dict.fromkeys(ROLE_METRICS.get(comparison["role"], ()))
                ).items()
            }
        )
        row.update(comparison["detection_changes"])
        rows.append(row)
    payloads = [
        (
            output_dir / "comparison.csv",
            _csv(rows, ("variant_id", "stem_id", "operation", "comparison_status")),
        ),
        (output_dir / "comparison.json", _json(result)),
    ]
    preflight_outputs([path for path, _ in payloads], [Path(path) for path in protected_inputs])
    for path, text in payloads:
        _atomic_text(path, text)
