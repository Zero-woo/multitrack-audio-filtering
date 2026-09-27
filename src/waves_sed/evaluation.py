"""Evaluate cached probabilities without importing a SED backend or decoding audio."""

import math
import re
from pathlib import Path

import numpy as np

from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.events import extract_events
from waves_sed.mapping import ManualMapping, MappingResolution, aggregate_target
from waves_sed.metadata import StemMetadata
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256
from waves_sed.temporal_metrics import ambience_metrics, onset_metrics, span_metrics
from waves_sed.temporal_reference import TemporalReference, WavesPlannedReference

METRIC_CONVENTIONS = {
    "implementation_version": 1,
    "intervals": "half-open [start, end), seconds",
    "onset_matching": "maximum cardinality within tolerance, then minimum absolute onset error",
    "onset_error_statistics": "absolute milliseconds; matches also retain signed errors",
    "span_boundaries": "signed detected minus expected outer boundaries, seconds",
    "duration_sets": "union overlapping or touching intervals before measuring duration",
    "out_of_window_activation": "seconds of detected support outside expected support",
    "out_of_window_activity": "seconds of detected support outside expected support",
    "mean_target_confidence": "raw target probability weighted by overlap duration in expected support",
    "zero_denominators": "null (not a zero score)",
    "empty_planned_support": "unavailable; not evidence of an expected silent stem",
    "reference_scope": "WAVES plans measure internal consistency, not video synchronization",
    "decision": "none; extraction thresholds are not quality thresholds",
}


def _digest(value) -> str | None:
    return (
        value.lower()
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value)
        else None
    )


def validate_cache_identity(stem: StemMetadata, prediction: FramePrediction) -> dict:
    """Require matching bytes, or an explicit manifest digest if bytes are absent.

    A filename, audio ID, or cache's own path is never proof of stem identity.
    A matching manifest hash is recorded separately from reading current bytes.
    """
    cache_digest = _digest(prediction.metadata.get("audio_sha256"))
    raw_final = stem.provenance.get("raw_final_stem", {})
    declared = raw_final.get("sha256") if isinstance(raw_final, dict) else None
    manifest_digest = _digest(declared)
    result = {
        "status": "unavailable",
        "verified": False,
        "source_file_verified": False,
        "cache_audio_sha256": cache_digest,
        "manifest_audio_sha256": manifest_digest,
        "current_audio_sha256": None,
        "reason_codes": [],
    }
    reasons = result["reason_codes"]
    if cache_digest is None:
        reasons.append("missing_or_invalid_cache_audio_sha256")
    if declared is not None and manifest_digest is None:
        reasons.append("invalid_manifest_audio_sha256")
    current = None
    if stem.audio_path is not None and Path(stem.audio_path).exists():
        try:
            current = sha256(stem.audio_path)
        except OSError as exc:
            reasons.append("unreadable_source_audio")
            result["error"] = str(exc)
    result["current_audio_sha256"] = current
    if current is not None and manifest_digest is not None and current != manifest_digest:
        reasons.append("source_audio_manifest_mismatch")
    expected = current if current is not None else manifest_digest
    if expected is None:
        reasons.append("audio_identity_unverified")
    elif cache_digest is not None and cache_digest != expected:
        reasons.append("prediction_audio_mismatch")
    if not reasons:
        result.update(
            status="verified_current_audio" if current is not None else "verified_manifest_hash",
            verified=True,
            source_file_verified=current is not None,
        )
    return result


def _observation_reasons(prediction: FramePrediction) -> list[str]:
    """A temporal report needs the complete track, including its silent portions."""
    reasons = []
    if (
        not isinstance(prediction.metadata.get("backend"), str)
        or not prediction.metadata["backend"].strip()
    ):
        reasons.append("missing_prediction_backend")
    duration = prediction.metadata.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        reasons.append("missing_or_invalid_prediction_duration")
    elif not math.isclose(
        float(prediction.frame_end_seconds[-1]), duration, abs_tol=1e-12, rel_tol=0
    ):
        reasons.append("prediction_duration_mismatch")
    # Actual gaps remain unknown time, even if all expected events fit elsewhere.
    if prediction.frame_start_seconds[0] != 0 or np.any(
        prediction.frame_start_seconds[1:] != prediction.frame_end_seconds[:-1]
    ):
        reasons.append("incomplete_prediction_timeline")
    return reasons


def semantic_evidence(
    prediction: FramePrediction,
    resolution: MappingResolution,
    mapping: ManualMapping,
    config: EvaluationConfig,
) -> dict:
    """Outside-family scores are observations, never automatic foreign decisions.

    Ontology ancestors/descendants outside the explicit allowed set are marked
    as related; unavailable hierarchy is reported rather than fabricated.
    """
    ontology = mapping.ontology
    ancestors, descendants = set(), set()
    allowed = set(resolution.allowed_class_ids)
    known_allowed = allowed & ontology.nodes.keys()
    for class_id in known_allowed:
        ancestors.update(ontology.ancestors(class_id, include_self=False))
        descendants.update(ontology.descendants(class_id, include_self=False))
    foreign = set(resolution.foreign_class_ids)
    durations = prediction.frame_end_seconds - prediction.frame_start_seconds
    duration = float(durations.sum())
    items = []
    explicit = []
    for column, class_id in enumerate(prediction.class_ids):
        if class_id in allowed:
            continue
        scores = prediction.probabilities[:, column]
        peak = float(scores.max())
        if peak < config.outside_family_threshold and class_id not in foreign:
            continue
        relations = []
        if class_id in ancestors:
            relations.append("ancestor")
        if class_id in descendants:
            relations.append("descendant")
        if not relations:
            relations = [
                "unknown_hierarchy"
                if class_id not in ontology.nodes or len(known_allowed) != len(allowed)
                else "no_ancestor_or_descendant_relation"
            ]
        item = {
            "class_id": class_id,
            "model_name": prediction.class_names[column],
            "ontology_name": ontology.nodes[class_id].name if class_id in ontology.nodes else None,
            "relations_to_allowed": relations,
            "explicit_foreign": class_id in foreign,
            "max_probability": peak,
            "mean_probability": float(np.dot(scores.astype(np.float64), durations) / duration),
            "activation_duration_seconds": float(
                durations[scores >= config.outside_family_threshold].sum()
            ),
            "above_threshold": peak >= config.outside_family_threshold,
        }
        if item["above_threshold"]:
            items.append(item)
        if class_id in foreign:
            explicit.append(item)
    items.sort(key=lambda item: (-item["max_probability"], item["class_id"]))
    explicit.sort(key=lambda item: (-item["max_probability"], item["class_id"]))
    return {
        "threshold": config.outside_family_threshold,
        "threshold_basis": "raw frame probability >= threshold, no smoothing or duration filtering",
        "top_k": config.outside_family_top_k,
        "outside_allowed_family": items[: config.outside_family_top_k],
        "outside_allowed_family_class_count_above_threshold": len(items),
        "explicit_foreign_classes": explicit,
        "unavailable_foreign_class_ids": sorted(foreign - set(prediction.class_ids)),
        "interpretation": "outside allowed family is not automatically a foreign sound or a failure",
    }


def evaluate_stem(
    stem: StemMetadata,
    prediction: FramePrediction | None,
    mapping: ManualMapping,
    config: EvaluationConfig,
    *,
    reference: TemporalReference | None = None,
    prediction_path: Path | None = None,
    prediction_error: str | None = None,
) -> dict:
    """Return a self-contained stem report; unavailable evidence stays null.

    A trusted cache can still supply detected events when temporal reference is
    unusable. Such a row is excluded from metric aggregates. No model is invoked.
    """
    config.validate()
    if prediction is not None:
        prediction.validate()
    path = Path(prediction_path).resolve() if prediction_path is not None else None
    reference = reference if reference is not None else WavesPlannedReference()
    support = reference.resolve(stem, allow_ambiguous=config.allow_ambiguous_reference)
    resolution = mapping.resolve(
        stem.source_description,
        prediction_class_ids=prediction.class_ids if prediction is not None else None,
    )
    report = {
        "schema_version": 1,
        "stem_id": stem.stem_id,
        "candidate_id": stem.candidate_id,
        "clip_key": stem.clip_key,
        "source_description": stem.source_description,
        "role": stem.role,
        "sed_backend": prediction.metadata.get("backend") if prediction is not None else None,
        "mapping_status": resolution.status,
        "mapping": resolution.to_dict(),
        "mapping_vocabulary": "prediction"
        if prediction is not None
        else "default_backend_metadata",
        "reference": support.to_dict(),
        "expected_intervals": (
            [list(interval) for interval in support.intervals]
            if support.intervals is not None
            else None
        ),
        "detected_intervals": None,
        "evaluation_status": "unavailable",
        "detection_status": "unavailable",
        "reason_codes": [],
        "metrics": None,
        "semantic_evidence": None,
        "cache_validation": {"status": "unavailable", "verified": False},
        "provenance": {
            "stem": stem.to_dict(),
            "prediction": None,
            "prediction_request": {"path": str(path) if path is not None else None},
            "mapping": mapping.provenance,
            "evaluation_config": config.to_dict(),
            "metric_implementation_version": 1,
            "metric_conventions": METRIC_CONVENTIONS,
        },
        "decision": None,
    }
    reasons = report["reason_codes"]
    if resolution.status != "supported":
        reasons.append("unsupported_mapping")
    if stem.role not in {"onset", "span", "ambience"}:
        reasons.append("unsupported_role")
    if not support.usable:
        reasons.append(support.reason or "unavailable_reference")
    if prediction is None:
        reasons.append(
            "invalid_prediction"
            if prediction_error and prediction_error.startswith("invalid_prediction")
            else "missing_prediction"
        )
        if prediction_error:
            report["prediction_error"] = prediction_error
        return report
    report["provenance"]["prediction"] = {
        "path": str(path) if path is not None else None,
        "sha256": sha256(path) if path is not None else None,
        "metadata": prediction.metadata,
        "class_ids": list(prediction.class_ids),
        "class_names": list(prediction.class_names),
        "frame_count": len(prediction.probabilities),
    }
    identity = validate_cache_identity(stem, prediction)
    report["cache_validation"] = identity
    reasons.extend(identity["reason_codes"])
    observation_reasons = _observation_reasons(prediction)
    reasons.extend(observation_reasons)
    if not identity["verified"] or observation_reasons or resolution.status != "supported":
        return report

    target = aggregate_target(prediction, resolution)
    detected = extract_events(prediction, target, config.event)
    report["detected_intervals"] = [list(interval) for interval in detected]
    report["detection_status"] = "available"
    report["semantic_evidence"] = semantic_evidence(prediction, resolution, mapping, config)
    durations = prediction.frame_end_seconds - prediction.frame_start_seconds
    report["target_summary"] = {
        "aggregation": "max_allowed_class_probability",
        "max_probability": float(target.max()),
        "mean_probability": float(np.dot(target.astype(np.float64), durations) / durations.sum()),
        "confidence_basis": "raw, before event smoothing and thresholding",
    }
    if support.usable and any(
        start < prediction.frame_start_seconds[0] or end > prediction.frame_end_seconds[-1]
        for start, end in support.intervals
    ):
        reasons.append("reference_outside_prediction")
    if reasons:
        return report
    if stem.role == "onset":
        metrics = onset_metrics(support.intervals, detected, config.onset_tolerance_seconds)
    elif stem.role == "span":
        metrics = span_metrics(support.intervals, detected)
    else:
        metrics = ambience_metrics(
            support.intervals,
            detected,
            prediction.frame_start_seconds,
            prediction.frame_end_seconds,
            target,
        )
    report["metrics"] = metrics
    report["evaluation_status"] = "evaluated"
    return report
