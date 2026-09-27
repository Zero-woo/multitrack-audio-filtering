"""Reference-provider boundary; WAVES planned support is never video ground truth."""

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

from waves_sed.metadata import StemMetadata


def _intervals(value: object) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise ValueError("Expected support must be a sequence of intervals or None")
    intervals = []
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Expected support intervals must contain a start and an end")
        try:
            valid = all(
                not isinstance(item, bool) and isinstance(item, Real) and math.isfinite(item)
                for item in pair
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected support boundaries must be finite real numbers")
        start, end = map(float, pair)
        if start < 0 or end <= start or (intervals and start < intervals[-1][0]):
            raise ValueError("Expected support must be positive, nonnegative and ordered by start")
        intervals.append((start, end))
    return tuple(intervals)


@dataclass(frozen=True)
class TemporalSupport:
    origin: str | None
    status: str | None
    intervals: tuple[tuple[float, float], ...] | None
    usable: bool
    reason: str | None
    provenance: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervals", _intervals(self.intervals))
        if not isinstance(self.usable, bool):
            raise ValueError("Temporal support usable must be boolean")
        if any(
            item is not None and not isinstance(item, str)
            for item in (self.origin, self.status, self.reason)
        ):
            raise ValueError("Temporal support origin, status and reason must be strings or None")
        if self.usable and (
            self.intervals is None or not self.origin or not self.status or self.reason is not None
        ):
            raise ValueError(
                "Usable temporal support needs intervals, origin, status and no reason"
            )
        if not isinstance(self.provenance, dict):
            raise ValueError("Temporal support provenance must be a JSON object")
        try:
            json.dumps(self.provenance, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("Temporal support provenance must be finite JSON data") from error

    def to_dict(self) -> dict:
        result = asdict(self)
        result["intervals"] = (
            [list(interval) for interval in self.intervals] if self.intervals is not None else None
        )
        return result


@runtime_checkable
class TemporalReference(Protocol):
    def resolve(self, stem: StemMetadata, *, allow_ambiguous: bool = False) -> TemporalSupport:
        """Return explicit source provenance and whether comparison is supported."""
        ...


class ExternalVideoReference(TemporalReference, Protocol):
    """Future independent video reference boundary; no detector is implemented here.

    Implementations must identify the video and detector/annotation provenance and
    declare ambiguous or unavailable support instead of silently substituting WAVES plans.
    External providers choose their own origin/status vocabulary; an explicitly usable
    empty annotation is permitted here, whereas WAVES empty planned support is unavailable.
    """


class WavesPlannedReference:
    def resolve(self, stem: StemMetadata, *, allow_ambiguous: bool = False) -> TemporalSupport:
        if not isinstance(allow_ambiguous, bool):
            raise ValueError("allow_ambiguous must be boolean")
        if not isinstance(stem.provenance, dict):
            raise ValueError("Stem provenance must be an object")
        origin = stem.provenance.get("reference_origin")
        status = stem.provenance.get("reference_status")
        if any(item is not None and not isinstance(item, str) for item in (origin, status)):
            raise ValueError("Reference origin and status must be strings or None")
        intervals = _intervals(stem.expected_intervals)
        ambiguity_issues = {
            "relabelled",
            "role_changed",
            "merged_support_parent_only",
            "missing_final_label",
            "missing_final_role",
            "missing_planned_label",
            "missing_planned_role",
            "missing_selected_attempt",
            "selected_attempt_not_found",
            "decision_mismatch",
        }
        conflicting_signals = sorted(ambiguity_issues.intersection(stem.issues))
        if stem.merged_candidate_ids:
            conflicting_signals.append("merged_candidate_ids_present")
        if (
            stem.source_description is not None
            and stem.planned_label is not None
            and stem.source_description != stem.planned_label
        ):
            conflicting_signals.append("final_label_differs_from_planned_label")
        if (
            stem.role is not None
            and stem.planned_role is not None
            and stem.role != stem.planned_role
        ):
            conflicting_signals.append("final_role_differs_from_planned_role")
        if origin != "waves_pass1_planned":
            reason = "unknown_reference_origin"
        elif status not in {"planned", "ambiguous", "missing"}:
            reason = "unknown_reference_status"
        elif status == "planned" and conflicting_signals:
            reason = "inconsistent_reference_status"
        elif status == "missing" or intervals is None:
            reason = "missing_support"
        elif not intervals:
            reason = "empty_support"
        elif status == "ambiguous" and not allow_ambiguous:
            reason = "ambiguous_reference"
        else:
            reason = None
        return TemporalSupport(
            origin=origin,
            status=status,
            intervals=intervals,
            usable=reason is None,
            reason=reason,
            provenance={
                "provider": "WavesPlannedReference",
                "stem_provenance": deepcopy(stem.provenance),
                "stem_issues": list(stem.issues),
                "allow_ambiguous_reference": allow_ambiguous,
                "ambiguity_signals": conflicting_signals,
            },
        )
