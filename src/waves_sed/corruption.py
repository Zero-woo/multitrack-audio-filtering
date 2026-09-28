"""Deterministic sample edits for controlled experiments, without model dependencies.

Intervals are half-open. Edits preserve the original frame count, sample rate and
channels; insertion beyond either clip boundary is discarded. Insertions add to
existing audio without normalization, clipping, fades or time stretching.
"""

import json
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from numbers import Integral, Real
from pathlib import Path

import numpy as np

_OPERATIONS = {"shift", "remove", "duplicate", "shorten", "extend"}
_FIELDS = {
    "id",
    "operation",
    "interval",
    "shift_seconds",
    "destination_seconds",
    "duration_seconds",
}
ROUNDING_POLICY = "nearest sample; Decimal(str(seconds)) * sample_rate; ties away from zero"


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(value)
            and (minimum is None or value >= minimum)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(
            f"{name} must be a finite number" + (f" >= {minimum}" if minimum is not None else "")
        )
    return float(value)


@dataclass(frozen=True)
class CorruptionSpec:
    id: str
    operation: str
    interval: tuple[float, float] | None = None
    shift_seconds: float | None = None
    destination_seconds: float | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or self.id == "control":
            raise ValueError(
                "Corruption id must be a nonempty string other than reserved 'control'"
            )
        if not isinstance(self.operation, str) or self.operation not in _OPERATIONS:
            raise ValueError(f"Unknown corruption operation: {self.operation!r}")
        if self.interval is not None:
            if not isinstance(self.interval, (list, tuple)) or len(self.interval) != 2:
                raise ValueError("interval must contain [start_seconds, end_seconds]")
            start = _number(self.interval[0], "interval start", minimum=0)
            end = _number(self.interval[1], "interval end", minimum=0)
            if end <= start:
                raise ValueError("interval must have end > start")
            object.__setattr__(self, "interval", (start, end))
        elif self.operation != "shift":
            raise ValueError(f"{self.operation} requires an interval")

        required_field = {
            "shift": "shift_seconds",
            "duplicate": "destination_seconds",
            "shorten": "duration_seconds",
            "extend": "duration_seconds",
        }.get(self.operation)
        for field in ("shift_seconds", "destination_seconds", "duration_seconds"):
            value = getattr(self, field)
            if field != required_field:
                if value is not None:
                    raise ValueError(f"{field} is not applicable to {self.operation}")
                continue
            value = _number(value, field, minimum=None if field == "shift_seconds" else 0)
            if field != "destination_seconds" and value == 0:
                raise ValueError(f"{field} must be nonzero")
            object.__setattr__(self, field, value)
        if self.operation in {"shorten", "extend"}:
            with localcontext() as context:
                context.prec = 400
                original_duration = Decimal(str(self.interval[1])) - Decimal(str(self.interval[0]))
                requested_duration = Decimal(str(self.duration_seconds))
            if self.operation == "shorten" and requested_duration >= original_duration:
                raise ValueError("shorten duration_seconds must be less than the interval duration")
            if self.operation == "extend" and requested_duration <= original_duration:
                raise ValueError("extend duration_seconds must exceed the interval duration")

    def to_dict(self) -> dict:
        result = {"id": self.id, "operation": self.operation}
        if self.interval is not None:
            result["interval"] = list(self.interval)
        for field in ("shift_seconds", "destination_seconds", "duration_seconds"):
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        return result


def load_corruption_specs(path: str | Path) -> tuple[CorruptionSpec, ...]:
    """Read strict schema 1; every variant is independently applied to its source."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate corruption config JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"Nonfinite corruption config JSON value: {value}")

    value = json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=unique_object,
        parse_constant=nonfinite,
    )
    if not isinstance(value, dict) or set(value) != {"schema_version", "variants"}:
        raise ValueError("Corruption config requires exactly schema_version and variants")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Corruption config requires schema_version 1")
    if not isinstance(value["variants"], list) or not value["variants"]:
        raise ValueError("variants must be a nonempty list")
    specs = []
    seen = set()
    for item in value["variants"]:
        if (
            not isinstance(item, dict)
            or set(item) - _FIELDS
            or not {"id", "operation"} <= set(item)
        ):
            raise ValueError("Each corruption variant requires id, operation and known fields")
        spec = CorruptionSpec(**item)
        applicable = {"id", "operation", "interval"} | {
            "shift": {"shift_seconds"},
            "duplicate": {"destination_seconds"},
            "shorten": {"duration_seconds"},
            "extend": {"duration_seconds"},
            "remove": set(),
        }[spec.operation]
        if set(item) - applicable:
            raise ValueError(
                f"Variant {spec.id} contains fields not applicable to {spec.operation}"
            )
        if spec.id in seen:
            raise ValueError(f"Duplicate corruption variant id: {spec.id}")
        seen.add(spec.id)
        specs.append(spec)
    return tuple(specs)


def _sample(seconds: float, sample_rate: int) -> int:
    # Enough precision for finite IEEE floats, including deliberately huge shifts.
    # Huge destinations are compared as integers; they never determine allocations.
    with localcontext() as context:
        context.prec = max(400, len(str(sample_rate)) + 350)
        value = Decimal(str(seconds)) * sample_rate
        return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def _merged_intervals(intervals: list[tuple[int, int]]) -> list[list[int]]:
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def apply_corruption(
    samples: np.ndarray, sample_rate: int, spec: CorruptionSpec
) -> tuple[np.ndarray, dict]:
    """Return an independent edited array and a JSON-compatible application record.

    ``cropped_source_sample_frames`` counts insertion frames discarded at the clip
    boundaries (including repeated frames for extend). ``additive_overlap_sample_frames``
    counts frames where nonzero inserted and existing samples share a channel.
    Shorten removes a segment's tail; extend loops the segment, without time stretching.
    A sub-sample request may quantize to an unchanged signal; ``audio_changed`` records it.
    """
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 2
        or 0 in samples.shape
        or not np.issubdtype(samples.dtype, np.floating)
        or not np.isfinite(samples).all()
    ):
        raise ValueError("samples must be a nonempty finite real floating array [frames, channels]")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate < 1:
        raise ValueError("sample_rate must be a positive integer")
    if not isinstance(spec, CorruptionSpec):
        raise ValueError("spec must be a CorruptionSpec")
    sample_rate = int(sample_rate)
    frame_count, channel_count = samples.shape
    if spec.interval is None:
        source_start, source_end = 0, frame_count
    else:
        requested_end = spec.interval[1]
        clip_duration = frame_count / sample_rate
        if requested_end > clip_duration and not math.isclose(
            requested_end, clip_duration, rel_tol=2e-15, abs_tol=0
        ):
            raise ValueError("Source interval extends beyond the input audio duration")
        source_start, source_end = (_sample(value, sample_rate) for value in spec.interval)
        if source_start < 0 or source_end > frame_count or source_end <= source_start:
            raise ValueError("Source interval must quantize to at least one sample within the clip")
    source_length = source_end - source_start
    source = samples[source_start:source_end].copy()
    result = samples.copy()
    shift_samples = _sample(spec.shift_seconds, sample_rate) if spec.operation == "shift" else None
    duration_samples = (
        _sample(spec.duration_seconds, sample_rate)
        if spec.operation in {"shorten", "extend"}
        else None
    )
    if duration_samples is not None:
        if duration_samples < 1:
            raise ValueError("duration_seconds must quantize to at least one sample")
        if (spec.operation == "shorten" and duration_samples > source_length) or (
            spec.operation == "extend" and duration_samples < source_length
        ):
            raise ValueError(
                "Quantized duration contradicts the requested shorten/extend operation"
            )

    destination_start = destination_end = None
    visible_start = visible_end = None
    cropped = overlap = 0
    affected = []
    if spec.operation == "remove":
        result[source_start:source_end] = 0
        affected.append((source_start, source_end))
    elif spec.operation == "shorten":
        result[source_start + duration_samples : source_end] = 0
        affected.append((source_start + duration_samples, source_end))
    else:
        insertion_length = duration_samples if spec.operation == "extend" else source_length
        destination_start = (
            _sample(spec.destination_seconds, sample_rate)
            if spec.operation == "duplicate"
            else source_start + (shift_samples or 0)
        )
        destination_end = destination_start + insertion_length
        if spec.operation in {"shift", "extend"}:
            result[source_start:source_end] = 0
            affected.append((source_start, source_end))
        visible_start = min(frame_count, max(0, destination_start))
        visible_end = min(frame_count, max(0, destination_end))
        visible_count = visible_end - visible_start
        cropped = insertion_length - visible_count
        if visible_count:
            offset = visible_start - destination_start
            if spec.operation == "extend":
                # Allocate at most the visible clip portion, even for huge requested durations.
                indices = (np.arange(visible_count) + offset % source_length) % source_length
                inserted = source[indices]
            else:
                inserted = source[offset : offset + visible_count]
            existing = result[visible_start:visible_end]
            overlap = int(np.count_nonzero(np.any((existing != 0) & (inserted != 0), axis=1)))
            with np.errstate(over="ignore", invalid="ignore"):
                existing += inserted
            affected.append((visible_start, visible_end))
    if not np.isfinite(result).all():
        raise ValueError("Additive corruption overflowed the audio dtype; no clipping was applied")
    record = {
        "schema_version": 1,
        "variant_id": spec.id,
        "operation": spec.operation,
        "sample_rate": sample_rate,
        "input_frames": frame_count,
        "output_frames": frame_count,
        "channels": channel_count,
        "source_start_sample": source_start,
        "source_end_sample": source_end,
        "requested_source_interval_seconds": None if spec.interval is None else list(spec.interval),
        "realized_source_interval_seconds": [source_start / sample_rate, source_end / sample_rate],
        "destination_start_sample": destination_start,
        "destination_end_sample": destination_end,
        "requested_destination_seconds": spec.destination_seconds,
        "realized_destination_seconds": (
            None if destination_start is None else destination_start / sample_rate
        ),
        "visible_destination_start_sample": visible_start,
        "visible_destination_end_sample": visible_end,
        "affected_output_intervals_samples": _merged_intervals(affected),
        "requested_shift_seconds": spec.shift_seconds,
        "shift_samples": shift_samples,
        "realized_shift_seconds": None if shift_samples is None else shift_samples / sample_rate,
        "requested_duration_seconds": spec.duration_seconds,
        "duration_samples": duration_samples,
        "realized_duration_seconds": None
        if duration_samples is None
        else duration_samples / sample_rate,
        "cropped_source_sample_frames": cropped,
        "additive_overlap_sample_frames": overlap,
        "peak_before": float(np.max(np.abs(samples))),
        "peak_after": float(np.max(np.abs(result))),
        "clipping_applied": False,
        "audio_changed": not np.array_equal(samples, result),
        "rounding_policy": ROUNDING_POLICY,
        "interval_policy": "half-open [start_sample, end_sample)",
        "boundary_policy": "preserve clip length; zero-pad vacated samples; discard out-of-clip insertions",
        "insertion_policy": "additive overlay; no normalization or clipping",
        "duration_policy": "shorten keeps prefix; extend repeats source; no time stretching",
        "transition_policy": "hard cuts; no fades; no resampling",
    }
    return result, record
