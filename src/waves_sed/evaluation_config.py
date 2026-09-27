"""Explicit, uncalibrated extraction and reporting settings; no quality decisions."""

import json
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path


def _finite_number(value: object, name: str, *, upper: float | None = None) -> None:
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(value)
            and value >= 0
            and (upper is None or value <= upper)
        )
    except OverflowError:
        valid = False
    if not valid:
        bounds = "[0, 1]" if upper == 1 else "[0, infinity)"
        raise ValueError(f"{name} must be a finite number in {bounds}")


def _positive_integer(value: object, name: str, *, odd: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 1
        or (odd and value % 2 != 1)
    ):
        adjective = "odd positive" if odd else "positive"
        raise ValueError(f"{name} must be an {adjective} integer")


@dataclass(frozen=True)
class EventConfig:
    threshold: float = 0.5
    median_window_frames: int = 1
    min_duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _finite_number(self.threshold, "threshold", upper=1)
        _positive_integer(self.median_window_frames, "median_window_frames", odd=True)
        _finite_number(self.min_duration_seconds, "min_duration_seconds")

    def to_dict(self) -> dict:
        return {
            "threshold": float(self.threshold),
            "median_window_frames": int(self.median_window_frames),
            "min_duration_seconds": float(self.min_duration_seconds),
        }


@dataclass(frozen=True)
class EvaluationConfig:
    event: EventConfig = field(default_factory=EventConfig)
    onset_tolerance_seconds: float = 0.2
    allow_ambiguous_reference: bool = False
    outside_family_threshold: float = 0.5
    outside_family_top_k: int = 5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.event, EventConfig):
            raise ValueError("event must be an EventConfig")
        self.event.validate()
        _finite_number(self.onset_tolerance_seconds, "onset_tolerance_seconds")
        if not isinstance(self.allow_ambiguous_reference, bool):
            raise ValueError("allow_ambiguous_reference must be boolean")
        _finite_number(self.outside_family_threshold, "outside_family_threshold", upper=1)
        _positive_integer(self.outside_family_top_k, "outside_family_top_k")

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "event": self.event.to_dict(),
            "onset_tolerance_seconds": float(self.onset_tolerance_seconds),
            "allow_ambiguous_reference": self.allow_ambiguous_reference,
            "outside_family_threshold": float(self.outside_family_threshold),
            "outside_family_top_k": int(self.outside_family_top_k),
        }

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationConfig":
        """Load schema 1, rejecting duplicate keys, unknown fields and nonfinite JSON."""

        def unique_object(pairs: list[tuple[str, object]]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate evaluation config JSON key: {key}")
                result[key] = value
            return result

        def nonfinite(value: str) -> None:
            raise ValueError(f"Nonfinite evaluation config JSON value: {value}")

        value = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=unique_object,
            parse_constant=nonfinite,
        )
        fields = {
            "schema_version",
            "event",
            "onset_tolerance_seconds",
            "allow_ambiguous_reference",
            "outside_family_threshold",
            "outside_family_top_k",
        }
        if not isinstance(value, dict) or set(value) - fields:
            raise ValueError("Evaluation config must be an object with known fields")
        if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
            raise ValueError("Evaluation config requires schema_version 1")
        event = value.get("event", {})
        if not isinstance(event, dict) or set(event) - {
            "threshold",
            "median_window_frames",
            "min_duration_seconds",
        }:
            raise ValueError("event must be an object with known EventConfig fields")
        return cls(
            event=EventConfig(**event),
            **{key: item for key, item in value.items() if key not in {"schema_version", "event"}},
        )
