"""Optional, explicit quality policy over already verified temporal measurements.

These thresholds are user policy, not calibrated model accuracy claims. No rule
uses outside-family scores, changes a measurement, or establishes video truth.
"""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path


class FilterDecision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    UNSUPPORTED = "UNSUPPORTED"


# role -> rule -> (reported metric, direction, upper bound on metric values)
_RULES = {
    "onset": {
        "min_event_recall": ("event_recall", "min", 1.0),
        "max_mean_onset_error_ms": ("mean_onset_error_ms", "max", None),
    },
    "span": {"min_tiou": ("temporal_iou", "min", 1.0)},
    "ambience": {"min_occupancy": ("occupancy_in_expected_span", "min", 1.0)},
}


@dataclass(frozen=True)
class _Threshold:
    role: str
    rule: str
    pass_threshold: float
    fail_threshold: float


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bound(value: object, name: str, upper: float | None) -> float:
    result = _number(value)
    if result is None or result < 0 or (upper is not None and result > upper):
        bounds = "[0, 1]" if upper is not None else "[0, infinity)"
        raise ValueError(f"{name} must be a finite number in {bounds}")
    return result


class FilterConfig:
    """A normalized policy with no enabled rules by default.

    Use ``from_dict`` for programmatic policies or ``load`` to retain the exact
    input file's path and digest. Returned dictionaries never expose internals.
    """

    def __init__(self) -> None:
        self._rules: tuple[_Threshold, ...] = ()
        self._provenance: dict = {}

    @property
    def enabled(self) -> bool:
        return bool(self._rules)

    @property
    def provenance(self) -> dict:
        return deepcopy(self._provenance)

    def to_dict(self) -> dict:
        value = {role: dict.fromkeys(rules) for role, rules in _RULES.items()}
        for bound in self._rules:
            value[bound.role][bound.rule] = {
                "pass": bound.pass_threshold,
                "fail": bound.fail_threshold,
            }
        return {"schema_version": 1, "filter": value}

    @classmethod
    def from_dict(cls, value: dict) -> "FilterConfig":
        if not isinstance(value, dict) or set(value) != {"schema_version", "filter"}:
            raise ValueError("Filter config requires exactly schema_version and filter")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Filter config requires schema_version 1")
        filtering = value["filter"]
        if not isinstance(filtering, dict) or set(filtering) - _RULES.keys():
            raise ValueError("filter must be an object with known roles")
        bounds = []
        for role, definitions in _RULES.items():
            rules = filtering.get(role, {})
            if not isinstance(rules, dict) or set(rules) - definitions.keys():
                raise ValueError(f"filter.{role} must be an object with known rules")
            for rule, (_, direction, upper) in definitions.items():
                setting = rules.get(rule)
                if setting is None:
                    continue
                if isinstance(setting, dict):
                    if set(setting) != {"pass", "fail"}:
                        raise ValueError(f"{role}.{rule} band requires exactly pass and fail")
                    passed = _bound(setting["pass"], f"{role}.{rule}.pass", upper)
                    failed = _bound(setting["fail"], f"{role}.{rule}.fail", upper)
                else:
                    passed = failed = _bound(setting, f"{role}.{rule}", upper)
                if (direction == "min" and failed > passed) or (
                    direction == "max" and passed > failed
                ):
                    ordering = "fail <= pass" if direction == "min" else "pass <= fail"
                    raise ValueError(f"{role}.{rule} requires {ordering}")
                bounds.append(_Threshold(role, rule, passed, failed))
        config = cls()
        config._rules = tuple(bounds)
        return config

    @classmethod
    def load(cls, path: str | Path) -> "FilterConfig":
        def unique_object(pairs: list[tuple[str, object]]) -> dict:
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"Duplicate filter config JSON key: {key}")
                value[key] = item
            return value

        def nonfinite(value: str) -> None:
            raise ValueError(f"Nonfinite filter config JSON value: {value}")

        path = Path(path).resolve()
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=unique_object,
            parse_constant=nonfinite,
        )
        config = cls.from_dict(value)
        config._provenance = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        return config


def _guards(report: dict) -> list[str]:
    reasons = []
    reference = report.get("reference")
    if not isinstance(reference, dict):
        reasons.append("missing_reference")
    else:
        if (
            not isinstance(reference.get("origin"), str)
            or not reference["origin"].strip()
            or not isinstance(reference.get("status"), str)
            or not reference["status"].strip()
            or reference.get("intervals") is None
        ):
            reasons.append("missing_reference")
        if reference.get("usable") is not True or reference.get("reason") is not None:
            reasons.append("unusable_reference")
        status = reference.get("status")
        if status == "ambiguous":
            reasons.append("ambiguous_reference")
        elif status in ("missing", "unavailable", "unknown"):
            reasons.append("unusable_reference")
        provenance = reference.get("provenance")
        if isinstance(provenance, dict) and provenance.get("ambiguity_signals"):
            reasons.append("ambiguous_reference")
    if report.get("evaluation_status") != "evaluated":
        reasons.append("evaluation_unavailable")
    if report.get("detection_status") != "available":
        reasons.append("detection_unavailable")
    cache = report.get("cache_validation")
    if not isinstance(cache, dict) or cache.get("verified") is not True:
        reasons.append("unverified_cache")
    return list(dict.fromkeys(reasons))


def _check(bound: _Threshold, metrics: object, blocked: list[str], decision: str) -> dict:
    metric, direction, upper = _RULES[bound.role][bound.rule]
    raw = metrics.get(metric) if isinstance(metrics, dict) else None
    value = _number(raw)
    check = {
        "rule": bound.rule,
        "metric": metric,
        "direction": direction,
        "value": value,
        "pass_threshold": bound.pass_threshold,
        "fail_threshold": bound.fail_threshold,
        "status": "blocked" if blocked else "checked",
        "decision": decision,
        "reason_codes": list(blocked),
    }
    if blocked:
        return check
    if not isinstance(metrics, dict) or metric not in metrics:
        reason = "missing_metric"
    elif raw is None:
        reason = "unavailable_metric"
    elif value is None or value < 0 or (upper is not None and value > upper):
        reason = "invalid_metric"
    else:
        reason = None
    if reason is not None:
        check.update(status="unavailable", decision="REVIEW", reason_codes=[reason])
        return check
    passed = value >= bound.pass_threshold if direction == "min" else value <= bound.pass_threshold
    failed = value < bound.fail_threshold if direction == "min" else value > bound.fail_threshold
    outcome = "PASS" if passed else "FAIL" if failed else "REVIEW"
    check.update(
        decision=outcome,
        reason_codes=[
            {"PASS": "threshold_passed", "FAIL": "threshold_failed", "REVIEW": "review_band"}[
                outcome
            ]
        ],
    )
    return check


def decide(report: dict, config: FilterConfig) -> dict:
    """Audit enabled rules without mutating the report or its measurements.

    Unsupported roles/mappings cannot produce quality decisions. Reference or
    evidence guards produce REVIEW before evaluating any quality threshold.
    Within trusted evidence, a valid FAIL dominates an unavailable other metric.
    """
    if not isinstance(report, dict) or not isinstance(config, FilterConfig):
        raise ValueError("decide requires a report object and FilterConfig")
    audit = {
        "decision": None,
        "status": "disabled",
        "reason_codes": ["filter_disabled"],
        "checks": [],
        "config": config.to_dict(),
        "config_provenance": config.provenance,
        "implementation_version": 1,
    }
    if not config.enabled:
        return audit
    role = report.get("role")
    if not isinstance(role, str) or role not in _RULES:
        audit.update(decision="UNSUPPORTED", status="decided", reason_codes=["unsupported_role"])
        return audit
    bounds = [bound for bound in config._rules if bound.role == role]
    if not bounds:
        audit.update(status="not_configured", reason_codes=["role_filter_not_configured"])
        return audit
    audit["status"] = "decided"
    if report.get("mapping_status") != "supported":
        blocked, outcome = ["unsupported_mapping"], "UNSUPPORTED"
    else:
        blocked = _guards(report)
        outcome = "REVIEW" if blocked else "PASS"
    checks = [_check(bound, report.get("metrics"), blocked, outcome) for bound in bounds]
    audit["checks"] = checks
    if blocked:
        audit.update(decision=outcome, reason_codes=blocked)
        return audit
    outcomes = {check["decision"] for check in checks}
    audit["decision"] = (
        "FAIL" if "FAIL" in outcomes else "REVIEW" if "REVIEW" in outcomes else "PASS"
    )
    audit["reason_codes"] = [
        f"{check['rule']}_{reason}" for check in checks for reason in check["reason_codes"]
    ]
    return audit
