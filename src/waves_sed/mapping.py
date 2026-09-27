"""Explicit source descriptions to ontology families; no semantic guessing."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from waves_sed.labels import load_labels
from waves_sed.ontology import AudioSetOntology
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate mapping JSON key: {key}")
        result[key] = value
    return result


def normalize_description(text: str) -> str:
    """Only case and whitespace are normalized; no fuzzy/substring matching."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Source description must be a nonempty string")
    return " ".join(text.split()).casefold()


@dataclass(frozen=True)
class MappingRule:
    key: str
    descriptions: tuple[str, ...]
    allowed_class_ids: tuple[str, ...]
    foreign_class_ids: tuple[str, ...]
    include_descendants: bool


@dataclass(frozen=True)
class MappingResolution:
    status: str
    mapping_key: str | None
    reason: str | None
    allowed_class_ids: tuple[str, ...] = ()
    target_class_ids: tuple[str, ...] = ()
    foreign_class_ids: tuple[str, ...] = ()
    unavailable_class_ids: tuple[str, ...] = ()
    ontology_missing_class_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


class ManualMapping:
    def __init__(
        self, rules: tuple[MappingRule, ...], ontology: AudioSetOntology, provenance: dict
    ):
        self.rules = rules
        self.ontology = ontology
        self.provenance = provenance
        self._by_description: dict[str, MappingRule] = {}
        for rule in rules:
            for text in rule.descriptions:
                key = normalize_description(text)
                if key in self._by_description:
                    raise ValueError(f"Duplicate/ambiguous mapping description: {text!r}")
                self._by_description[key] = rule

    @classmethod
    def load(cls, path: str | Path, ontology: AudioSetOntology | None = None) -> "ManualMapping":
        path = Path(path).resolve()
        ontology = ontology if ontology is not None else AudioSetOntology.load()
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or set(value) != {"schema_version", "mappings"}:
            raise ValueError("Mapping config requires exactly schema_version and mappings")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported mapping schema_version")
        if not isinstance(value["mappings"], dict):
            raise ValueError("mappings must be an object keyed by mapping name")
        rules = []
        model_ids = set(load_labels()[0])
        for key, item in value["mappings"].items():
            if not isinstance(key, str) or not key.strip() or not isinstance(item, dict):
                raise ValueError("Each mapping must have a nonempty name and an object value")
            if set(item) - {
                "descriptions",
                "allowed_classes",
                "foreign_classes",
                "include_descendants",
                "notes",
            }:
                raise ValueError(f"Unknown mapping fields in {key}")
            descriptions = item.get("descriptions")
            if not isinstance(descriptions, list) or not descriptions:
                raise ValueError(f"{key}: descriptions must be a nonempty list")
            for text in descriptions:
                normalize_description(text)
            include_descendants = item.get("include_descendants", False)
            if not isinstance(include_descendants, bool):
                raise ValueError(f"{key}: include_descendants must be boolean")
            if "notes" in item and not isinstance(item["notes"], str):
                raise ValueError(f"{key}: notes must be a string")

            def class_ids(field: str, required: bool) -> tuple[str, ...]:
                values = item.get(field, [])
                if (
                    not isinstance(values, list)
                    or (required and not values)
                    or any(not isinstance(v, str) or not v for v in values)
                ):
                    raise ValueError(f"{key}: {field} must be a list of ontology class IDs")
                if len(set(values)) != len(values):
                    raise ValueError(f"{key}: duplicate {field}")
                for class_id in values:
                    if class_id not in ontology.nodes and class_id not in model_ids:
                        raise ValueError(
                            f"Unknown class ID in ontology and model metadata: {class_id}"
                        )
                return tuple(values)

            allowed = class_ids("allowed_classes", required=True)
            foreign = class_ids("foreign_classes", required=False)
            expanded = set(allowed)
            if include_descendants:
                for class_id in allowed:
                    if class_id not in ontology.nodes:
                        raise ValueError(
                            f"Cannot expand descendants without ontology entry: {class_id}"
                        )
                    expanded.update(ontology.descendants(class_id, include_self=True))
            if expanded.intersection(foreign):
                raise ValueError(f"{key}: allowed family overlaps explicit foreign classes")
            rules.append(
                MappingRule(
                    key, tuple(descriptions), tuple(sorted(expanded)), foreign, include_descendants
                )
            )
        return cls(
            tuple(rules),
            ontology,
            {
                "path": str(path),
                "sha256": sha256(path),
                "description_matching": "casefold_and_collapsed_whitespace_exact",
                "ontology": ontology.provenance,
            },
        )

    def resolve(
        self, source_description: str | None, prediction_class_ids: tuple[str, ...] | None = None
    ) -> MappingResolution:
        if source_description is None:
            return MappingResolution("unsupported_mapping", None, "missing_source_description")
        if not isinstance(source_description, str):
            raise ValueError("Source description must be a string or None")
        if not source_description.strip():
            return MappingResolution("unsupported_mapping", None, "missing_source_description")
        rule = self._by_description.get(normalize_description(source_description))
        if rule is None:
            return MappingResolution("unsupported_mapping", None, "no_explicit_mapping")
        available = load_labels()[0] if prediction_class_ids is None else prediction_class_ids
        if any(not isinstance(class_id, str) or not class_id.strip() for class_id in available):
            raise ValueError("Prediction class IDs must be nonempty strings")
        if len(set(available)) != len(available):
            raise ValueError("Duplicate prediction class IDs")
        target = tuple(class_id for class_id in available if class_id in rule.allowed_class_ids)
        unavailable = tuple(sorted(set(rule.allowed_class_ids) - set(available)))
        return MappingResolution(
            "supported" if target else "unsupported_mapping",
            rule.key,
            None if target else "no_allowed_classes_in_backend",
            rule.allowed_class_ids,
            target,
            rule.foreign_class_ids,
            unavailable,
            tuple(
                sorted(
                    (set(rule.allowed_class_ids) | set(rule.foreign_class_ids))
                    - self.ontology.nodes.keys()
                )
            ),
        )


def aggregate_target(prediction: FramePrediction, resolution: MappingResolution) -> np.ndarray:
    """P_target(t) = max allowed modeled class probabilities, without thresholding."""
    prediction.validate()
    if resolution.status != "supported" or not resolution.target_class_ids:
        raise ValueError("Cannot aggregate an unsupported mapping")
    columns = {class_id: index for index, class_id in enumerate(prediction.class_ids)}
    missing = set(resolution.target_class_ids) - columns.keys()
    if missing:
        raise ValueError(f"Prediction is missing mapped classes: {sorted(missing)}")
    indices = [columns[class_id] for class_id in resolution.target_class_ids]
    return prediction.probabilities[:, indices].max(axis=1)
