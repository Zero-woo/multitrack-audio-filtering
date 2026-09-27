"""Model-independent metadata for a final WAVES stem."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class StemMetadata:
    """Keep final semantics separate from the selected parent's original plan.

    ``expected_intervals`` are planned support, never video ground truth. Consult
    ``provenance['reference_status']`` before using them as a temporal reference.
    """

    stem_id: str
    candidate_id: str
    clip_key: str
    audio_path: Path | None
    source_description: str | None
    role: str | None
    expected_intervals: tuple[tuple[float, float], ...] | None
    optional_video_id: str | None = None
    description_origin: str | None = None
    planned_description: str | None = None
    planned_label: str | None = None
    planned_role: str | None = None
    merged_candidate_ids: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return plain JSON types, including portable string paths."""
        result = asdict(self)
        result["audio_path"] = str(self.audio_path) if self.audio_path is not None else None
        result["expected_intervals"] = (
            [list(interval) for interval in self.expected_intervals]
            if self.expected_intervals is not None
            else None
        )
        result["merged_candidate_ids"] = list(self.merged_candidate_ids)
        result["issues"] = list(self.issues)
        return result

    @classmethod
    def from_dict(cls, value: dict, *, base_path: Path | None = None) -> "StemMetadata":
        """Read a normalized stem; Phase 3 mapping hints are intentionally recomputed.

        Relative paths are resolved against the manifest directory. Missing
        provenance remains missing; this loader never upgrades a temporal plan.
        """
        from waves_sed.temporal_metrics import validate_intervals

        names = {item.name for item in fields(cls)}
        required = {
            "stem_id",
            "candidate_id",
            "clip_key",
            "audio_path",
            "source_description",
            "role",
            "expected_intervals",
        }
        if not isinstance(value, dict) or required - value.keys():
            raise ValueError("Normalized stem is missing required fields")
        if value.keys() - names - {"mapping"}:
            raise ValueError("Unknown normalized stem fields")
        values = {key: item for key, item in value.items() if key in names}
        for name in ("stem_id", "candidate_id", "clip_key"):
            if not isinstance(values[name], str) or not values[name].strip():
                raise ValueError(f"Stem {name} must be a nonempty string")
        for name in (
            "source_description",
            "role",
            "optional_video_id",
            "description_origin",
            "planned_description",
            "planned_label",
            "planned_role",
        ):
            if values.get(name) is not None and not isinstance(values[name], str):
                raise ValueError(f"Stem {name} must be a string or null")
        base = Path(base_path) if base_path is not None else Path.cwd()

        def resolve_path(item):
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Stem paths must be nonempty strings")
            path = Path(item)
            return (base / path).resolve() if not path.is_absolute() else path.resolve()

        if values["audio_path"] is not None:
            values["audio_path"] = resolve_path(values["audio_path"])
        if values["expected_intervals"] is not None:
            values["expected_intervals"] = validate_intervals(values["expected_intervals"])
            if any(
                left[0] > right[0]
                for left, right in zip(
                    values["expected_intervals"], values["expected_intervals"][1:]
                )
            ):
                raise ValueError("Normalized expected intervals must be ordered by start")
        for name in ("merged_candidate_ids", "issues"):
            items = values.get(name, [])
            if not isinstance(items, list) or any(
                not isinstance(item, str) or not item.strip() for item in items
            ):
                raise ValueError(f"Stem {name} must be a list of nonempty strings")
            if len(set(items)) != len(items):
                raise ValueError(f"Duplicate stem {name}")
            values[name] = tuple(items)
        provenance = values.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("Stem provenance must be an object")
        try:
            provenance = json.loads(json.dumps(provenance, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise ValueError("Stem provenance must be finite JSON data") from exc
        if "source_media_paths" in provenance:
            if not isinstance(provenance["source_media_paths"], list):
                raise ValueError("source_media_paths must be a list")
            provenance["source_media_paths"] = [
                str(resolve_path(item)) for item in provenance["source_media_paths"]
            ]
        if "inputs" in provenance:
            if not isinstance(provenance["inputs"], dict):
                raise ValueError("Stem provenance inputs must be an object")
            for item in provenance["inputs"].values():
                if not isinstance(item, dict):
                    raise ValueError("Stem provenance input must be an object")
                if "path" in item:
                    item["path"] = str(resolve_path(item["path"]))
        if "raw_final_stem" in provenance and not isinstance(provenance["raw_final_stem"], dict):
            raise ValueError("raw_final_stem provenance must be an object")
        values["provenance"] = provenance
        return cls(**values)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate normalized metadata field: {key}")
        result[key] = value
    return result


def load_stems(path: str | Path) -> list[StemMetadata]:
    """Load the schema-1 manifest written by ``adapt-waves`` without a model."""
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    # This also rejects nonfinite data in informational fields we do not consume.
    json.dumps(value, allow_nan=False)
    allowed = {
        "schema_version",
        "input_format",
        "temporal_reference",
        "summary",
        "stems",
        "mapping_provenance",
        "vocabulary_coverage",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Invalid normalized metadata manifest")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported normalized metadata schema_version")
    if not isinstance(value.get("stems"), list):
        raise ValueError("Normalized metadata stems must be a list")
    stems = [StemMetadata.from_dict(item, base_path=path.parent) for item in value["stems"]]
    if len({stem.stem_id for stem in stems}) != len(stems):
        raise ValueError("Duplicate normalized stem_id")
    if len({(stem.clip_key, stem.candidate_id) for stem in stems}) != len(stems):
        raise ValueError("Duplicate candidate in a normalized clip")
    return stems
