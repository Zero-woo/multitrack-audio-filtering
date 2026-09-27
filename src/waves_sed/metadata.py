"""Model-independent metadata for a final WAVES stem."""

from dataclasses import asdict, dataclass, field
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
