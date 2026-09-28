"""Apply an explicit policy to an existing synthetic Phase 6 demo, without inference.

This verifies policy plumbing and cache reuse, not calibrated quality or video
synchronization. Only the synthetic full-track demonstration is accepted. The
original WAVs, metadata, caches and reports are never rewritten.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from waves_sed.corruption_experiment import recorded_paths
from waves_sed.decision import FilterConfig
from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import load_stems
from waves_sed.phase4 import load_prediction_index
from waves_sed.phase6 import _load
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256
from waves_sed.reporting import _atomic_text, preflight_outputs, write_reports
from waves_sed.temporal_reference import TemporalSupport

_ORIGIN = "synthetic_full_track_demonstration"
_STATUS = "demonstration"


class SyntheticFullTrackReference:
    def resolve(self, stem, *, allow_ambiguous=False):
        if (
            stem.provenance.get("reference_origin") != _ORIGIN
            or stem.provenance.get("reference_status") != _STATUS
            or stem.provenance.get("video_annotation") is not False
            or "synthetic_demo_metadata" not in stem.issues
        ):
            raise ValueError("Only explicitly synthetic Phase 6 demo metadata is accepted")
        return TemporalSupport(
            origin=_ORIGIN,
            status=_STATUS,
            intervals=stem.expected_intervals,
            usable=True,
            reason=None,
            provenance={
                "provider": type(self).__name__,
                "synthetic_support": True,
                "video_annotation": False,
            },
        )


def _key(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_unchanged(snapshots):
    for path, digest in snapshots.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during policy verification: {path}")


def _validate_reference(stem, prediction, original):
    duration = prediction.metadata.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("The original prediction must record a finite positive duration")
    expected = [[0.0, duration]]
    support = SyntheticFullTrackReference().resolve(stem)
    reference = original.get("reference") or {}
    provenance = reference.get("provenance") or {}
    if (
        support.to_dict()["intervals"] != expected
        or reference.get("origin") != _ORIGIN
        or reference.get("status") != _STATUS
        or reference.get("intervals") != expected
        or reference.get("usable") is not True
        or reference.get("reason") is not None
        or provenance.get("synthetic_support") is not True
        or provenance.get("video_annotation") is not False
        or original.get("expected_intervals") != expected
    ):
        raise ValueError("Original reports must use explicit synthetic full-track support")
    if (original.get("provenance") or {}).get("stem") != stem.to_dict():
        raise ValueError("Original report is not bound to the current normalized stem")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument("--filter-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plots", action="store_true", help="Also write PNGs and an HTML gallery")
    args = parser.parse_args()

    demo, output = args.demo_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise ValueError("Policy verification output directory must be new")
    inputs = {
        "mapping": demo / "inputs/mapping.json",
        "evaluation": demo / "inputs/evaluation.json",
        "stems": demo / "experiment/stems.json",
        "predictions": demo / "batch/prediction-index.json",
        "original_report": demo / "reports/dataset.json",
        "filter_config": args.filter_config.resolve(),
        "verification_script": Path(__file__).resolve(),
    }
    experiment = demo / "experiment/experiment.json"
    if experiment.is_file():
        inputs["experiment"] = experiment
    input_hashes = {path: sha256(path) for path in inputs.values()}
    stems = load_stems(inputs["stems"])
    if not stems:
        raise ValueError("The Phase 6 demonstration must contain at least one stem")
    stem_ids = {stem.stem_id for stem in stems}
    index = load_prediction_index(inputs["predictions"], stem_ids)
    if set(index) != stem_ids:
        raise ValueError("Each demonstration stem must have an existing prediction cache")
    dataset = _load(inputs["original_report"])
    rows = dataset.get("stems")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Original demonstration report must contain a list of stem reports")
    originals = {row.get("stem_id"): row for row in rows}
    if len(originals) != len(rows) or set(originals) != stem_ids:
        raise ValueError("Original report must contain exactly one row for every current stem")
    mapping = ManualMapping.load(inputs["mapping"])
    config = EvaluationConfig.load(inputs["evaluation"])
    filtering = FilterConfig.load(inputs["filter_config"])
    protected = [*inputs.values(), *index.values()]
    for key, path in inputs.items():
        if key != "verification_script":
            protected.extend(recorded_paths(_load(path), base=path.parent))
    predictions = {}
    cache_hashes = {path.resolve(): sha256(path) for path in index.values()}
    for stem in stems:
        cache = index[stem.stem_id]
        prediction = FramePrediction.load(cache)
        predictions[stem.stem_id] = prediction
        original = originals[stem.stem_id]
        _validate_reference(stem, prediction, original)
        original_prediction = (original.get("provenance") or {}).get("prediction") or {}
        if (
            original_prediction.get("sha256") != cache_hashes[cache.resolve()]
            or (original.get("provenance") or {}).get("evaluation_config") != config.to_dict()
            or ((original.get("provenance") or {}).get("mapping") or {}).get("sha256")
            != mapping.provenance["sha256"]
        ):
            raise ValueError("Original report cache, mapping or extraction settings have changed")
        protected.extend(recorded_paths(stem.to_dict(), base=inputs["stems"].parent))
        protected.extend(recorded_paths(prediction.metadata, base=cache.parent))

    snapshots = {path.resolve(): sha256(path) for path in protected if path.is_file()}
    _verify_unchanged(input_hashes)
    _verify_unchanged(cache_hashes)
    report_dir, visual_dir = output / "reports", output / "visuals"
    planned = [
        report_dir / "dataset.json",
        report_dir / "stems.csv",
        report_dir / "clips.csv",
        *(report_dir / "stems" / (_key(stem.stem_id) + ".json") for stem in stems),
        *(report_dir / "clips" / (_key(key) + ".json") for key in {s.clip_key for s in stems}),
    ]
    if args.plots:
        planned.extend(
            [
                visual_dir / "index.html",
                visual_dir / "visualization-index.json",
                *(visual_dir / "plots" / (_key(stem.stem_id) + ".png") for stem in stems),
            ]
        )
    preflight_outputs(planned, protected)
    reports = []
    for stem in stems:
        current = evaluate_stem(
            stem,
            predictions[stem.stem_id],
            mapping,
            config,
            reference=SyntheticFullTrackReference(),
            prediction_path=index[stem.stem_id],
            filter_config=filtering,
        )
        original = originals[stem.stem_id]
        for field in ("metrics", "detected_intervals", "evaluation_status", "detection_status"):
            if current[field] != original[field]:
                raise AssertionError(f"Policy changed {field} for {stem.stem_id}")
        reports.append(current)
    _verify_unchanged(snapshots)

    provenance = {
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes[path]} for name, path in inputs.items()
        },
        "inference_performed": False,
        "cache_unchanged": True,
        "metrics_unchanged": True,
        "synthetic_support": True,
        "video_annotation": False,
        "reference_origin": _ORIGIN,
        "reference_status": _STATUS,
        "purpose": "explicit policy demonstration; no calibrated quality or sensitivity claim",
        "filter_config": filtering.to_dict(),
        "filter_config_provenance": filtering.provenance,
    }
    cards = []
    if args.plots:
        from waves_sed.phase5 import _gallery
        from waves_sed.visualization import render_stem

        for stem, report in zip(stems, reports, strict=True):
            plot = Path("plots") / (_key(stem.stem_id) + ".png")
            rendered = render_stem(
                stem,
                predictions[stem.stem_id],
                report,
                mapping,
                config,
                visual_dir / plot,
                protected_inputs=protected,
                filter_config=filtering,
            )
            cards.append(
                {
                    "stem_id": stem.stem_id,
                    "evaluation": report,
                    "status": "rendered",
                    "path": plot.as_posix(),
                    "visualization": rendered,
                    "error": None,
                }
            )
    _verify_unchanged(snapshots)
    result = write_reports(reports, report_dir, protected_inputs=protected, provenance=provenance)
    if args.plots:
        _atomic_text(visual_dir / "index.html", _gallery(cards))
        _atomic_text(
            visual_dir / "visualization-index.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "summary": {
                        "stem_count": len(cards),
                        "rendered_count": len(cards),
                        "failed_count": 0,
                    },
                    "stems": cards,
                    "provenance": provenance,
                    "files": {"gallery": "index.html"},
                    "decision": None,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
    _verify_unchanged(snapshots)
    _verify_unchanged(cache_hashes)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "stem_count": len(reports),
                "decision_counts": result["summary"]["decision_summary"]["counts"],
                "inference_performed": False,
                "cache_unchanged": True,
                "metrics_unchanged": True,
                "synthetic_support": True,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
