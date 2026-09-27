"""Exercise two extraction thresholds on an existing cache; never infer annotations.

The full-track temporal reference is deliberately synthetic. Results verify the
evaluation plumbing, not recognition quality or the timing of any actual video.
"""

import argparse
import json
from pathlib import Path

from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig, EventConfig
from waves_sed.mapping import ManualMapping, MappingRule
from waves_sed.metadata import StemMetadata
from waves_sed.ontology import AudioSetOntology
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256
from waves_sed.reporting import write_reports
from waves_sed.temporal_reference import TemporalSupport


class FullTrackDemonstration:
    """An explicit interface fixture, not WAVES planned timing or an annotation."""

    def __init__(self, duration: float):
        self.duration = duration

    def resolve(self, stem, *, allow_ambiguous=False):
        return TemporalSupport(
            origin="synthetic_full_track_demonstration",
            status="demonstration",
            intervals=((0.0, self.duration),),
            usable=True,
            reason=None,
            provenance={"synthetic_support": True, "annotation": False},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    digest = sha256(args.prediction)
    prediction = FramePrediction.load(args.prediction)
    if args.class_id not in prediction.class_ids:
        raise ValueError("Explicit test class must exist in the cache")
    ontology = AudioSetOntology.load()
    # Resolve the exact known ID, without any guessed source mapping.
    ontology.get(args.class_id)
    name = prediction.class_names[prediction.class_ids.index(args.class_id)]
    mapping = ManualMapping(
        (MappingRule("explicit_test_target", (name,), (args.class_id,), (), False),),
        ontology,
        {
            "kind": "explicit_cli_test_class",
            "class_id": args.class_id,
            "ontology": ontology.provenance,
        },
    )
    stem = StemMetadata(
        stem_id="cache-demonstration::target",
        candidate_id="target",
        clip_key="cache-demonstration",
        audio_path=args.audio.resolve(),
        source_description=name,
        role="span",
        expected_intervals=None,
        provenance={"purpose": "cached evaluation plumbing only"},
    )
    reference = FullTrackDemonstration(prediction.metadata["duration_seconds"])
    protected = [args.prediction, args.audio, Path(__file__)]
    original_audio = prediction.metadata.get("audio_path")
    if isinstance(original_audio, str) and original_audio:
        protected.extend([Path(original_audio), args.prediction.resolve().parent / original_audio])
    summaries = []
    for threshold in (0.2, 0.5):
        config = EvaluationConfig(event=EventConfig(threshold=threshold))
        report = evaluate_stem(
            stem,
            prediction,
            mapping,
            config,
            reference=reference,
            prediction_path=args.prediction,
        )
        if report["evaluation_status"] != "evaluated":
            raise ValueError(f"Cache verification failed: {report['reason_codes']}")
        folder = args.output_dir / f"threshold-{threshold}"
        write_reports(
            [report],
            folder,
            protected_inputs=protected,
            provenance={"synthetic_support": True, "inference_performed": False},
        )
        summaries.append(
            {
                "threshold": threshold,
                "detected_event_count": len(report["detected_intervals"]),
                "temporal_iou": report["metrics"]["temporal_iou"],
                "output": str(folder.resolve()),
            }
        )
    if sha256(args.prediction) != digest:
        raise AssertionError("Raw prediction cache changed during evaluation")
    print(
        json.dumps(
            {
                "purpose": "plumbing test, synthetic full-track support, not video annotation",
                "class_id": args.class_id,
                "model_class_name": name,
                "frame_count": len(prediction.probabilities),
                "raw_cache_unchanged": True,
                "inference_performed": False,
                "results": summaries,
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
