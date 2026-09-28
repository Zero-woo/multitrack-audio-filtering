"""Run a real-model plumbing experiment with explicitly synthetic full-track support.

This is a demonstration, not a WAVES/video annotation or a sensitivity benchmark.
The original WAV is never rewritten. Choose a new output directory for each run.
"""

import argparse
import json
from pathlib import Path

from waves_sed.batch import run_batch
from waves_sed.corruption_experiment import generate_experiment, recorded_paths
from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig, EventConfig
from waves_sed.labels import load_labels
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import StemMetadata, load_stems
from waves_sed.phase4 import load_prediction_index
from waves_sed.phase5 import _gallery
from waves_sed.provenance import CHECKPOINT_NAME, sha256
from waves_sed.reporting import _atomic_text, preflight_outputs, write_reports
from waves_sed.temporal_reference import TemporalSupport


class SyntheticFullTrackReference:
    def resolve(self, stem, *, allow_ambiguous=False):
        return TemporalSupport(
            origin="synthetic_full_track_demonstration",
            status="demonstration",
            intervals=stem.expected_intervals,
            usable=True,
            reason=None,
            provenance={
                "provider": type(self).__name__,
                "synthetic_support": True,
                "video_annotation": False,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--corruptions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/checkpoints") / CHECKPOINT_NAME
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import soundfile

    from waves_sed.cli import main as cli
    from waves_sed.prediction import FramePrediction
    from waves_sed.visualization import render_stem

    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Demonstration output directory must be new")
    ids, names = load_labels()
    if args.class_id not in ids:
        raise ValueError("Explicit class ID must belong to the actual model vocabulary")
    name = names[ids.index(args.class_id)]
    audio = args.audio.resolve()
    digest = sha256(audio)
    info = soundfile.info(audio)
    stem = StemMetadata(
        stem_id="phase6-demo::source",
        candidate_id="source",
        clip_key="phase6-demo",
        audio_path=audio,
        source_description=name,
        role="span",
        expected_intervals=((0.0, info.frames / info.samplerate),),
        issues=("synthetic_demo_metadata",),
        provenance={
            "reference_origin": "synthetic_full_track_demonstration",
            "reference_status": "demonstration",
            "raw_final_stem": {"sha256": digest},
            "source_media_paths": [str(audio)],
            "video_annotation": False,
        },
    )
    inputs = output / "inputs"
    payloads = {
        inputs / "stems.json": {"schema_version": 1, "stems": [stem.to_dict()]},
        inputs / "mapping.json": {
            "schema_version": 1,
            "mappings": {
                "explicit_demo": {
                    "descriptions": [name],
                    "allowed_classes": [args.class_id],
                }
            },
        },
        inputs / "evaluation.json": EvaluationConfig(event=EventConfig(threshold=0.2)).to_dict(),
    }
    protected = [audio, args.corruptions, args.checkpoint, Path(__file__)]
    preflight_outputs(list(payloads), protected)
    for path, value in payloads.items():
        _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")
    experiment = generate_experiment(
        inputs / "stems.json", stem.stem_id, args.corruptions, output / "experiment"
    )
    if experiment["summary"]["failed_count"]:
        raise ValueError("Some demo edits failed; inspect experiment.json before continuing")
    manifest = output / "experiment/stems.json"
    stems = load_stems(manifest)
    batch = run_batch(
        stems,
        output / "batch",
        checkpoint=args.checkpoint,
        threads=args.threads,
        protected_inputs=[*protected, *payloads, manifest],
    )
    if batch["summary"]["failed_count"]:
        raise ValueError("Demo inference failed; inspect batch-status.json")
    index = load_prediction_index(
        output / "batch/prediction-index.json", {s.stem_id for s in stems}
    )
    mapping = ManualMapping.load(inputs / "mapping.json")
    config = EvaluationConfig.load(inputs / "evaluation.json")
    protected.extend([*payloads, manifest, *index.values()])
    protected.extend(recorded_paths(experiment, base=output / "experiment"))
    reports, cards = [], []
    for derived in stems:
        cache = index[derived.stem_id]
        prediction = FramePrediction.load(cache)
        report = evaluate_stem(
            derived,
            prediction,
            mapping,
            config,
            reference=SyntheticFullTrackReference(),
            prediction_path=cache,
        )
        reports.append(report)
        plot = Path("plots") / (cache.stem + ".png")
        render_stem(
            derived,
            prediction,
            report,
            mapping,
            config,
            output / "visuals" / plot,
            protected_inputs=protected,
        )
        cards.append(
            {
                "stem_id": derived.stem_id,
                "evaluation": report,
                "status": "rendered",
                "path": plot.as_posix(),
                "error": None,
            }
        )
    write_reports(
        reports,
        output / "reports",
        protected_inputs=protected,
        provenance={
            "synthetic_support": True,
            "video_annotation": False,
            "purpose": "real model plumbing; no recognition accuracy claim",
        },
    )
    if cli(
        [
            "compare-corruptions",
            "--experiment",
            str(output / "experiment/experiment.json"),
            "--report",
            str(output / "reports/dataset.json"),
            "--output-dir",
            str(output / "comparison"),
        ]
    ):
        raise ValueError("Comparison failed")
    _atomic_text(output / "visuals/index.html", _gallery(cards))
    _atomic_text(
        output / "visuals/visualization-index.json",
        json.dumps(
            {
                "schema_version": 1,
                "synthetic_support": True,
                "video_annotation": False,
                "stems": cards,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    if sha256(audio) != digest:
        raise AssertionError("Original audio changed")
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "original_audio_unchanged": True,
                "batch": batch["summary"],
                "synthetic_support": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
