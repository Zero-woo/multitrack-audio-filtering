"""Metadata/mapping CLI workflows; these never invoke a SED model."""

import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from waves_sed.adapters import load_frozen, load_materialized
from waves_sed.labels import load_labels
from waves_sed.mapping import ManualMapping, aggregate_target
from waves_sed.ontology import AudioSetOntology
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256


def add_commands(commands) -> None:
    adapt = commands.add_parser(
        "adapt-waves", help="Normalize actual WAVES metadata without inference"
    )
    source = adapt.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path, help="Materialized final/<key>/metadata.json")
    source.add_argument("--frozen-finals", type=Path, help="Frozen final decision list")
    adapt.add_argument("--sam-manifest", type=Path)
    adapt.add_argument("--dsp-report", type=Path)
    adapt.add_argument("--frozen-reports", type=Path)
    adapt.add_argument("--audio-paths", type=Path, help="Explicit frozen audio_id to path JSON")
    adapt.add_argument("--mappings", type=Path, help="Optional explicit source mapping JSON")
    adapt.add_argument("--ontology", type=Path, help="Override the bundled official ontology")
    adapt.add_argument("--output", type=Path, help="Write normalized JSON; default is stdout")
    mapping = commands.add_parser(
        "map-source", help="Resolve an explicit mapping and optional cached curve"
    )
    mapping.add_argument("--description", required=True)
    mapping.add_argument("--mappings", required=True, type=Path)
    mapping.add_argument("--ontology", type=Path)
    mapping.add_argument(
        "--prediction", type=Path, help="Raw NPZ; aggregate probabilities without inference"
    )
    mapping.add_argument("--output", type=Path)


def _class_details(ids, ontology, model_names) -> list[dict]:
    return [
        {
            "id": class_id,
            "ontology_name": ontology.nodes[class_id].name if class_id in ontology.nodes else None,
            "model_name": model_names.get(class_id),
        }
        for class_id in ids
    ]


def _resolution_dict(resolution, ontology, model_names) -> dict:
    return {
        **resolution.to_dict(),
        "allowed_classes": _class_details(resolution.allowed_class_ids, ontology, model_names),
        "foreign_classes": _class_details(resolution.foreign_class_ids, ontology, model_names),
    }


def _emit(result: dict, output: Path | None, protected: list[Path]) -> None:
    if output is None:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return
    for path in protected:
        if output.resolve() == path.resolve() or (
            output.exists() and path.exists() and output.samefile(path)
        ):
            raise ValueError(
                "JSON output must not overwrite source metadata, config, audio or prediction"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with NamedTemporaryFile(
        dir=output.parent, mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(serialized)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output.resolve()), **result["summary"]}, ensure_ascii=False))


def run(args) -> int:
    protected = [
        value
        for value in vars(args).values()
        if isinstance(value, Path) and value is not args.output
    ]
    if args.command == "adapt-waves":
        if args.metadata is not None:
            if args.frozen_reports or args.audio_paths:
                raise ValueError("--frozen-reports and --audio-paths require --frozen-finals")
            stems = load_materialized(
                args.metadata, sam_manifest=args.sam_manifest, dsp_report=args.dsp_report
            )
            input_format = "waves_materialized"
        else:
            if args.frozen_reports is None:
                raise ValueError("--frozen-finals requires --frozen-reports")
            if args.sam_manifest or args.dsp_report:
                raise ValueError("--sam-manifest and --dsp-report require --metadata")
            stems = load_frozen(
                args.frozen_finals, args.frozen_reports, audio_paths=args.audio_paths
            )
            input_format = "waves_frozen"
        rows = [stem.to_dict() for stem in stems]
        protected.extend(stem.audio_path for stem in stems if stem.audio_path is not None)
        protected.extend(
            Path(path) for stem in stems for path in stem.provenance["source_media_paths"]
        )
        result = {
            "schema_version": 1,
            "input_format": input_format,
            "temporal_reference": "waves_pass1_planned",
            "summary": {
                "stem_count": len(stems),
                "clip_count": len({stem.clip_key for stem in stems}),
                "issues": dict(
                    sorted(Counter(issue for stem in stems for issue in stem.issues).items())
                ),
            },
            "stems": rows,
        }
        if args.ontology and not args.mappings:
            raise ValueError("--ontology is used with --mappings")
        if args.mappings:
            ontology = AudioSetOntology.load(args.ontology)
            mappings = ManualMapping.load(args.mappings, ontology)
            model_names = dict(zip(*load_labels()))
            for stem, row in zip(stems, rows):
                resolution = mappings.resolve(stem.source_description)
                row["mapping"] = _resolution_dict(resolution, ontology, model_names)
            result["mapping_provenance"] = mappings.provenance
            result["vocabulary_coverage"] = ontology.vocabulary_coverage(*load_labels())
            result["summary"]["mapping_statuses"] = dict(
                sorted(Counter(row["mapping"]["status"] for row in rows).items())
            )
    else:
        ontology = AudioSetOntology.load(args.ontology)
        mappings = ManualMapping.load(args.mappings, ontology)
        prediction = FramePrediction.load(args.prediction) if args.prediction else None
        ids, names = (
            (prediction.class_ids, prediction.class_names)
            if prediction is not None
            else load_labels()
        )
        resolution = mappings.resolve(args.description, prediction_class_ids=ids)
        result = {
            "schema_version": 1,
            "source_description": args.description,
            "mapping": _resolution_dict(resolution, ontology, dict(zip(ids, names))),
            "mapping_provenance": mappings.provenance,
            "vocabulary_coverage": ontology.vocabulary_coverage(ids, names),
            "summary": {"mapping_status": resolution.status, "mapping_key": resolution.mapping_key},
            "target_probability": None,
        }
        if prediction is not None:
            source_audio = prediction.metadata.get("audio_path")
            if isinstance(source_audio, str) and source_audio:
                protected.append(Path(source_audio))
            result["prediction"] = {
                "path": str(args.prediction.resolve()),
                "sha256": sha256(args.prediction),
                "metadata": prediction.metadata,
            }
            result["frame_start_seconds"] = prediction.frame_start_seconds.tolist()
            result["frame_end_seconds"] = prediction.frame_end_seconds.tolist()
            result["summary"]["frame_count"] = len(prediction.probabilities)
            if resolution.status == "supported":
                result["target_probability"] = aggregate_target(prediction, resolution).tolist()
                result["aggregation"] = "max_allowed_class_probability"
        result["summary"]["decision"] = None
    _emit(result, args.output, protected)
    return 0
