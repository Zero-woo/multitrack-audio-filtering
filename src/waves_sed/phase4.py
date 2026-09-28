"""Evaluate supplied raw caches without importing or running an inference backend."""

import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import load_stems
from waves_sed.ontology import AudioSetOntology
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256
from waves_sed.reporting import write_reports


def add_commands(commands) -> None:
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate normalized stems using raw NPZ caches"
    )
    evaluate.add_argument("--stems", required=True, type=Path, help="adapt-waves JSON manifest")
    evaluate.add_argument(
        "--predictions", required=True, type=Path, help="Stem ID to raw NPZ index"
    )
    evaluate.add_argument("--mappings", required=True, type=Path)
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument(
        "--filter-config", type=Path, help="Optional explicit quality decision thresholds"
    )
    evaluate.add_argument("--ontology", type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)


def _unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate prediction index JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite prediction index JSON value: {value}")


def load_prediction_index(path: Path, stem_ids: set[str]) -> dict[str, Path]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=_unique_object,
        parse_constant=_nonfinite,
    )
    if not isinstance(value, dict) or set(value) != {"schema_version", "predictions"}:
        raise ValueError("Prediction index requires exactly schema_version and predictions")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported prediction index schema_version")
    if not isinstance(value["predictions"], dict):
        raise ValueError("Prediction index predictions must be an object keyed by stem ID")
    result = {}
    for stem_id, raw_path in value["predictions"].items():
        if stem_id not in stem_ids:
            raise ValueError(f"Unknown stem ID in prediction index: {stem_id}")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Prediction path must be a nonempty string for {stem_id}")
        result[stem_id] = (path.resolve().parent / raw_path).resolve()
    return result


def _provenance_paths(stems) -> list[Path]:
    protected = []
    for stem in stems:
        if stem.audio_path is not None:
            protected.append(stem.audio_path)
        for raw_path in stem.provenance.get("source_media_paths", []):
            if isinstance(raw_path, str) and raw_path:
                protected.append(Path(raw_path))
        inputs = stem.provenance.get("inputs", {})
        if isinstance(inputs, dict):
            for item in inputs.values():
                if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"]:
                    protected.append(Path(item["path"]))
    return protected


def _invalid_cache_audio_path(path: Path) -> str | None:
    """Protect parseable source metadata even when score validation has failed.

    Only the small metadata member is read, with pickle disabled. This does not
    rehabilitate the cache or make its scores usable for evaluation.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(data["metadata_json"].item())
        audio = metadata.get("audio_path") if isinstance(metadata, dict) else None
        return audio if isinstance(audio, str) and audio else None
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        return None


def run(args) -> int:
    stems = load_stems(args.stems)
    index = load_prediction_index(args.predictions, {stem.stem_id for stem in stems})
    ontology = AudioSetOntology.load(args.ontology)
    mapping = ManualMapping.load(args.mappings, ontology)
    config = EvaluationConfig.load(args.config)
    filter_config = None
    if args.filter_config is not None:
        from waves_sed.decision import FilterConfig

        filter_config = FilterConfig.load(args.filter_config)
    input_paths = {
        "stems": args.stems,
        "predictions": args.predictions,
        "mappings": args.mappings,
        "config": args.config,
    }
    if args.ontology is not None:
        input_paths["ontology"] = args.ontology
    if args.filter_config is not None:
        input_paths["filter_config"] = args.filter_config
    protected = [*input_paths.values(), *index.values(), *_provenance_paths(stems)]
    records = []
    for stem in stems:
        cache_path = index.get(stem.stem_id)
        prediction = None
        error = None
        if cache_path is None or not cache_path.exists():
            error = "missing_prediction"
        else:
            try:
                prediction = FramePrediction.load(cache_path)
            except (ValueError, OSError) as problem:
                error = f"invalid_prediction: {problem}"
        audio = (
            prediction.metadata.get("audio_path")
            if prediction is not None
            else _invalid_cache_audio_path(cache_path)
            if cache_path is not None
            else None
        )
        if isinstance(audio, str) and audio:
            protected.extend([Path(audio), cache_path.parent / audio])
        records.append(
            evaluate_stem(
                stem,
                prediction,
                mapping,
                config,
                prediction_path=cache_path,
                prediction_error=error,
                filter_config=filter_config,
            )
        )
    if (
        filter_config is not None
        and sha256(args.filter_config) != filter_config.provenance["sha256"]
    ):
        raise ValueError("Filter config changed during evaluation; no reports were written")
    dataset = write_reports(
        records,
        args.output_dir,
        protected_inputs=protected,
        provenance={
            "inputs": {
                key: {"path": str(path.resolve()), "sha256": sha256(path)}
                for key, path in input_paths.items()
            },
            "inference_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "dataset": str((args.output_dir / "dataset.json").resolve()),
                **{
                    key: value
                    for key, value in dataset["summary"].items()
                    if key not in {"roles", "aggregation_context"}
                },
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0
