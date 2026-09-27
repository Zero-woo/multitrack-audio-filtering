"""Batch inference and offline visualization CLI orchestration."""

import hashlib
import html
import importlib
import json
import sys
from pathlib import Path

from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import load_stems
from waves_sed.ontology import AudioSetOntology
from waves_sed.phase4 import _invalid_cache_audio_path, _provenance_paths, load_prediction_index
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import CHECKPOINT_NAME, sha256
from waves_sed.reporting import _atomic_text, preflight_outputs


def add_commands(commands) -> None:
    batch = commands.add_parser(
        "batch-infer", help="Infer normalized stems with one frozen backend"
    )
    batch.add_argument("--stems", required=True, type=Path)
    batch.add_argument("--output-dir", required=True, type=Path)
    batch.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/checkpoints") / CHECKPOINT_NAME
    )
    batch.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    batch.add_argument("--threads", type=int, default=4)
    batch.add_argument("--overwrite", action="store_true", help="Recompute existing raw caches")
    visual = commands.add_parser(
        "visualize", help="Export cached stem evidence as PNG/SVG and HTML"
    )
    visual.add_argument("--stems", required=True, type=Path)
    visual.add_argument("--predictions", required=True, type=Path)
    visual.add_argument("--mappings", required=True, type=Path)
    visual.add_argument("--config", required=True, type=Path)
    visual.add_argument("--ontology", type=Path)
    visual.add_argument("--output-dir", required=True, type=Path)
    visual.add_argument("--format", choices=["png", "svg"], default="png")
    visual.add_argument("--max-waveform-points", type=int, default=4000)
    visual.add_argument(
        "--stem-id", help="Render one exact stem ID; default is the complete manifest"
    )


def _gallery(records: list[dict]) -> str:
    """A portable, escaped gallery, with no remote assets or executable metadata."""
    cards = []
    escape = html.escape
    for item in records:
        report = item["evaluation"]
        description = report.get("source_description") or "Description unavailable"
        reference = report["reference"]
        details = (
            f"Role: {report.get('role') or 'unknown'} · "
            f"Evaluation: {report['evaluation_status']} · "
            f"Reference: {reference['status'] or 'unknown'}"
        )
        reasons = ", ".join(report["reason_codes"])
        if item["status"] == "rendered":
            relative = escape(item["path"], quote=True)
            media = (
                f'<a href="{relative}"><img loading="lazy" src="{relative}" '
                f'alt="{escape(description, quote=True)} temporal evidence"></a>'
            )
        else:
            media = (
                f'<p class="error">Plot unavailable: {escape(item["error"] or "unknown error")}</p>'
            )
        cards.append(
            "<article><h2>"
            + escape(description)
            + '</h2><p class="id">'
            + escape(item["stem_id"])
            + "</p><p>"
            + escape(details)
            + "</p>"
            + ('<p class="reason">' + escape(reasons) + "</p>" if reasons else "")
            + media
            + "</article>"
        )
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stem temporal evidence</title><style>
body{font:16px/1.5 system-ui,sans-serif;background:#f4f6f8;color:#142c40;margin:0}
main{max-width:1280px;margin:32px auto;padding:0 24px}h1{margin-bottom:8px}
article{background:white;border:1px solid #d5dfe5;border-radius:8px;padding:20px;margin:24px 0}
h2{font-size:20px;margin:0;overflow-wrap:anywhere}p{margin:8px 0}.id{color:#536875;overflow-wrap:anywhere}
img{display:block;max-width:100%;height:auto;margin:16px auto 0}.reason,.error{color:#874500}
a{color:#005f87}</style></head><body><main><h1>Stem temporal evidence</h1>
<p>WAVES planned support measures internal consistency, not video synchronization.
Unavailable evidence stays unavailable. No quality decision is assigned.</p>
<p><a href="visualization-index.json">Evaluation settings, provenance and plot index (JSON)</a></p>
"""
        + ("\n".join(cards) if cards else "<p>No stems selected.</p>")
        + "\n</main></body></html>\n"
    )


def _visualize(args) -> int:
    if args.max_waveform_points < 2:
        raise ValueError("--max-waveform-points must be at least 2")
    stems = load_stems(args.stems)
    index = load_prediction_index(args.predictions, {stem.stem_id for stem in stems})
    mapping = ManualMapping.load(args.mappings, AudioSetOntology.load(args.ontology))
    config = EvaluationConfig.load(args.config)
    selected = (
        stems if args.stem_id is None else [stem for stem in stems if stem.stem_id == args.stem_id]
    )
    if args.stem_id is not None and not selected:
        raise ValueError(f"Unknown stem ID: {args.stem_id}")
    inputs = {
        "stems": args.stems,
        "predictions": args.predictions,
        "mappings": args.mappings,
        "config": args.config,
    }
    if args.ontology is not None:
        inputs["ontology"] = args.ontology
    protected = [*inputs.values(), *index.values(), *_provenance_paths(stems)]
    # Inspect all cache headers before any plot write, including later/unselected
    # stems and malformed scores whose metadata still records original sources.
    for cache in index.values():
        audio = _invalid_cache_audio_path(cache)
        if audio:
            protected.extend([Path(audio), cache.parent / audio])
    paths = {
        stem.stem_id: f"plots/{hashlib.sha256(stem.stem_id.encode('utf-8')).hexdigest()}.{args.format}"
        for stem in selected
    }
    outputs = [args.output_dir / path for path in paths.values()]
    outputs.extend([args.output_dir / "index.html", args.output_dir / "visualization-index.json"])
    preflight_outputs(outputs, protected)
    if selected:
        try:
            for name in ("matplotlib", "soundfile"):
                importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                'Visualization requires matplotlib and soundfile; install "waves-stem-sed[visualization]" or ".[visualization]"'
            ) from exc
    from waves_sed.visualization import render_stem

    records = []
    for stem in selected:
        cache = index.get(stem.stem_id)
        prediction, error = None, None
        if cache is None or not cache.exists():
            error = "missing_prediction"
        else:
            try:
                prediction = FramePrediction.load(cache)
            except (OSError, ValueError) as exc:
                error = f"invalid_prediction: {exc}"
        report = evaluate_stem(
            stem,
            prediction,
            mapping,
            config,
            prediction_path=cache,
            prediction_error=error,
        )
        record = {
            "stem_id": stem.stem_id,
            "path": paths[stem.stem_id],
            "status": "failed",
            "evaluation": report,
            "visualization": None,
            "error": None,
        }
        try:
            record["visualization"] = render_stem(
                stem,
                prediction,
                report,
                mapping,
                config,
                args.output_dir / paths[stem.stem_id],
                max_waveform_points=args.max_waveform_points,
                protected_inputs=protected,
            )
            record["status"] = "rendered"
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            record["error"] = str(exc)
            print(f"waves-sed: plot {stem.stem_id}: {exc}", file=sys.stderr)
        records.append(record)
    failed = sum(row["status"] == "failed" for row in records)
    result = {
        "schema_version": 1,
        "summary": {
            "stem_count": len(records),
            "rendered_count": len(records) - failed,
            "failed_count": failed,
        },
        "stems": records,
        "provenance": {
            "inputs": {
                key: {"path": str(path.resolve()), "sha256": sha256(path)}
                for key, path in inputs.items()
            },
            "evaluation_config": config.to_dict(),
            "format": args.format,
            "max_waveform_points": args.max_waveform_points,
            "inference_performed": False,
        },
        "files": {"gallery": "index.html"},
        "index_policy": "only rendered rows are current plots; old/unindexed files are not current results",
        "decision": None,
    }
    _atomic_text(args.output_dir / "index.html", _gallery(records))
    _atomic_text(
        args.output_dir / "visualization-index.json",
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), **result["summary"]}, ensure_ascii=False
        )
    )
    return 1 if failed else 0


def run(args) -> int:
    if args.command == "visualize":
        return _visualize(args)
    from waves_sed.batch import run_batch

    input_provenance = {"stems": {"path": str(args.stems.resolve()), "sha256": sha256(args.stems)}}
    result = run_batch(
        load_stems(args.stems),
        args.output_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        threads=args.threads,
        overwrite=args.overwrite,
        protected_inputs=[args.stems],
        input_provenance=input_provenance,
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), **result["summary"]},
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 1 if result["summary"]["failed_count"] else 0
