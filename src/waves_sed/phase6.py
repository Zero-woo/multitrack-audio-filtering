"""Controlled waveform experiments and paired cached-evaluation CLI."""

import json
from pathlib import Path

from waves_sed.corruption_experiment import generate_experiment, recorded_paths
from waves_sed.provenance import sha256


def add_commands(commands) -> None:
    corrupt = commands.add_parser("corrupt", help="Create independent controlled WAV corruptions")
    corrupt.add_argument("--stems", required=True, type=Path)
    corrupt.add_argument("--stem-id", required=True)
    corrupt.add_argument("--config", required=True, type=Path)
    corrupt.add_argument("--output-dir", required=True, type=Path)
    compare = commands.add_parser(
        "compare-corruptions", help="Compare control and variant cached evaluation reports"
    )
    compare.add_argument("--experiment", required=True, type=Path)
    compare.add_argument("--report", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)


def _load(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("Experiment and dataset reports must be JSON objects")
    return value


def run(args) -> int:
    if args.command == "corrupt":
        result = generate_experiment(args.stems, args.stem_id, args.config, args.output_dir)
        print(
            json.dumps(
                {
                    "experiment": str((args.output_dir / "experiment.json").resolve()),
                    "summary": result["summary"],
                },
                indent=2,
                allow_nan=False,
            )
        )
        return 1 if result["summary"]["failed_count"] else 0
    from waves_sed.corruption_comparison import compare_corruptions, write_comparison

    inputs = {"experiment": args.experiment, "report": args.report}
    input_hashes = {key: sha256(path) for key, path in inputs.items()}
    experiment, dataset = _load(args.experiment), _load(args.report)
    protected = [args.experiment, args.report]
    protected.extend(recorded_paths(experiment, base=args.experiment.resolve().parent))
    protected.extend(recorded_paths(dataset, base=args.report.resolve().parent))
    result = compare_corruptions(experiment, dataset)
    result.setdefault("provenance", {})["inputs"] = {
        key: {"path": str(path.resolve()), "sha256": input_hashes[key]}
        for key, path in inputs.items()
    }
    if any(sha256(path) != input_hashes[key] for key, path in inputs.items()):
        raise ValueError("Comparison input changed while reading; no outputs were written")
    write_comparison(result, args.output_dir, protected_inputs=protected)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), "summary": result["summary"]},
            indent=2,
            allow_nan=False,
        )
    )
    return 0
