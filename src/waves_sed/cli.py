import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from waves_sed import __version__
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    PREPROCESSING,
    UPSTREAM_REVISION,
    download_checkpoint,
    sha256,
)


def summary(prediction: FramePrediction, top_k: int = 10) -> dict:
    peaks = prediction.probabilities.max(axis=0)
    means = prediction.probabilities.mean(axis=0)
    indices = np.argsort(-peaks, kind="stable")[:top_k]
    return {
        "shape": list(prediction.probabilities.shape),
        "start_seconds": float(prediction.frame_start_seconds[0]),
        "end_seconds": float(prediction.frame_end_seconds[-1]),
        "probability_min": float(prediction.probabilities.min()),
        "probability_max": float(prediction.probabilities.max()),
        "top_classes_by_peak": [
            {
                "class_id": prediction.class_ids[i],
                "class_name": prediction.class_names[i],
                "peak_probability": float(peaks[i]),
                "mean_probability": float(means[i]),
            }
            for i in indices
        ],
        "metadata": prediction.metadata,
    }


def export_csv(prediction: FramePrediction, path: Path) -> None:
    source = prediction.metadata.get("audio_path")
    if isinstance(source, str) and source:
        source_path = Path(source)
        if path.resolve() == source_path.resolve() or (
            path.exists() and source_path.exists() and path.samefile(source_path)
        ):
            raise ValueError("CSV must not overwrite source audio recorded in prediction metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame_start_seconds", "frame_end_seconds", "class_id", "class_name", "probability"]
        )
        for start, end, row in zip(
            prediction.frame_start_seconds, prediction.frame_end_seconds, prediction.probabilities
        ):
            for class_id, name, value in zip(prediction.class_ids, prediction.class_names, row):
                writer.writerow([f"{start:.8f}", f"{end:.8f}", class_id, name, float(value)])


def reusable(prediction: FramePrediction, audio: Path) -> bool:
    from waves_sed.labels import load_labels

    ids, names = load_labels()
    expected = {
        "audio_sha256": sha256(audio),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "preprocessing": PREPROCESSING,
        "upstream_revision": UPSTREAM_REVISION,
        "package_version": __version__,
        "backend": "atst_f_strong",
    }
    return (
        all(prediction.metadata.get(key) == value for key, value in expected.items())
        and prediction.class_ids == ids
        and prediction.class_names == names
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen ATST-F Strong raw frame inference")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch the pinned official Strong checkpoint")
    download.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/checkpoints") / CHECKPOINT_NAME
    )
    infer = commands.add_parser("infer", help="Infer one WAV, or reuse a matching raw cache")
    infer.add_argument("audio", type=Path)
    infer.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/checkpoints") / CHECKPOINT_NAME
    )
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    infer.add_argument("--threads", type=int, default=4)
    infer.add_argument(
        "--overwrite", action="store_true", help="Replace an existing cache with fresh inference"
    )
    infer.add_argument("--csv", type=Path, help="Optional long-format raw frame probability CSV")
    inspect = commands.add_parser("inspect", help="Read raw cache without torch or checkpoint")
    inspect.add_argument("prediction", type=Path)
    inspect.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        if args.command == "download":
            path = download_checkpoint(args.checkpoint)
            print(json.dumps({"checkpoint": str(path.resolve()), "sha256": sha256(path)}))
            return 0
        if args.csv:
            protected = (
                {args.prediction.resolve()}
                if args.command == "inspect"
                else {args.audio.resolve(), args.output.resolve(), args.checkpoint.resolve()}
            )
            if args.csv.resolve() in protected:
                raise ValueError("CSV must not overwrite audio, checkpoint or prediction cache")
        if args.command == "inspect":
            prediction = FramePrediction.load(args.prediction)
            result = summary(prediction)
        else:
            if args.threads < 1:
                raise ValueError("--threads must be positive")
            if args.output.resolve() in {args.audio.resolve(), args.checkpoint.resolve()}:
                raise ValueError("Output must not overwrite the source audio or checkpoint")
            if args.output.exists() and not args.overwrite:
                prediction = FramePrediction.load(args.output)
                if not reusable(prediction, args.audio):
                    raise ValueError(
                        "Existing cache does not match this input/model; "
                        "choose a different --output or use --overwrite"
                    )
                cache_hit = True
            else:
                import torch

                from waves_sed.backends.atst import ATSTBackend

                torch.set_num_threads(args.threads)
                print("Loading frozen ATST-F Strong and running inference...", file=sys.stderr)
                prediction = ATSTBackend(args.checkpoint, args.device).predict(args.audio)
                prediction.save(args.output)
                cache_hit = False
            result = summary(prediction)
            result.update(output=str(args.output.resolve()), cache_hit=cache_hit)
        if args.csv:
            export_csv(prediction, args.csv)
            result["csv"] = str(args.csv.resolve())
        result["elapsed_seconds"] = round(perf_counter() - started, 3)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f"waves-sed: {error}", file=sys.stderr)
        return 1
