"""Sequential, resumable raw inference with one lazily initialized frozen backend."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

from waves_sed import __version__
from waves_sed.metadata import StemMetadata
from waves_sed.phase4 import _invalid_cache_audio_path, _provenance_paths
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import (
    CHECKPOINT_SHA256,
    PREPROCESSING,
    UPSTREAM_REVISION,
    sha256,
)
from waves_sed.reporting import _atomic_text, preflight_outputs


def _recorded_audio_paths(cache: Path) -> list[Path]:
    audio = _invalid_cache_audio_path(cache)
    return [Path(audio), cache.parent / audio] if audio else []


def _default_backend(checkpoint: Path, device: str, threads: int):
    import torch

    from waves_sed.backends.atst import ATSTBackend

    torch.set_num_threads(threads)
    backend = ATSTBackend(checkpoint, device)
    # Initialize here so a broken shared model/checkpoint is attempted only once.
    # ATSTBackend.predict then reuses this model for every stem.
    backend._load_model()
    return backend


def run_batch(
    stems: list[StemMetadata],
    output_dir: Path,
    *,
    checkpoint: Path,
    device: str = "cpu",
    threads: int = 4,
    overwrite: bool = False,
    protected_inputs: list[Path] | None = None,
    input_provenance: dict | None = None,
    backend_factory=None,
) -> dict:
    """Write a current-run prediction index and per-stem processing status.

    Existing caches require the same identity as the single-file ``infer`` CLI.
    ``overwrite`` forces fresh inference, including for matching caches. Errors
    local to a stem are recorded and processing continues. Structural output
    collisions are rejected before any directory creation or model loading.
    An injected factory receives ``(checkpoint, device)`` and needs a ``predict``
    method; it avoids all inference-library imports in synthetic tests.
    """
    from waves_sed.cli import reusable

    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a boolean")
    if input_provenance is not None and not isinstance(input_provenance, dict):
        raise ValueError("input_provenance must be a JSON object")
    inputs = json.loads(json.dumps(input_provenance or {}, allow_nan=False))
    stem_ids = [stem.stem_id for stem in stems]
    if any(not isinstance(key, str) or not key.strip() for key in stem_ids):
        raise ValueError("Batch stem IDs must be nonempty strings")
    if len(set(stem_ids)) != len(stem_ids):
        raise ValueError("Batch stem IDs must be unique")
    output_dir = Path(output_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    relative_paths = {
        key: f"predictions/{hashlib.sha256(key.encode('utf-8')).hexdigest()}.npz"
        for key in stem_ids
    }
    cache_paths = {key: output_dir / path for key, path in relative_paths.items()}
    index_path = output_dir / "prediction-index.json"
    status_path = output_dir / "batch-status.json"
    outputs = [*cache_paths.values(), index_path, status_path]
    protected = [
        checkpoint,
        *(Path(path) for path in (protected_inputs or [])),
        *_provenance_paths(stems),
    ]
    # Include obsolete caches: their recorded audio remains a protected source.
    existing = set(cache_paths.values())
    if (output_dir / "predictions").is_dir():
        existing.update((output_dir / "predictions").rglob("*.npz"))
    for path in existing:
        if path.is_file():
            protected.extend(_recorded_audio_paths(path))
    preflight_outputs(outputs, protected)

    started = perf_counter()
    records = []
    successful = {}
    backend = None
    initialization_error = None
    initializations = 0
    inference_performed = False

    for stem in stems:
        cache_path = cache_paths[stem.stem_id]
        audio = Path(stem.audio_path).resolve() if stem.audio_path is not None else None
        row = {
            "stem_id": stem.stem_id,
            "clip_key": stem.clip_key,
            "candidate_id": stem.candidate_id,
            "audio_path": str(audio) if audio is not None else None,
            "audio_sha256": None,
            "cache_path": str(cache_path),
            "cache_sha256": None,
            "status": None,
            "error": None,
            "reason_code": None,
            "prediction_provenance": None,
        }
        records.append(row)
        if audio is None or not audio.is_file():
            row.update(status="missing_audio", reason_code="missing_audio")
            continue
        try:
            original_digest = sha256(audio)
            row["audio_sha256"] = original_digest
        except OSError as error:
            row.update(status="audio_unreadable", reason_code="audio_unreadable", error=str(error))
            continue
        declared = stem.provenance.get("raw_final_stem", {}).get("sha256")
        if declared is not None and declared != original_digest:
            row.update(
                status="source_hash_mismatch",
                reason_code="manifest_audio_hash_mismatch",
                error="Source audio SHA-256 differs from raw_final_stem.sha256",
            )
            continue

        prediction = None
        if cache_path.exists() and not overwrite:
            try:
                prediction = FramePrediction.load(cache_path)
                if not reusable(prediction, audio):
                    raise ValueError("Existing cache does not match this input/model")
                if sha256(audio) != original_digest:
                    raise ValueError("Source audio changed during cache validation")
                row["status"] = "cached"
            except (OSError, ValueError) as error:
                row.update(status="cache_conflict", reason_code="cache_conflict", error=str(error))
                continue
        else:
            if backend is None and initialization_error is None:
                initializations += 1
                try:
                    backend = (
                        backend_factory(checkpoint, device)
                        if backend_factory is not None
                        else _default_backend(checkpoint, device, threads)
                    )
                    if not callable(getattr(backend, "predict", None)):
                        raise TypeError("Backend factory must return an object with predict")
                except Exception as error:
                    initialization_error = f"{type(error).__name__}: {error}"
            if initialization_error is not None:
                row.update(
                    status="inference_failed",
                    reason_code="backend_initialization_failed",
                    error=initialization_error,
                )
                continue
            try:
                inference_performed = True
                prediction = backend.predict(audio)
                if not isinstance(prediction, FramePrediction):
                    raise ValueError("Backend did not return a FramePrediction")
                prediction.validate()
                if sha256(audio) != original_digest:
                    raise ValueError("Source audio changed during inference")
                if not reusable(prediction, audio):
                    raise ValueError("Generated prediction does not match this input/model")
                recorded_audio = prediction.metadata.get("audio_path")
                if isinstance(recorded_audio, str) and recorded_audio:
                    protected.extend([Path(recorded_audio), cache_path.parent / recorded_audio])
                preflight_outputs(outputs, protected)
                prediction.save(cache_path)
                row["status"] = "inferred"
            except Exception as error:
                row.update(
                    status="inference_failed",
                    reason_code="prediction_failed",
                    error=f"{type(error).__name__}: {error}",
                )
                continue
        try:
            row["cache_sha256"] = sha256(cache_path)
        except OSError as error:
            row.update(status="cache_unreadable", reason_code="cache_unreadable", error=str(error))
            continue
        row["prediction_provenance"] = prediction.metadata
        successful[stem.stem_id] = relative_paths[stem.stem_id]

    summary = {
        "stem_count": len(stems),
        "cached_count": sum(row["status"] == "cached" for row in records),
        "inferred_count": sum(row["status"] == "inferred" for row in records),
        "failed_count": sum(row["status"] not in {"cached", "inferred"} for row in records),
        "backend_initializations": initializations,
    }
    status = {
        "schema_version": 1,
        "summary": summary,
        "stems": records,
        "files": {"prediction_index": "prediction-index.json", "batch_status": "batch-status.json"},
        "provenance": {
            "inputs": inputs,
            "backend": "atst_f_strong",
            "package_version": __version__,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_hash_scope": "pinned expected hash; checkpoint bytes are read only for fresh inference",
            "upstream_revision": UPSTREAM_REVISION,
            "preprocessing": PREPROCESSING,
            "requested_device": device,
            "threads": threads,
            "overwrite": overwrite,
            "inference_performed": inference_performed,
            "execution": "sequential; at most one backend initialization per run",
            "cache_reuse": "exact input SHA-256, pinned model/preprocessing/package and 447-class order",
            "cache_reuse_device": "device-independent; each cache retains its original device provenance",
            "index_policy": "current successful stems only; obsolete caches are retained and unindexed",
        },
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    preflight_outputs(outputs, protected)
    _atomic_text(
        status_path, json.dumps(status, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    _atomic_text(
        index_path,
        json.dumps(
            {"schema_version": 1, "predictions": successful},
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
    )
    return status
