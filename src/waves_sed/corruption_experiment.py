"""Create immutable, source-preserving controlled waveform experiments."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from waves_sed.corruption import apply_corruption, load_corruption_specs
from waves_sed.metadata import load_stems
from waves_sed.phase4 import _provenance_paths
from waves_sed.provenance import sha256
from waves_sed.reporting import _atomic_text, preflight_outputs


def recorded_paths(value, *, base: Path, _file_collection: bool = False) -> list[Path]:
    """Collect recorded inputs, including nested source stems and unused reports."""
    paths = []
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                (
                    key
                    in {"path", "file", "audio_path", "source_attempt_file", "mixed_audio", "video"}
                    or key.endswith("_path")
                    or _file_collection
                )
                and isinstance(item, str)
                and item.strip()
            ):
                paths.extend([Path(item), base / item])
            elif key == "source_media_paths" and isinstance(item, list):
                for path in item:
                    if isinstance(path, str) and path.strip():
                        paths.extend([Path(path), base / path])
            paths.extend(
                recorded_paths(item, base=base, _file_collection=_file_collection or key == "files")
            )
    elif isinstance(value, list):
        for item in value:
            if _file_collection and isinstance(item, str) and item.strip():
                paths.extend([Path(item), base / item])
            paths.extend(recorded_paths(item, base=base, _file_collection=_file_collection))
    return paths


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int, soundfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".wav", delete=False) as stream:
            temporary = Path(stream.name)
        soundfile.write(temporary, samples, sample_rate, format="WAV", subtype="FLOAT")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate_experiment(
    stems_path: Path, stem_id: str, config_path: Path, output_dir: Path
) -> dict:
    """Decode once, derive each variant from the original, and publish manifests last.

    The original control keeps its exact path and bytes. Generated FLOAT WAVs
    preserve native rate, channels and length, with no amplitude normalization.
    Missing/corrupt source or config aborts; individual edit failures are recorded.
    """
    stems_path, config_path, output_dir = (
        Path(stems_path).resolve(),
        Path(config_path).resolve(),
        Path(output_dir).resolve(),
    )
    input_hashes = {stems_path: sha256(stems_path), config_path: sha256(config_path)}
    stems = load_stems(stems_path)
    specs = load_corruption_specs(config_path)
    source = next((stem for stem in stems if stem.stem_id == stem_id), None)
    if source is None:
        raise ValueError(f"Unknown stem ID: {stem_id}")
    if source.audio_path is None or not source.audio_path.is_file():
        raise ValueError(f"Source audio is missing for {stem_id}")
    source_hash = sha256(source.audio_path)
    expected_hash = source.provenance.get("raw_final_stem", {}).get("sha256")
    if expected_hash is not None and (
        not isinstance(expected_hash, str) or expected_hash.lower() != source_hash
    ):
        raise ValueError("Source audio SHA256 does not match normalized stem provenance")

    control_id = source.stem_id + "::control"
    variant_ids = {spec.id: source.stem_id + "::corruption::" + spec.id for spec in specs}
    audio_paths = {
        key: output_dir / "audio" / (hashlib.sha256(value.encode("utf-8")).hexdigest() + ".wav")
        for key, value in variant_ids.items()
    }
    manifest_path, experiment_path = output_dir / "stems.json", output_dir / "experiment.json"
    outputs = [manifest_path, experiment_path, *audio_paths.values()]
    protected = [stems_path, config_path, *_provenance_paths(stems)]
    for stem in stems:
        protected.extend(recorded_paths(stem.to_dict(), base=stems_path.parent))
    preflight_outputs(outputs, protected)
    for path in outputs:
        if path.exists():
            raise ValueError(
                f"Experiment output already exists; choose a new output directory: {path}"
            )

    try:
        import soundfile
    except ImportError as exc:
        raise ImportError(
            'Corruption generation requires soundfile; install "waves-stem-sed[corruption]" or ".[corruption]"'
        ) from exc
    samples, sample_rate = soundfile.read(source.audio_path, dtype="float32", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Source audio must contain finite, nonempty samples")
    if sha256(source.audio_path) != source_hash or any(
        sha256(path) != digest for path, digest in input_hashes.items()
    ):
        raise ValueError("Experiment input changed during loading")
    source_audio = {"path": str(source.audio_path), "sha256": source_hash}
    semantic_config = [spec.to_dict() for spec in specs]
    experiment_id = hashlib.sha256(
        json.dumps(
            {
                "source_stem": source.to_dict(),
                "source_audio": source_audio,
                "variants": semantic_config,
                "implementation_version": 1,
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    def derived(*, spec=None, audio_path=None, audio_hash=None, application=None):
        provenance = copy.deepcopy(source.provenance)
        provenance["source_media_paths"] = list(
            dict.fromkeys([*provenance.get("source_media_paths", []), str(source.audio_path)])
        )
        provenance["corruption"] = {
            "experiment_id": experiment_id,
            "kind": "control" if spec is None else "variant",
            "variant_id": "control" if spec is None else spec.id,
            "parent_stem_id": source.stem_id,
            "control_stem_id": control_id,
            "operation": "control" if spec is None else spec.operation,
            "source_audio": source_audio,
            "spec": None if spec is None else spec.to_dict(),
            "application": application,
            "source_stem": source.to_dict(),
        }
        provenance.setdefault("inputs", {}).update(
            {
                "corruption_stems": {"path": str(stems_path), "sha256": input_hashes[stems_path]},
                "corruption_config": {
                    "path": str(config_path),
                    "sha256": input_hashes[config_path],
                },
            }
        )
        if spec is not None:
            provenance.setdefault("raw_final_stem", {}).update(
                {
                    "file": str(audio_path),
                    "sha256": audio_hash,
                }
            )
        suffix = "::control" if spec is None else "::corruption::" + spec.id
        return replace(
            source,
            stem_id=source.stem_id + suffix,
            candidate_id=source.candidate_id + suffix,
            audio_path=source.audio_path if spec is None else audio_path,
            provenance=provenance,
        )

    control = derived()
    generated_stems = [control]
    variants = []
    for spec in specs:
        record = {
            "id": spec.id,
            "stem_id": variant_ids[spec.id],
            "operation": spec.operation,
            "status": "failed",
            "error": None,
            "audio_path": None,
            "audio_sha256": None,
            "spec": spec.to_dict(),
            "application": None,
            "stem": None,
        }
        try:
            modified, application = apply_corruption(samples, sample_rate, spec)
            application.update(
                {
                    "decoded_sample_dtype": "float32",
                    "output_wav_subtype": "FLOAT",
                    "audio_changed_basis": "decoded_float32_samples",
                }
            )
            _write_wav(audio_paths[spec.id], modified, sample_rate, soundfile)
            digest = sha256(audio_paths[spec.id])
            stem = derived(
                spec=spec,
                audio_path=audio_paths[spec.id],
                audio_hash=digest,
                application=application,
            )
            generated_stems.append(stem)
            record.update(
                status="generated",
                audio_path=str(audio_paths[spec.id]),
                audio_sha256=digest,
                application=application,
                stem=stem.to_dict(),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        variants.append(record)
    if sha256(source.audio_path) != source_hash or any(
        sha256(path) != digest for path, digest in input_hashes.items()
    ):
        raise ValueError("Experiment input changed during generation; manifests were not published")
    manifest = _json(
        {
            "schema_version": 1,
            "input_format": "controlled_corruption",
            "stems": [stem.to_dict() for stem in generated_stems],
        }
    )
    result = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "source_stem": source.to_dict(),
        "source_audio": {
            **source_audio,
            "sample_rate": sample_rate,
            "frames": len(samples),
            "channels": samples.shape[1],
        },
        "config": {
            "path": str(config_path),
            "sha256": input_hashes[config_path],
            "variants": semantic_config,
        },
        "control": {
            "stem_id": control_id,
            "audio_path": str(source.audio_path),
            "audio_sha256": source_hash,
            "stem": control.to_dict(),
        },
        "variants": variants,
        "stems_file": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        },
        "reference_policy": "unchanged_from_source",
        "summary": {
            "variant_count": len(variants),
            "generated_count": len(generated_stems) - 1,
            "failed_count": len(variants) - len(generated_stems) + 1,
            "changed_count": sum(
                bool((row["application"] or {}).get("audio_changed")) for row in variants
            ),
        },
        "provenance": {
            "implementation_version": 1,
            "inference_performed": False,
            "control_policy": "original_audio_bytes_and_path",
            "reference_policy": "unchanged_from_source",
            "variant_policy": "independent_from_original_fixed_clip_length",
            "output_wav_subtype": "FLOAT",
            "normalization_applied": False,
        },
    }
    _atomic_text(manifest_path, manifest)
    _atomic_text(experiment_path, _json(result))
    return result
