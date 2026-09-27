"""Cached evaluation is reproducible, offline, and cannot overwrite source inputs."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from waves_sed.cli import main
from waves_sed.metadata import StemMetadata
from waves_sed.phase4 import load_prediction_index
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def cached_stem(tmp_path):
    audio = tmp_path / "stem.wav"
    audio.write_bytes(b"synthetic audio identity only; never decoded during cached evaluation")
    stem = StemMetadata(
        stem_id="clip::candidate",
        candidate_id="candidate",
        clip_key="clip",
        audio_path=audio,
        source_description="dog barking",
        role="onset",
        expected_intervals=((0.1, 0.3),),
        planned_role="onset",
        planned_label="dog barking",
        planned_description="dog barking",
        description_origin="final_label",
        provenance={
            "reference_origin": "waves_pass1_planned",
            "reference_status": "planned",
            "adapter": "waves_materialized",
            "adapter_schema_version": 1,
            "support_policy": "selected_parent_only",
            "inputs": {},
            "source_media_paths": [],
            "raw_final_stem": {"sha256": sha256(audio)},
        },
    )
    manifest = write_json(
        tmp_path / "stems.json",
        {
            "schema_version": 1,
            "input_format": "waves_materialized",
            "temporal_reference": "waves_pass1_planned",
            "summary": {"stem_count": 1, "clip_count": 1},
            "stems": [stem.to_dict()],
        },
    )
    cache = tmp_path / "raw.npz"
    prediction = FramePrediction(
        probabilities=np.array([[0.1], [0.9], [0.8], [0.1]], dtype=np.float32),
        frame_start_seconds=np.array([0.0, 0.1, 0.2, 0.3]),
        frame_end_seconds=np.array([0.1, 0.2, 0.3, 0.4]),
        class_ids=("/m/05tny_",),
        class_names=("Bark",),
        metadata={
            "audio_path": str(audio),
            "audio_sha256": sha256(audio),
            "backend": "synthetic",
            "duration_seconds": 0.4,
            "purpose": "synthetic CLI evaluation fixture",
        },
    )
    prediction.save(cache)
    index = write_json(
        tmp_path / "predictions.json",
        {
            "schema_version": 1,
            "predictions": {stem.stem_id: "raw.npz"},
        },
    )
    mapping = write_json(
        tmp_path / "mappings.json",
        {
            "schema_version": 1,
            "mappings": {
                "bark": {
                    "descriptions": ["dog barking"],
                    "allowed_classes": ["/m/05tny_"],
                }
            },
        },
    )
    config = write_json(
        tmp_path / "config.json",
        {
            "schema_version": 1,
            "event": {"threshold": 0.5, "median_window_frames": 1, "min_duration_seconds": 0},
            "onset_tolerance_seconds": 0.05,
            "allow_ambiguous_reference": False,
            "outside_family_threshold": 0.5,
            "outside_family_top_k": 5,
        },
    )
    return SimpleNamespace(
        audio=audio,
        stem=stem,
        manifest=manifest,
        cache=cache,
        prediction=prediction,
        index=index,
        mapping=mapping,
        config=config,
        output=tmp_path / "reports",
    )


def evaluate_args(files, *extra):
    return [
        "evaluate",
        "--stems",
        str(files.manifest),
        "--predictions",
        str(files.index),
        "--mappings",
        str(files.mapping),
        "--config",
        str(files.config),
        "--output-dir",
        str(files.output),
        *extra,
    ]


def dataset(files):
    return json.loads((files.output / "dataset.json").read_text(encoding="utf-8"))


def test_evaluate_outputs_expected_metrics_and_preserves_raw_cache(cached_stem, capsys):
    original = cached_stem.cache.read_bytes()
    assert main(evaluate_args(cached_stem)) == 0
    assert json.loads(capsys.readouterr().out)["evaluation_statuses"] == {"evaluated": 1}
    report = dataset(cached_stem)
    row = report["stems"][0]
    assert row["evaluation_status"] == "evaluated"
    assert row["metrics"]["matched_event_count"] == 1
    assert row["metrics"]["missing_event_count"] == 0
    assert row["detected_intervals"] == [[0.1, 0.3]]
    assert row["decision"] is None
    assert report["provenance"]["inference_performed"] is False
    assert report["provenance"]["inputs"]["config"]["sha256"] == sha256(cached_stem.config)
    assert cached_stem.cache.read_bytes() == original


def test_threshold_change_reuses_cache_but_changes_detection(cached_stem):
    original = cached_stem.cache.read_bytes()
    assert main(evaluate_args(cached_stem)) == 0
    assert dataset(cached_stem)["stems"][0]["metrics"]["detected_event_count"] == 1
    config = json.loads(cached_stem.config.read_text())
    config["event"]["threshold"] = 0.95
    write_json(cached_stem.config, config)
    assert main(evaluate_args(cached_stem)) == 0
    row = dataset(cached_stem)["stems"][0]
    assert row["metrics"]["detected_event_count"] == 0
    assert row["metrics"]["missing_event_count"] == 1
    assert cached_stem.cache.read_bytes() == original


@pytest.mark.parametrize(
    "problem", ["absent_entry", "absent_file", "corrupt_file", "mapping", "reference", "identity"]
)
def test_unavailable_inputs_produce_explicit_records_without_zero_metrics(cached_stem, problem):
    if problem == "absent_entry":
        write_json(cached_stem.index, {"schema_version": 1, "predictions": {}})
    elif problem == "absent_file":
        cached_stem.cache.unlink()
    elif problem == "corrupt_file":
        cached_stem.cache.write_bytes(b"not an NPZ")
    elif problem == "mapping":
        write_json(cached_stem.mapping, {"schema_version": 1, "mappings": {}})
    elif problem == "reference":
        manifest = json.loads(cached_stem.manifest.read_text())
        manifest["stems"][0]["expected_intervals"] = None
        manifest["stems"][0]["provenance"]["reference_status"] = "missing"
        manifest["stems"][0]["provenance"]["reference_origin"] = None
        write_json(cached_stem.manifest, manifest)
    else:
        cached_stem.prediction.metadata["audio_sha256"] = "0" * 64
        cached_stem.prediction.save(cached_stem.cache)
    assert main(evaluate_args(cached_stem)) == 0
    report = dataset(cached_stem)
    row = report["stems"][0]
    assert row["evaluation_status"] == "unavailable"
    assert row["metrics"] is None
    assert row["reason_codes"]
    assert report["summary"]["roles"]["onset"]["onset_micro"]["missing_event_count"] is None


@pytest.mark.parametrize(
    "content",
    [
        '{"schema_version":1,"predictions":{"unknown":"raw.npz"}}',
        '{"schema_version":1,"predictions":{"clip::candidate":"a","clip::candidate":"b"}}',
        '{"schema_version":1,"predictions":{"clip::candidate":NaN}}',
        '{"schema_version":1,"predictions":{"clip::candidate":1}}',
        '{"schema_version":1,"predictions":{"clip::candidate":" "}}',
        '{"schema_version":true,"predictions":{}}',
        '{"schema_version":1,"predictions":{},"extra":1}',
    ],
)
def test_prediction_index_rejects_ambiguous_or_malformed_inputs(cached_stem, content):
    cached_stem.index.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_prediction_index(cached_stem.index, {cached_stem.stem.stem_id})


def test_prediction_index_resolves_paths_relative_to_index(cached_stem):
    assert load_prediction_index(cached_stem.index, {cached_stem.stem.stem_id}) == {
        cached_stem.stem.stem_id: cached_stem.cache.resolve(),
    }


@pytest.mark.parametrize(
    "source_name", ["manifest", "index", "mapping", "config", "audio", "cache"]
)
def test_all_inputs_protected_from_report_hardlink_aliases(cached_stem, capsys, source_name):
    source = getattr(cached_stem, source_name)
    cached_stem.output.mkdir()
    target = cached_stem.output / "dataset.json"
    try:
        target.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = source.read_bytes()
    assert main(evaluate_args(cached_stem)) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert source.read_bytes() == target.read_bytes() == original
    assert not (cached_stem.output / "stems.csv").exists()
    assert not (cached_stem.output / "stems").exists()


@pytest.mark.parametrize("kind", ["source_media", "source_metadata", "prediction_audio"])
def test_provenance_inputs_are_also_protected_before_any_writes(
    cached_stem, tmp_path, capsys, kind
):
    source = tmp_path / "original.dat"
    source.write_bytes(b"original source")
    if kind == "prediction_audio":
        cached_stem.prediction.metadata["audio_path"] = str(source)
        cached_stem.prediction.save(cached_stem.cache)
    else:
        manifest = json.loads(cached_stem.manifest.read_text())
        provenance = manifest["stems"][0]["provenance"]
        if kind == "source_media":
            provenance["source_media_paths"] = [str(source)]
        else:
            provenance["inputs"] = {"original": {"path": str(source), "sha256": sha256(source)}}
        write_json(cached_stem.manifest, manifest)
    cached_stem.output.mkdir()
    alias = cached_stem.output / "clips.csv"
    try:
        alias.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    assert main(evaluate_args(cached_stem)) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert source.read_bytes() == alias.read_bytes() == b"original source"
    assert not (cached_stem.output / "stems").exists()


def test_cached_evaluation_runs_in_fresh_process_without_inference_imports(cached_stem):
    script = """
import builtins
import json
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchaudio', 'librosa', 'soundfile'}:
        raise AssertionError('Offline evaluation imported ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from waves_sed.cli import main
raise SystemExit(main(json.loads(sys.argv[1])))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(evaluate_args(cached_stem))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert dataset(cached_stem)["summary"]["evaluation_statuses"] == {"evaluated": 1}


def test_invalid_cache_still_protects_parseable_source_metadata(cached_stem, tmp_path, capsys):
    source = tmp_path / "otherwise-unreferenced-source.wav"
    source.write_bytes(b"must survive a failed prediction validation")
    with np.load(cached_stem.cache, allow_pickle=False) as data:
        payload = dict(data)
    payload["probabilities"] = np.full((4, 1), 2.0)
    metadata = json.loads(payload["metadata_json"].item())
    metadata["audio_path"] = str(source)
    payload["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(cached_stem.cache, **payload)
    cached_stem.output.mkdir()
    alias = cached_stem.output / "dataset.json"
    try:
        alias.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    assert main(evaluate_args(cached_stem)) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert (
        source.read_bytes() == alias.read_bytes() == b"must survive a failed prediction validation"
    )
    assert not (cached_stem.output / "stems").exists()
