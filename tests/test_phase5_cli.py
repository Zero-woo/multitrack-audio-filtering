"""Batch cache reuse and plots remain separate from model inference."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import wave
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from waves_sed import __version__
from waves_sed.cli import main
from waves_sed.labels import load_labels
from waves_sed.metadata import StemMetadata
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import CHECKPOINT_SHA256, PREPROCESSING, UPSTREAM_REVISION, sha256

requires_plotting = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("matplotlib", "soundfile")),
    reason="Requires visualization optional dependencies",
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stem_key(stem_id):
    return hashlib.sha256(stem_id.encode("utf-8")).hexdigest()


@pytest.fixture
def files(tmp_path):
    audio = tmp_path / "stem.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        samples = np.zeros(6400, dtype="<i2")
        samples[1600:4800] = 10000
        stream.writeframes(samples.tobytes())
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
            "purpose": "synthetic visualization fixture",
        },
    )
    cache = tmp_path / "raw.npz"
    prediction.save(cache)
    index = write_json(
        tmp_path / "predictions.json",
        {"schema_version": 1, "predictions": {stem.stem_id: "raw.npz"}},
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
        prediction=prediction,
        cache=cache,
        index=index,
        mapping=mapping,
        config=config,
        output=tmp_path / "visuals",
        batch_output=tmp_path / "batch",
    )


def visual_args(files, *extra):
    return [
        "visualize",
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


def batch_args(files, *extra):
    return [
        "batch-infer",
        "--stems",
        str(files.manifest),
        "--output-dir",
        str(files.batch_output),
        *extra,
    ]


def render_manifest(files):
    return load_json(files.output / "visualization-index.json")


def pinned_cache(files):
    ids, names = load_labels()
    prediction = replace(
        files.prediction,
        probabilities=np.tile(files.prediction.probabilities, (1, len(ids))),
        class_ids=ids,
        class_names=names,
        metadata={
            **files.prediction.metadata,
            "backend": "atst_f_strong",
            "package_version": __version__,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "preprocessing": PREPROCESSING,
            "upstream_revision": UPSTREAM_REVISION,
        },
    )
    path = files.batch_output / "predictions" / f"{stem_key(files.stem.stem_id)}.npz"
    prediction.save(path)
    return path


def fresh_process(args, blocked, *, missing_dependency=False):
    exception = "ImportError" if missing_dependency else "AssertionError"
    script = f"""
import builtins
import json
import sys
original_import = builtins.__import__
blocked = set(json.loads(sys.argv[2]))
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise {exception}('Forbidden optional import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from waves_sed.cli import main
raise SystemExit(main(json.loads(sys.argv[1])))
"""
    return subprocess.run(
        [sys.executable, "-c", script, json.dumps(args), json.dumps(blocked)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


@pytest.mark.parametrize("format_name", ["png", "svg"])
@requires_plotting
def test_visualize_outputs_portable_plots_and_current_evaluation(files, format_name):
    original = files.cache.read_bytes()
    assert main(visual_args(files, "--format", format_name)) == 0
    manifest = render_manifest(files)
    assert manifest["summary"]["stem_count"] == 1
    assert manifest["summary"]["rendered_count"] == 1
    assert manifest["summary"]["failed_count"] == 0
    row = manifest["stems"][0]
    assert row["stem_id"] == files.stem.stem_id
    assert row["status"] == "rendered"
    expected = f"plots/{stem_key(files.stem.stem_id)}.{format_name}"
    assert row["path"] == expected
    plot = files.output / row["path"]
    assert plot.is_file() and plot.stat().st_size > 1000
    if format_name == "png":
        assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    else:
        assert "<svg" in plot.read_text(encoding="utf-8")
    report = row["evaluation"]
    assert report["evaluation_status"] == "evaluated"
    assert report["metrics"]["matched_event_count"] == 1
    assert report["decision"] is None
    assert expected in (files.output / "index.html").read_text(encoding="utf-8")
    assert files.cache.read_bytes() == original


@requires_plotting
def test_visualization_recomputes_after_config_changes_without_mutating_cache(files):
    original = files.cache.read_bytes()
    assert main(visual_args(files)) == 0
    assert render_manifest(files)["stems"][0]["evaluation"]["detected_intervals"] == [[0.1, 0.3]]
    config = load_json(files.config)
    config["event"]["threshold"] = 0.95
    write_json(files.config, config)
    assert main(visual_args(files)) == 0
    report = render_manifest(files)["stems"][0]["evaluation"]
    assert report["detected_intervals"] == []
    assert report["metrics"]["missing_event_count"] == 1
    assert files.cache.read_bytes() == original


@pytest.mark.parametrize(
    "problem", ["missing_entry", "missing_cache", "corrupt_cache", "mapping", "ambiguous"]
)
@requires_plotting
def test_unavailable_evaluation_still_renders_an_explicit_record(files, problem):
    if problem == "missing_entry":
        write_json(files.index, {"schema_version": 1, "predictions": {}})
    elif problem == "missing_cache":
        files.cache.unlink()
    elif problem == "corrupt_cache":
        files.cache.write_bytes(b"not an NPZ")
    elif problem == "mapping":
        write_json(files.mapping, {"schema_version": 1, "mappings": {}})
    else:
        manifest = load_json(files.manifest)
        manifest["stems"][0]["provenance"]["reference_status"] = "ambiguous"
        write_json(files.manifest, manifest)
    assert main(visual_args(files)) == 0
    row = render_manifest(files)["stems"][0]
    assert row["status"] == "rendered"
    assert (files.output / row["path"]).is_file()
    assert row["evaluation"]["evaluation_status"] == "unavailable"
    assert row["evaluation"]["metrics"] is None
    assert row["evaluation"]["reason_codes"]


@requires_plotting
def test_missing_waveform_does_not_discard_verified_cache_evaluation(files):
    files.audio.unlink()
    assert main(visual_args(files)) == 0
    row = render_manifest(files)["stems"][0]
    assert row["status"] == "rendered"
    assert row["evaluation"]["evaluation_status"] == "evaluated"
    assert row["evaluation"]["cache_validation"]["source_file_verified"] is False


@requires_plotting
def test_mixed_stems_and_exact_selection_do_not_invent_missing_cache_scores(files):
    manifest = load_json(files.manifest)
    second = replace(files.stem, stem_id="clip::missing", candidate_id="missing")
    manifest["stems"].append(second.to_dict())
    manifest["summary"]["stem_count"] = 2
    write_json(files.manifest, manifest)
    assert main(visual_args(files)) == 0
    rows = {row["stem_id"]: row for row in render_manifest(files)["stems"]}
    assert rows[files.stem.stem_id]["evaluation"]["evaluation_status"] == "evaluated"
    assert rows[second.stem_id]["evaluation"]["evaluation_status"] == "unavailable"
    assert main(visual_args(files, "--stem-id", second.stem_id)) == 0
    assert [row["stem_id"] for row in render_manifest(files)["stems"]] == [second.stem_id]


@requires_plotting
def test_gallery_escapes_metadata_and_uses_local_hashed_links(files):
    hostile = '<script>alert("stem")</script> & dog'
    manifest = load_json(files.manifest)
    manifest["stems"][0]["source_description"] = hostile
    write_json(files.manifest, manifest)
    assert main(visual_args(files)) == 0
    html = (files.output / "index.html").read_text(encoding="utf-8")
    assert hostile not in html
    assert "&lt;script&gt;" in html
    assert "&amp; dog" in html
    assert "<script" not in html.lower()
    assert "https://" not in html.lower()
    assert "http://" not in html.lower()
    assert f"plots/{stem_key(files.stem.stem_id)}.png" in html


@pytest.mark.parametrize("target_kind", ["gallery", "index", "plot"])
@pytest.mark.parametrize(
    "source_name", ["manifest", "index", "mapping", "config", "audio", "cache"]
)
def test_visualization_preflights_all_output_hardlinks(files, capsys, target_kind, source_name):
    source = getattr(files, source_name)
    targets = {
        "gallery": files.output / "index.html",
        "index": files.output / "visualization-index.json",
        "plot": files.output / "plots" / f"{stem_key(files.stem.stem_id)}.png",
    }
    target = targets[target_kind]
    target.parent.mkdir(parents=True)
    try:
        target.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = source.read_bytes()
    assert main(visual_args(files)) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert source.read_bytes() == target.read_bytes() == original
    assert {path for path in files.output.rglob("*") if path.is_file()} == {target}


@pytest.mark.parametrize("kind", ["source_media", "source_metadata", "invalid_cache_audio"])
def test_visualization_protects_provenance_inputs_before_writes(files, tmp_path, capsys, kind):
    source = tmp_path / "source-original.dat"
    source.write_bytes(b"original source bytes")
    if kind == "invalid_cache_audio":
        with np.load(files.cache, allow_pickle=False) as archive:
            payload = dict(archive)
        payload["probabilities"] = np.full((4, 1), 2.0)
        metadata = json.loads(payload["metadata_json"].item())
        metadata["audio_path"] = str(source)
        payload["metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(files.cache, **payload)
    else:
        manifest = load_json(files.manifest)
        provenance = manifest["stems"][0]["provenance"]
        if kind == "source_media":
            provenance["source_media_paths"] = [str(source)]
        else:
            provenance["inputs"] = {"original": {"path": str(source), "sha256": sha256(source)}}
        write_json(files.manifest, manifest)
    files.output.mkdir()
    alias = files.output / "index.html"
    try:
        alias.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    assert main(visual_args(files)) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert alias.read_bytes() == source.read_bytes() == b"original source bytes"
    assert not (files.output / "plots").exists()


@pytest.mark.parametrize("problem", ["unknown_stem", "unknown_index_id"])
def test_invalid_selection_or_index_fails_before_creating_outputs(files, problem):
    extra = []
    if problem == "unknown_stem":
        extra = ["--stem-id", "no-such-stem"]
    else:
        write_json(files.index, {"schema_version": 1, "predictions": {"unknown": "raw.npz"}})
    assert main(visual_args(files, *extra)) == 1
    assert not files.output.exists()


@pytest.mark.parametrize("max_points", ["0", "1", "-10"])
def test_invalid_waveform_point_limit_fails_before_writing(files, max_points):
    assert main(visual_args(files, "--max-waveform-points", max_points)) == 1
    assert not files.output.exists()


def test_selecting_one_stem_still_protects_other_stem_sources(files, tmp_path, capsys):
    other_audio = tmp_path / "other-source.wav"
    other_audio.write_bytes(b"unselected source")
    second = replace(
        files.stem,
        stem_id="other::candidate",
        clip_key="other",
        audio_path=other_audio,
    )
    manifest = load_json(files.manifest)
    manifest["stems"].append(second.to_dict())
    manifest["summary"].update(stem_count=2, clip_count=2)
    write_json(files.manifest, manifest)
    files.output.mkdir()
    alias = files.output / "index.html"
    try:
        alias.hardlink_to(other_audio)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    assert main(visual_args(files, "--stem-id", files.stem.stem_id)) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert other_audio.read_bytes() == b"unselected source"
    assert not (files.output / "plots").exists()


@requires_plotting
def test_visualization_never_imports_inference_dependencies(files):
    result = fresh_process(visual_args(files), ["torch", "torchaudio", "librosa"])
    assert result.returncode == 0, result.stderr
    assert render_manifest(files)["summary"]["rendered_count"] == 1


@pytest.mark.parametrize("dependency", ["matplotlib", "soundfile"])
def test_missing_plot_dependency_reports_actionable_error(files, dependency):
    result = fresh_process(visual_args(files), [dependency], missing_dependency=True)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert dependency in result.stderr or "visual" in result.stderr.lower()


def test_batch_cache_hits_use_pinned_caches_and_index_can_feed_evaluate(files):
    cache = pinned_cache(files)
    original = cache.read_bytes()
    result = fresh_process(
        batch_args(files), ["torch", "torchaudio", "librosa", "soundfile", "matplotlib"]
    )
    assert result.returncode == 0, result.stderr
    index = load_json(files.batch_output / "prediction-index.json")
    assert index["schema_version"] == 1
    indexed = files.batch_output / index["predictions"][files.stem.stem_id]
    assert indexed.resolve() == cache.resolve()
    assert cache.read_bytes() == original
    status = load_json(files.batch_output / "batch-status.json")
    assert status["summary"]["cached_count"] == 1
    assert status["summary"]["failed_count"] == 0
    assert status["summary"]["backend_initializations"] == 0
    assert status["provenance"]["inference_performed"] is False
    assert status["provenance"]["inputs"]["stems"] == {
        "path": str(files.manifest.resolve()),
        "sha256": sha256(files.manifest),
    }
    evaluate_output = files.batch_output / "reports"
    args = visual_args(files)
    args[0] = "evaluate"
    args[args.index("--predictions") + 1] = str(files.batch_output / "prediction-index.json")
    args[args.index("--output-dir") + 1] = str(evaluate_output)
    assert main(args) == 0
    assert (
        load_json(evaluate_output / "dataset.json")["stems"][0]["evaluation_status"] == "evaluated"
    )


def test_batch_missing_audio_keeps_successful_cache_in_index(files):
    pinned_cache(files)
    manifest = load_json(files.manifest)
    missing = replace(files.stem, stem_id="clip::absent", candidate_id="absent", audio_path=None)
    manifest["stems"].append(missing.to_dict())
    manifest["summary"]["stem_count"] = 2
    write_json(files.manifest, manifest)
    result = fresh_process(
        batch_args(files), ["torch", "torchaudio", "librosa", "soundfile", "matplotlib"]
    )
    assert result.returncode == 1, result.stderr
    index = load_json(files.batch_output / "prediction-index.json")
    assert set(index["predictions"]) == {files.stem.stem_id}
    status = load_json(files.batch_output / "batch-status.json")
    assert status["summary"]["cached_count"] == 1
    assert status["summary"]["failed_count"] == 1
    assert status["summary"]["backend_initializations"] == 0
    rows = {row["stem_id"]: row for row in status["stems"]}
    assert rows[missing.stem_id]["status"] == "missing_audio"


@pytest.mark.parametrize("problem", ["checkpoint", "vocabulary", "corrupt"])
def test_batch_conflicting_cache_stays_untouched_without_loading_model(files, problem):
    cache = pinned_cache(files)
    if problem == "corrupt":
        cache.write_bytes(b"invalid cache, preserved for explicit overwrite")
    else:
        prediction = FramePrediction.load(cache)
        if problem == "checkpoint":
            prediction.metadata["checkpoint_sha256"] = "0" * 64
        else:
            prediction.class_ids = tuple(reversed(prediction.class_ids))
            prediction.class_names = tuple(reversed(prediction.class_names))
        prediction.save(cache)
    original = cache.read_bytes()
    result = fresh_process(
        batch_args(files), ["torch", "torchaudio", "librosa", "soundfile", "matplotlib"]
    )
    assert result.returncode == 1, result.stderr
    assert cache.read_bytes() == original
    assert load_json(files.batch_output / "prediction-index.json")["predictions"] == {}
    status = load_json(files.batch_output / "batch-status.json")
    assert status["summary"]["failed_count"] == 1
    assert status["summary"]["backend_initializations"] == 0
    assert status["stems"][0]["status"] == "cache_conflict"


@pytest.mark.parametrize("target_name", ["prediction-index.json", "batch-status.json"])
def test_batch_protects_metadata_from_report_hardlink_before_writes(files, capsys, target_name):
    files.batch_output.mkdir()
    target = files.batch_output / target_name
    try:
        target.hardlink_to(files.manifest)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = files.manifest.read_bytes()
    assert main(batch_args(files)) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert files.manifest.read_bytes() == target.read_bytes() == original
    assert not (files.batch_output / "predictions").exists()


@pytest.mark.parametrize("threads", ["0", "-1"])
def test_batch_invalid_threads_fail_before_writing(files, threads):
    assert main(batch_args(files, "--threads", threads)) == 1
    assert not files.batch_output.exists()
