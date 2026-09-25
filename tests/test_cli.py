"""Offline cache workflows must not load an inference library or alter inputs."""

import builtins
import csv
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from waves_sed import __version__
from waves_sed.labels import load_labels
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import CHECKPOINT_SHA256, PREPROCESSING, UPSTREAM_REVISION, sha256


@pytest.fixture
def cache_files(tmp_path):
    audio = tmp_path / "검증 입력.wav"
    audio.write_bytes(b"dummy audio bytes for cache identity checks; not a WAV")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint must not be read on a cache hit")
    cache = tmp_path / "prediction.npz"
    ids, names = load_labels()
    scores = np.zeros((2, len(ids)), dtype=np.float32)
    scores[0, 0] = 0.125
    scores[1, -1] = 0.875
    prediction = FramePrediction(
        probabilities=scores,
        frame_start_seconds=np.array([0.0, 0.04], dtype=np.float64),
        frame_end_seconds=np.array([0.04, 0.06], dtype=np.float64),
        class_ids=ids,
        class_names=names,
        metadata={
            "audio_sha256": sha256(audio),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "preprocessing": PREPROCESSING,
            "upstream_revision": UPSTREAM_REVISION,
            "package_version": __version__,
            "backend": "atst_f_strong",
            "purpose": "synthetic offline CLI fixture, not model predictions",
            "audio_path": str(audio),
        },
    )
    prediction.save(cache)
    return SimpleNamespace(audio=audio, checkpoint=checkpoint, cache=cache, prediction=prediction)


@pytest.fixture
def offline_main(monkeypatch):
    original_import = builtins.__import__

    def import_without_inference(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchaudio", "librosa", "soundfile"}:
            raise AssertionError(f"Offline workflow attempted to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_inference)
    from waves_sed.cli import main

    return main


def infer_args(files, *extra):
    return [
        "infer",
        str(files.audio),
        "--checkpoint",
        str(files.checkpoint),
        "--output",
        str(files.cache),
        *extra,
    ]


def test_inspect_and_csv_work_in_fresh_process_without_inference_imports(cache_files, tmp_path):
    csv_path = tmp_path / "raw probabilities.csv"
    cache_bytes = cache_files.cache.read_bytes()
    script = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchaudio', 'librosa', 'soundfile'}:
        raise AssertionError('Offline cache reader imported ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from waves_sed.cli import main
raise SystemExit(main(['inspect', sys.argv[1], '--csv', sys.argv[2]]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(cache_files.cache), str(csv_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["shape"] == [2, 447]
    assert report["end_seconds"] == pytest.approx(0.06)
    assert report["probability_min"] == 0.0
    assert report["probability_max"] == 0.875
    assert report["metadata"] == cache_files.prediction.metadata
    assert report["top_classes_by_peak"][0]["class_name"] == "Zipper (clothing)"
    assert cache_files.cache.read_bytes() == cache_bytes
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 447
    assert rows[0] == {
        "frame_start_seconds": "0.00000000",
        "frame_end_seconds": "0.04000000",
        "class_id": "/m/07q2z82",
        "class_name": "Accelerating, revving, vroom",
        "probability": "0.125",
    }
    assert rows[-1]["class_id"] == "/m/01s0vc"
    assert float(rows[-1]["frame_end_seconds"]) == pytest.approx(0.06)
    assert float(rows[-1]["probability"]) == 0.875


def test_infer_reuses_matching_cache_without_loading_audio_or_checkpoint(
    offline_main, cache_files, capsys
):
    original = cache_files.cache.read_bytes()
    assert offline_main(infer_args(cache_files)) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["cache_hit"] is True
    assert report["shape"] == [2, 447]
    assert "Loading" not in captured.err
    assert cache_files.cache.read_bytes() == original


@pytest.mark.parametrize(
    "field",
    [
        "audio_sha256",
        "checkpoint_sha256",
        "preprocessing",
        "upstream_revision",
        "package_version",
        "backend",
    ],
)
def test_infer_refuses_cache_with_mismatching_provenance(offline_main, cache_files, capsys, field):
    cache_files.prediction.metadata[field] = "different"
    cache_files.prediction.save(cache_files.cache)
    original = cache_files.cache.read_bytes()

    assert offline_main(infer_args(cache_files)) == 1
    assert "does not match" in capsys.readouterr().err
    assert cache_files.cache.read_bytes() == original


@pytest.mark.parametrize("field", ["class_ids", "class_names"])
def test_infer_refuses_cache_with_different_class_order(offline_main, cache_files, capsys, field):
    labels = list(getattr(cache_files.prediction, field))
    labels[0], labels[1] = labels[1], labels[0]
    setattr(cache_files.prediction, field, tuple(labels))
    cache_files.prediction.save(cache_files.cache)
    original = cache_files.cache.read_bytes()

    assert offline_main(infer_args(cache_files)) == 1
    assert "does not match" in capsys.readouterr().err
    assert cache_files.cache.read_bytes() == original


def test_infer_detects_changed_audio_contents(offline_main, cache_files, capsys):
    original = cache_files.cache.read_bytes()
    cache_files.audio.write_bytes(b"different source audio")

    assert offline_main(infer_args(cache_files)) == 1
    assert "does not match" in capsys.readouterr().err
    assert cache_files.cache.read_bytes() == original


@pytest.mark.parametrize("target", ["audio", "checkpoint"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_output_path_cannot_overwrite_audio_or_checkpoint(
    offline_main, cache_files, capsys, target, overwrite
):
    protected = getattr(cache_files, target)
    original = protected.read_bytes()
    args = infer_args(cache_files, "--output", str(protected))
    if overwrite:
        args.append("--overwrite")

    assert offline_main(args) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert protected.read_bytes() == original


@pytest.mark.parametrize("target", ["audio", "checkpoint", "cache"])
def test_infer_csv_cannot_overwrite_audio_checkpoint_or_cache(
    offline_main, cache_files, capsys, target
):
    protected = getattr(cache_files, target)
    original = protected.read_bytes()

    assert offline_main(infer_args(cache_files, "--csv", str(protected))) == 1
    assert "CSV must not overwrite" in capsys.readouterr().err
    assert protected.read_bytes() == original


def test_inspect_csv_cannot_overwrite_its_cache(offline_main, cache_files, capsys):
    original = cache_files.cache.read_bytes()

    assert offline_main(["inspect", str(cache_files.cache), "--csv", str(cache_files.cache)]) == 1
    assert "CSV must not overwrite" in capsys.readouterr().err
    assert cache_files.cache.read_bytes() == original


@pytest.mark.parametrize("hardlink", [False, True])
def test_inspect_csv_cannot_overwrite_source_audio(
    offline_main, cache_files, capsys, tmp_path, hardlink
):
    original = cache_files.audio.read_bytes()
    target = cache_files.audio
    if hardlink:
        target = tmp_path / "audio-alias.csv"
        try:
            target.hardlink_to(cache_files.audio)
        except OSError as error:
            pytest.skip(f"Hard links unavailable: {error}")

    assert offline_main(["inspect", str(cache_files.cache), "--csv", str(target)]) == 1
    assert "CSV must not overwrite source audio" in capsys.readouterr().err
    assert cache_files.audio.read_bytes() == original


def test_direct_csv_export_protects_source_audio(cache_files):
    from waves_sed.cli import export_csv

    original = cache_files.audio.read_bytes()
    with pytest.raises(ValueError, match="CSV must not overwrite source audio"):
        export_csv(cache_files.prediction, cache_files.audio)
    assert cache_files.audio.read_bytes() == original


@pytest.mark.parametrize("threads", ["0", "-1"])
def test_infer_requires_positive_thread_count(offline_main, cache_files, capsys, threads):
    original = cache_files.cache.read_bytes()

    assert offline_main(infer_args(cache_files, "--threads", threads)) == 1
    assert "--threads must be positive" in capsys.readouterr().err
    assert cache_files.cache.read_bytes() == original
