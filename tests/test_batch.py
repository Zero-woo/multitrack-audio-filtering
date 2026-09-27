"""Batch processing remains resumable and protects source data before writes."""

import builtins
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from waves_sed import __version__
from waves_sed.batch import run_batch
from waves_sed.labels import load_labels
from waves_sed.metadata import StemMetadata
from waves_sed.phase4 import load_prediction_index
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import CHECKPOINT_SHA256, PREPROCESSING, UPSTREAM_REVISION, sha256


def make_prediction(audio):
    ids, names = load_labels()
    return FramePrediction(
        probabilities=np.zeros((2, len(ids)), dtype=np.float32),
        frame_start_seconds=np.array([0.0, 0.04]),
        frame_end_seconds=np.array([0.04, 0.06]),
        class_ids=ids,
        class_names=names,
        metadata={
            "audio_path": str(audio),
            "audio_sha256": sha256(audio),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "preprocessing": PREPROCESSING,
            "upstream_revision": UPSTREAM_REVISION,
            "package_version": __version__,
            "backend": "atst_f_strong",
            "device": "cpu",
            "purpose": "synthetic batch test fixture; no acoustic claims",
        },
    )


def cache_path(output, stem):
    digest = hashlib.sha256(stem.stem_id.encode("utf-8")).hexdigest()
    return output / "predictions" / f"{digest}.npz"


@pytest.fixture
def files(tmp_path):
    stems = []
    for index in range(2):
        audio = tmp_path / f"audio-{index}.wav"
        audio.write_bytes(f"synthetic audio identity {index}".encode())
        stems.append(
            StemMetadata(
                stem_id=f"clip::candidate_{index}",
                candidate_id=f"candidate_{index}",
                clip_key="clip",
                audio_path=audio,
                source_description="dog barking",
                role="onset",
                expected_intervals=((0.0, 0.06),),
            )
        )
    return SimpleNamespace(
        stems=stems, output=tmp_path / "output", checkpoint=tmp_path / "missing-checkpoint.pt"
    )


@pytest.fixture
def forbid_model_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchaudio", "librosa", "soundfile"}:
            raise AssertionError(f"Batch imported inference dependency: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def run(files, **kwargs):
    return run_batch(files.stems, files.output, checkpoint=files.checkpoint, **kwargs)


def read_index(files):
    return load_prediction_index(
        files.output / "prediction-index.json", {stem.stem_id for stem in files.stems}
    )


def test_all_cache_hits_do_not_import_model_or_read_checkpoint(files, forbid_model_imports):
    originals = {}
    for stem in files.stems:
        path = cache_path(files.output, stem)
        make_prediction(stem.audio_path).save(path)
        originals[path] = path.read_bytes()
    report = run(files, device="cuda")
    assert report["summary"] == {
        "stem_count": 2,
        "cached_count": 2,
        "inferred_count": 0,
        "failed_count": 0,
        "backend_initializations": 0,
    }
    assert report["provenance"]["inference_performed"] is False
    assert report["provenance"]["requested_device"] == "cuda"
    assert len(read_index(files)) == 2
    for row in report["stems"]:
        assert row["status"] == "cached"
        assert row["prediction_provenance"]["device"] == "cpu"
        assert row["cache_sha256"] == sha256(Path(row["cache_path"]))
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert json.loads((files.output / "batch-status.json").read_text()) == report


def test_one_backend_is_reused_for_two_stems(files, forbid_model_imports):
    initializations, predictions = [], []

    def factory(checkpoint, device):
        initializations.append((checkpoint, device))

        def predict(audio):
            predictions.append(audio)
            return make_prediction(audio)

        return SimpleNamespace(predict=predict)

    report = run(files, backend_factory=factory)
    assert initializations == [(files.checkpoint.resolve(), "cpu")]
    assert predictions == [stem.audio_path for stem in files.stems]
    assert report["summary"]["inferred_count"] == 2
    assert report["summary"]["backend_initializations"] == 1
    assert report["provenance"]["inference_performed"] is True
    assert set(read_index(files)) == {stem.stem_id for stem in files.stems}
    assert all(path.name.count(":") == 0 for path in read_index(files).values())


def test_default_backend_sets_threads_and_loads_model_once(files, monkeypatch):
    calls = []

    class FakeATST:
        def __init__(self, checkpoint, device):
            calls.append(("initialize", checkpoint, device))

        def _load_model(self):
            calls.append(("load_model",))

        def predict(self, audio):
            calls.append(("predict", audio))
            return make_prediction(audio)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: calls.append(("threads", threads))),
    )
    monkeypatch.setitem(
        sys.modules, "waves_sed.backends.atst", SimpleNamespace(ATSTBackend=FakeATST)
    )
    report = run(files, threads=2)
    assert calls == [
        ("threads", 2),
        ("initialize", files.checkpoint.resolve(), "cpu"),
        ("load_model",),
        *(("predict", stem.audio_path) for stem in files.stems),
    ]
    assert report["summary"]["inferred_count"] == 2


def test_default_model_load_failure_is_attempted_only_once(files, monkeypatch):
    calls = []

    class FakeATST:
        def __init__(self, checkpoint, device):
            pass

        def _load_model(self):
            calls.append("load")
            raise RuntimeError("bad model structure")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setitem(
        sys.modules, "waves_sed.backends.atst", SimpleNamespace(ATSTBackend=FakeATST)
    )
    report = run(files)
    assert calls == ["load"]
    assert report["summary"]["failed_count"] == 2
    assert report["summary"]["backend_initializations"] == 1
    assert all("bad model structure" in row["error"] for row in report["stems"])


@pytest.mark.parametrize(
    "field",
    [
        "audio_sha256",
        "checkpoint_sha256",
        "preprocessing",
        "upstream_revision",
        "package_version",
        "backend",
        "class_ids",
        "class_names",
        "corrupt",
    ],
)
def test_cache_conflicts_are_kept_and_unindexed(files, forbid_model_imports, field):
    files.stems = files.stems[:1]
    stem = files.stems[0]
    path = cache_path(files.output, stem)
    prediction = make_prediction(stem.audio_path)
    if field in {"class_ids", "class_names"}:
        items = list(getattr(prediction, field))
        items[0], items[1] = items[1], items[0]
        setattr(prediction, field, tuple(items))
    elif field != "corrupt":
        prediction.metadata[field] = "different"
    prediction.save(path)
    if field == "corrupt":
        path.write_bytes(b"not an npz")
    before = path.read_bytes()
    report = run(files)
    assert report["summary"]["failed_count"] == 1
    assert report["summary"]["backend_initializations"] == 0
    assert report["stems"][0]["status"] == "cache_conflict"
    assert path.read_bytes() == before
    assert read_index(files) == {}


@pytest.mark.parametrize("existing", ["matching", "mismatch", "corrupt"])
def test_overwrite_recomputes_existing_cache(files, existing):
    files.stems = files.stems[:1]
    stem = files.stems[0]
    path = cache_path(files.output, stem)
    prediction = make_prediction(stem.audio_path)
    if existing == "mismatch":
        prediction.metadata["preprocessing"] = "different"
    prediction.save(path)
    if existing == "corrupt":
        path.write_bytes(b"corrupt")
    calls = []

    def predict(audio):
        calls.append(audio)
        result = make_prediction(audio)
        result.probabilities[:] = 0.75
        return result

    report = run(
        files,
        overwrite=True,
        backend_factory=lambda checkpoint, device: SimpleNamespace(predict=predict),
    )
    assert calls == [stem.audio_path]
    assert report["summary"]["inferred_count"] == 1
    assert np.all(FramePrediction.load(path).probabilities == 0.75)


def test_missing_audio_and_empty_dataset_never_initialize_backend(files, forbid_model_imports):
    files.stems = [
        replace(files.stems[0], audio_path=None),
        replace(files.stems[1], audio_path=files.output / "missing.wav"),
    ]
    report = run(files)
    assert report["summary"]["failed_count"] == 2
    assert report["summary"]["backend_initializations"] == 0
    assert [row["status"] for row in report["stems"]] == ["missing_audio"] * 2
    assert read_index(files) == {}
    files.stems = []
    report = run(files)
    assert all(value == 0 for value in report["summary"].values())
    assert read_index(files) == {}


def test_failed_prediction_continues_with_same_backend(files):
    calls = []

    def predict(audio):
        calls.append(audio)
        if audio == files.stems[0].audio_path:
            raise RuntimeError("invalid source WAV")
        return make_prediction(audio)

    report = run(files, backend_factory=lambda checkpoint, device: SimpleNamespace(predict=predict))
    assert len(calls) == 2
    assert report["summary"]["failed_count"] == 1
    assert report["summary"]["inferred_count"] == 1
    assert report["summary"]["backend_initializations"] == 1
    assert "invalid source WAV" in report["stems"][0]["error"]
    assert set(read_index(files)) == {files.stems[1].stem_id}


def test_failed_initialization_is_not_retried_and_later_cache_can_succeed(files):
    third = replace(files.stems[0], stem_id="other::cached", candidate_id="cached")
    files.stems.append(third)
    make_prediction(third.audio_path).save(cache_path(files.output, third))
    calls = []

    def factory(checkpoint, device):
        calls.append(checkpoint)
        raise RuntimeError("checkpoint not available")

    report = run(files, backend_factory=factory)
    assert len(calls) == 1
    assert report["summary"]["failed_count"] == 2
    assert report["summary"]["cached_count"] == 1
    assert report["summary"]["backend_initializations"] == 1
    assert report["provenance"]["inference_performed"] is False
    assert all(row["reason_code"] == "backend_initialization_failed" for row in report["stems"][:2])
    assert set(read_index(files)) == {third.stem_id}


@pytest.mark.parametrize("change", ["source", "prediction_identity", "invalid_scores"])
def test_inference_output_must_be_valid_and_match_original_audio(files, change):
    files.stems = files.stems[:1]

    def predict(audio):
        prediction = make_prediction(audio)
        if change == "source":
            audio.write_bytes(b"changed while predicting")
        elif change == "prediction_identity":
            prediction.metadata["checkpoint_sha256"] = "wrong"
        else:
            prediction.probabilities[0, 0] = np.nan
        return prediction

    report = run(files, backend_factory=lambda checkpoint, device: SimpleNamespace(predict=predict))
    assert report["summary"]["failed_count"] == 1
    assert report["stems"][0]["status"] == "inference_failed"
    assert read_index(files) == {}
    assert not cache_path(files.output, files.stems[0]).exists()


def test_manifest_audio_hash_is_checked_before_reusing_cache(files, forbid_model_imports):
    files.stems = [replace(files.stems[0], provenance={"raw_final_stem": {"sha256": "0" * 64}})]
    stem = files.stems[0]
    make_prediction(stem.audio_path).save(cache_path(files.output, stem))
    report = run(files)
    assert report["stems"][0]["status"] == "source_hash_mismatch"
    assert report["summary"]["backend_initializations"] == 0
    assert read_index(files) == {}


@pytest.mark.parametrize("protected_kind", ["audio", "checkpoint", "explicit", "media", "input"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_complete_write_set_preflight_preserves_sources(files, protected_kind, hardlink):
    source = files.stems[0].audio_path
    output = files.output / "batch-status.json"
    output.parent.mkdir(parents=True)
    if hardlink:
        try:
            output.hardlink_to(source)
        except OSError as error:
            pytest.skip(f"Hard links unavailable: {error}")
    else:
        output.write_bytes(b"protected input bytes")
        source = output
    kwargs = {}
    if protected_kind == "audio":
        files.stems[0] = replace(files.stems[0], audio_path=source)
    elif protected_kind == "checkpoint":
        files.checkpoint = source
    elif protected_kind == "explicit":
        kwargs["protected_inputs"] = [source]
    elif protected_kind == "media":
        files.stems[0] = replace(files.stems[0], provenance={"source_media_paths": [str(source)]})
    else:
        files.stems[0] = replace(
            files.stems[0], provenance={"inputs": {"metadata": {"path": str(source)}}}
        )
    before = source.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite input"):
        run(files, overwrite=True, **kwargs)
    assert source.read_bytes() == before
    assert not (files.output / "prediction-index.json").exists()
    assert not (files.output / "predictions").exists()


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_preflight_protects_recorded_audio_even_in_invalid_or_obsolete_cache(files, corrupt, stale):
    source = files.output / "prediction-index.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source recording with a misleading extension")
    stem = files.stems[0]
    cache = cache_path(files.output, stem)
    if stale:
        cache = files.output / "predictions" / "old-run.npz"
    prediction = make_prediction(stem.audio_path)
    prediction.metadata["audio_path"] = str(source)
    prediction.save(cache)
    if corrupt:
        # Preserve parseable metadata while invalidating the score payload.
        with cache.open("wb") as stream:
            np.savez(stream, metadata_json=np.array(json.dumps(prediction.metadata)))
    before = source.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite input"):
        run(files, overwrite=True)
    assert source.read_bytes() == before
    assert not (files.output / "batch-status.json").exists()


def test_npz_target_cannot_overwrite_another_stem_audio(files):
    target = cache_path(files.output, files.stems[1])
    files.stems[0] = replace(files.stems[0], audio_path=target)
    with pytest.raises(ValueError, match="must not overwrite input"):
        run(files)
    assert not files.output.exists()


def test_two_output_paths_cannot_be_hardlinks(files):
    first, second = [cache_path(files.output, stem) for stem in files.stems]
    first.parent.mkdir(parents=True)
    first.write_bytes(b"existing cache")
    try:
        second.hardlink_to(first)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    with pytest.raises(ValueError, match="alias each other"):
        run(files, overwrite=True)
    assert first.read_bytes() == b"existing cache"
    assert not (files.output / "prediction-index.json").exists()


def test_old_files_are_preserved_but_not_in_current_index(files):
    for stem in files.stems:
        make_prediction(stem.audio_path).save(cache_path(files.output, stem))
    run(files)
    obsolete = cache_path(files.output, files.stems[1])
    original = obsolete.read_bytes()
    files.stems = files.stems[:1]
    run(files)
    assert set(read_index(files)) == {files.stems[0].stem_id}
    assert obsolete.read_bytes() == original


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_threads_fail_before_writing(files, threads):
    with pytest.raises(ValueError, match="threads"):
        run(files, threads=threads)
    assert not files.output.exists()


def test_duplicate_stem_ids_fail_before_writing(files):
    files.stems.append(files.stems[0])
    with pytest.raises(ValueError, match="unique"):
        run(files)
    assert not files.output.exists()
