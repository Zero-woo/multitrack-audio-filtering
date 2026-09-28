"""Controlled waveform edits and cached comparisons remain separate from inference."""

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from waves_sed.cli import main
from waves_sed.metadata import StemMetadata, load_stems
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256

requires_audio = pytest.mark.skipif(
    importlib.util.find_spec("soundfile") is None,
    reason="Requires the optional soundfile dependency for waveform corruption",
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
    sf = pytest.importorskip("soundfile")
    audio = tmp_path / "original.wav"
    samples = np.zeros((16000, 2), dtype=np.float32)
    samples[1600:3200, 0] = 0.25
    samples[1600:3200, 1] = -0.5
    samples[8000:9600, 0] = -0.75
    samples[8000:9600, 1] = 0.125
    sf.write(str(audio), samples, 16000, subtype="FLOAT")
    stem = StemMetadata(
        stem_id="clip::candidate",
        candidate_id="candidate",
        clip_key="clip",
        audio_path=audio,
        source_description="dog barking",
        role="onset",
        expected_intervals=((0.1, 0.2), (0.5, 0.6)),
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
        {"schema_version": 1, "stems": [stem.to_dict()]},
    )
    config = write_json(
        tmp_path / "corruptions.json",
        {
            "schema_version": 1,
            "variants": [
                {"id": "shift100", "operation": "shift", "shift_seconds": 0.1},
                {"id": "removed", "operation": "remove", "interval": [0.1, 0.2]},
            ],
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
    evaluation_config = write_json(
        tmp_path / "evaluation.json",
        {
            "schema_version": 1,
            "event": {"threshold": 0.5, "median_window_frames": 1, "min_duration_seconds": 0},
            "onset_tolerance_seconds": 0.04,
            "allow_ambiguous_reference": False,
            "outside_family_threshold": 0.5,
            "outside_family_top_k": 5,
        },
    )
    return SimpleNamespace(
        audio=audio,
        samples=samples,
        stem=stem,
        manifest=manifest,
        config=config,
        mapping=mapping,
        evaluation_config=evaluation_config,
        output=tmp_path / "experiment",
        reports=tmp_path / "reports",
        comparison=tmp_path / "comparison",
    )


def corrupt_args(files, stem_id=None):
    return [
        "corrupt",
        "--stems",
        str(files.manifest),
        "--stem-id",
        files.stem.stem_id if stem_id is None else stem_id,
        "--config",
        str(files.config),
        "--output-dir",
        str(files.output),
    ]


def compare_args(files):
    return [
        "compare-corruptions",
        "--experiment",
        str(files.output / "experiment.json"),
        "--report",
        str(files.reports / "dataset.json"),
        "--output-dir",
        str(files.comparison),
    ]


def fresh_process(args, blocked):
    script = """
import builtins
import json
import sys
original_import = builtins.__import__
blocked = set(json.loads(sys.argv[2]))
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Forbidden optional import: ' + name)
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


def synthetic_reports(files, *, missing_variant=None, wrong_variant_hash=None):
    """Deliberate score oracles check the plumbing, not ATST detection behavior."""
    stems = load_stems(files.output / "stems.json")
    raw_dir = files.output / "synthetic-scores"
    raw_dir.mkdir()
    index = {}
    patterns = {
        "control": [0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
        "shift100": [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
        "removed": [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
    }
    boundaries = np.linspace(0.0, 1.0, 11)
    for stem in stems:
        variant = stem.stem_id.split("::")[-1]
        if variant == missing_variant:
            continue
        prediction = FramePrediction(
            probabilities=np.asarray(patterns[variant], dtype=np.float32)[:, None],
            frame_start_seconds=boundaries[:-1],
            frame_end_seconds=boundaries[1:],
            class_ids=("/m/05tny_",),
            class_names=("Bark",),
            metadata={
                "backend": "synthetic_oracle",
                "audio_path": str(stem.audio_path),
                "audio_sha256": (
                    sha256(files.audio)
                    if variant == wrong_variant_hash
                    else sha256(stem.audio_path)
                ),
                "duration_seconds": 1.0,
                "purpose": "synthetic pipeline test, not a model-quality claim",
            },
        )
        cache = raw_dir / f"{stem_key(stem.stem_id)}.npz"
        prediction.save(cache)
        index[stem.stem_id] = str(cache)
    index_path = write_json(raw_dir / "index.json", {"schema_version": 1, "predictions": index})
    assert (
        main(
            [
                "evaluate",
                "--stems",
                str(files.output / "stems.json"),
                "--predictions",
                str(index_path),
                "--mappings",
                str(files.mapping),
                "--config",
                str(files.evaluation_config),
                "--output-dir",
                str(files.reports),
            ]
        )
        == 0
    )
    return load_json(files.reports / "dataset.json")


@requires_audio
def test_generate_independent_variants_preserves_original_and_native_samples(files):
    import soundfile as sf

    original = files.audio.read_bytes()
    assert main(corrupt_args(files)) == 0
    assert files.audio.read_bytes() == original
    experiment = load_json(files.output / "experiment.json")
    stems = load_stems(files.output / "stems.json")
    expected_ids = {
        files.stem.stem_id + "::control",
        files.stem.stem_id + "::corruption::shift100",
        files.stem.stem_id + "::corruption::removed",
    }
    assert {stem.stem_id for stem in stems} == expected_ids
    control = experiment["control"]
    assert Path(control["audio_path"]).resolve() == files.audio.resolve()
    assert control["audio_sha256"] == sha256(files.audio)
    assert experiment["reference_policy"] == "unchanged_from_source"
    assert experiment["stems_file"]["sha256"] == sha256(files.output / "stems.json")
    assert experiment["config"]["sha256"] == sha256(files.config)
    rows = {row["id"]: row for row in experiment["variants"]}
    for variant, row in rows.items():
        assert row["status"] == "generated"
        path = Path(row["audio_path"])
        assert path == files.output / "audio" / f"{stem_key(row['stem_id'])}.wav"
        assert row["audio_sha256"] == sha256(path)
        info = sf.info(path)
        assert (info.samplerate, info.channels, info.frames, info.subtype) == (
            16000,
            2,
            16000,
            "FLOAT",
        )
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        expected = files.samples.copy()
        if variant == "shift100":
            expected[:] = 0
            expected[1600:] = files.samples[:-1600]
        else:
            expected[1600:3200] = 0
        np.testing.assert_array_equal(samples, expected)
        assert rate == 16000
    for stem in stems:
        assert stem.expected_intervals == files.stem.expected_intervals
        assert stem.role == files.stem.role
        assert stem.source_description == files.stem.source_description
        assert stem.clip_key == files.stem.clip_key
        assert stem.provenance["reference_status"] == "planned"
        assert stem.provenance["reference_origin"] == "waves_pass1_planned"
        assert stem.provenance["corruption"]["source_stem"] == files.stem.to_dict()
        assert stem.provenance["raw_final_stem"]["sha256"] == sha256(stem.audio_path)


@pytest.mark.parametrize("reference_status", ["ambiguous", "missing"])
def test_generation_does_not_upgrade_reference_status(files, reference_status):
    manifest = load_json(files.manifest)
    manifest["stems"][0]["provenance"]["reference_status"] = reference_status
    if reference_status == "missing":
        manifest["stems"][0]["expected_intervals"] = None
        manifest["stems"][0]["provenance"]["reference_origin"] = None
    write_json(files.manifest, manifest)
    assert main(corrupt_args(files)) == 0
    stems = load_stems(files.output / "stems.json")
    assert all(stem.provenance["reference_status"] == reference_status for stem in stems)
    if reference_status == "missing":
        assert all(stem.expected_intervals is None for stem in stems)


def test_out_of_bounds_variant_is_recorded_and_later_variant_runs(files):
    config = load_json(files.config)
    config["variants"].insert(0, {"id": "outside", "operation": "remove", "interval": [1.1, 1.2]})
    write_json(files.config, config)
    assert main(corrupt_args(files)) == 1
    experiment = load_json(files.output / "experiment.json")
    rows = {row["id"]: row for row in experiment["variants"]}
    assert rows["outside"]["status"] == "failed"
    assert rows["outside"]["error"]
    assert rows["outside"]["stem"] is None
    assert rows["shift100"]["status"] == rows["removed"]["status"] == "generated"
    assert len(load_stems(files.output / "stems.json")) == 3
    assert not (files.output / "audio" / f"{stem_key(rows['outside']['stem_id'])}.wav").exists()


def test_all_failed_variants_still_leave_a_readable_control_manifest(files):
    write_json(
        files.config,
        {
            "schema_version": 1,
            "variants": [{"id": "outside", "operation": "remove", "interval": [1.1, 1.2]}],
        },
    )
    assert main(corrupt_args(files)) == 1
    stems = load_stems(files.output / "stems.json")
    assert [stem.stem_id for stem in stems] == [files.stem.stem_id + "::control"]
    experiment = load_json(files.output / "experiment.json")
    assert experiment["variants"][0]["status"] == "failed"
    assert Path(experiment["control"]["audio_path"]).resolve() == files.audio.resolve()


def test_generated_audio_filename_does_not_interpret_variant_id_as_a_path(files):
    variant_id = "../한글: variant"
    write_json(
        files.config,
        {
            "schema_version": 1,
            "variants": [{"id": variant_id, "operation": "remove", "interval": [0.1, 0.2]}],
        },
    )
    assert main(corrupt_args(files)) == 0
    row = load_json(files.output / "experiment.json")["variants"][0]
    assert row["id"] == variant_id
    assert Path(row["audio_path"]) == (
        files.output
        / "audio"
        / f"{stem_key(files.stem.stem_id + '::corruption::' + variant_id)}.wav"
    )
    assert len(load_stems(files.output / "stems.json")) == 2


@pytest.mark.parametrize("problem", ["missing_audio", "manifest_hash", "unknown_stem"])
def test_invalid_source_fails_before_writing(files, problem):
    manifest = load_json(files.manifest)
    selection = files.stem.stem_id
    if problem == "missing_audio":
        manifest["stems"][0]["audio_path"] = str(files.audio.with_name("missing.wav"))
    elif problem == "manifest_hash":
        manifest["stems"][0]["provenance"]["raw_final_stem"]["sha256"] = "0" * 64
    else:
        selection = "unknown::candidate"
    write_json(files.manifest, manifest)
    assert main(corrupt_args(files, selection)) == 1
    assert not files.output.exists()


@pytest.mark.parametrize(
    "document",
    [
        '{"schema_version":1,"schema_version":1,"variants":[]}',
        '{"schema_version":1,"variants":[{"id":"x","operation":"shift","shift_seconds":NaN}]}',
        '{"schema_version":1,"variants":[{"id":"x","operation":"shift","shift_seconds":0.1},{"id":"x","operation":"remove","interval":[0.1,0.2]}]}',
        '{"schema_version":1,"variants":[{"id":"x","operation":"unknown"}]}',
    ],
)
def test_invalid_configuration_is_rejected_before_outputs(files, document):
    files.config.write_text(document, encoding="utf-8")
    assert main(corrupt_args(files)) == 1
    assert not files.output.exists()


def test_existing_experiment_is_immutable(files):
    assert main(corrupt_args(files)) == 0
    before = {path: path.read_bytes() for path in files.output.rglob("*") if path.is_file()}
    config = load_json(files.config)
    config["variants"][0]["shift_seconds"] = 0.2
    write_json(files.config, config)
    assert main(corrupt_args(files)) == 1
    assert {path: path.read_bytes() for path in files.output.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("target_kind", ["experiment", "manifest", "audio"])
@pytest.mark.parametrize("source_name", ["audio", "manifest", "config"])
def test_generation_preflights_every_output_before_first_write(files, target_kind, source_name):
    target = {
        "experiment": files.output / "experiment.json",
        "manifest": files.output / "stems.json",
        "audio": files.output
        / "audio"
        / f"{stem_key(files.stem.stem_id + '::corruption::shift100')}.wav",
    }[target_kind]
    source = getattr(files, source_name)
    target.parent.mkdir(parents=True)
    try:
        target.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = source.read_bytes()
    assert main(corrupt_args(files)) == 1
    assert source.read_bytes() == target.read_bytes() == original
    assert {path for path in files.output.rglob("*") if path.is_file()} == {target}


@pytest.mark.parametrize("source_kind", ["audio", "source_media_paths", "inputs"])
def test_unselected_stem_provenance_is_protected(files, source_kind):
    target = files.output / "experiment.json"
    second = replace(files.stem, stem_id="other::candidate", candidate_id="other")
    other = second.to_dict()
    if source_kind == "audio":
        other["audio_path"] = str(target)
    elif source_kind == "source_media_paths":
        other["provenance"]["source_media_paths"] = [str(target)]
    else:
        other["provenance"]["inputs"] = {"source": {"path": str(target)}}
    manifest = load_json(files.manifest)
    manifest["stems"].append(other)
    write_json(files.manifest, manifest)
    assert main(corrupt_args(files)) == 1
    assert not files.output.exists()


def test_corruption_command_imports_no_model_or_plot_dependencies(files):
    result = fresh_process(corrupt_args(files), ["torch", "torchaudio", "librosa", "matplotlib"])
    assert result.returncode == 0, result.stderr
    assert len(load_stems(files.output / "stems.json")) == 3


def test_generated_manifest_runs_evaluation_then_pairwise_comparison(files):
    assert main(corrupt_args(files)) == 0
    dataset = synthetic_reports(files)
    assert all(row["evaluation_status"] == "evaluated" for row in dataset["stems"])
    assert main(compare_args(files)) == 0
    report = load_json(files.comparison / "comparison.json")
    assert report["summary"]["variant_count"] == 2
    assert report["summary"]["compared_count"] == 2
    assert report["summary"]["unavailable_count"] == 0
    rows = {row["variant_id"]: row for row in report["comparisons"]}
    assert rows["shift100"]["comparison_status"] == "compared"
    assert rows["shift100"]["metric_deltas"]["matched_event_count"] == -2
    assert rows["shift100"]["metric_deltas"]["missing_event_count"] == 2
    assert rows["shift100"]["metric_deltas"]["event_recall"] == -1
    assert rows["removed"]["metric_deltas"]["missing_event_count"] == 1
    assert rows["removed"]["metric_deltas"]["event_recall"] == -0.5
    with (files.comparison / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert {row["variant_id"] for row in csv_rows} == {"shift100", "removed"}


@pytest.mark.parametrize("problem", ["missing_variant", "wrong_variant_hash"])
def test_unavailable_variant_is_not_reported_as_a_zero_delta(files, problem):
    assert main(corrupt_args(files)) == 0
    synthetic_reports(files, **{problem: "removed"})
    assert main(compare_args(files)) == 0
    report = load_json(files.comparison / "comparison.json")
    assert report["summary"]["compared_count"] == 1
    assert report["summary"]["unavailable_count"] == 1
    rows = {row["variant_id"]: row for row in report["comparisons"]}
    assert rows["removed"]["comparison_status"] == "unavailable"
    assert rows["removed"]["reason_codes"]
    assert not rows["removed"]["metric_deltas"] or all(
        value is None for value in rows["removed"]["metric_deltas"].values()
    )


def test_comparison_is_model_audio_and_plot_dependency_free(files):
    assert main(corrupt_args(files)) == 0
    synthetic_reports(files)
    result = fresh_process(
        compare_args(files), ["torch", "torchaudio", "librosa", "soundfile", "matplotlib"]
    )
    assert result.returncode == 0, result.stderr
    assert load_json(files.comparison / "comparison.json")["summary"]["compared_count"] == 2


@pytest.mark.parametrize("target_name", ["comparison.json", "comparison.csv"])
@pytest.mark.parametrize("source_kind", ["audio", "experiment", "dataset", "config"])
def test_comparison_preflights_source_and_provenance_hardlinks(files, target_name, source_kind):
    assert main(corrupt_args(files)) == 0
    synthetic_reports(files)
    source = {
        "audio": files.audio,
        "experiment": files.output / "experiment.json",
        "dataset": files.reports / "dataset.json",
        "config": files.evaluation_config,
    }[source_kind]
    files.comparison.mkdir()
    target = files.comparison / target_name
    try:
        target.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = source.read_bytes()
    assert main(compare_args(files)) == 1
    assert source.read_bytes() == target.read_bytes() == original
    assert {path for path in files.comparison.rglob("*") if path.is_file()} == {target}


@pytest.mark.parametrize("input_name", ["experiment", "dataset"])
@pytest.mark.parametrize("problem", ["duplicate", "nonfinite"])
def test_comparison_rejects_ambiguous_json_before_output(files, input_name, problem):
    assert main(corrupt_args(files)) == 0
    synthetic_reports(files)
    path = (
        files.output / "experiment.json"
        if input_name == "experiment"
        else files.reports / "dataset.json"
    )
    original = path.read_text(encoding="utf-8")
    prefix = '"schema_version": 1,' if problem == "duplicate" else '"nonfinite": NaN,'
    path.write_text("{" + prefix + original.lstrip()[1:], encoding="utf-8")
    assert main(compare_args(files)) == 1
    assert not files.comparison.exists()


@pytest.mark.parametrize(
    "recording", ["source_attempt_file", "mixed_audio", "video", "stem_index", "clip_index"]
)
def test_comparison_protects_nested_media_and_report_index(files, recording):
    assert main(corrupt_args(files)) == 0
    synthetic_reports(files)
    source = files.reports / "recorded-input.bin"
    source.write_bytes(b"keep this recorded input")
    path = files.reports / "dataset.json"
    dataset = load_json(path)
    if recording in {"stem_index", "clip_index"}:
        dataset["files"]["stems" if recording == "stem_index" else "clips"]["recorded"] = (
            source.name
        )
    else:
        dataset["provenance"]["raw_final_stem"] = {recording: str(source)}
    write_json(path, dataset)
    files.comparison.mkdir()
    target = files.comparison / "comparison.json"
    target.hardlink_to(source)
    assert main(compare_args(files)) == 1
    assert source.read_bytes() == target.read_bytes() == b"keep this recorded input"
    assert not (files.comparison / "comparison.csv").exists()


def test_comparison_detects_input_change_before_writing(files, monkeypatch):
    from waves_sed import phase6

    assert main(corrupt_args(files)) == 0
    synthetic_reports(files)
    original_load = phase6._load

    def changed_load(path):
        value = original_load(path)
        if path.name == "dataset.json":
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return value

    monkeypatch.setattr(phase6, "_load", changed_load)
    assert main(compare_args(files)) == 1
    assert not files.comparison.exists()
