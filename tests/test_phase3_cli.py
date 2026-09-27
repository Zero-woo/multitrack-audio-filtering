"""Phase 3 CLI joins and curves work offline and never overwrite their inputs."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from waves_sed.cli import main
from waves_sed.labels import load_labels
from waves_sed.prediction import FramePrediction


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def source_files(tmp_path):
    final_dir = tmp_path / "final" / "clip"
    final_dir.mkdir(parents=True)
    audio = final_dir / "stem.wav"
    audio.write_bytes(b"synthetic source identity; audio is never decoded by Phase 3")
    metadata = write_json(
        final_dir / "metadata.json",
        {
            "key": "clip",
            "video_id": "video-1",
            "stems": [
                {
                    "candidate_id": "clip__01",
                    "selected_attempt": 2,
                    "final_label": "Final source",
                    "final_role": "onset",
                    "merged_candidate_ids": [],
                    "file": "stem.wav",
                }
            ],
        },
    )
    dsp = write_json(
        tmp_path / "dsp.json",
        {
            "key": "clip",
            "candidates": [
                {
                    "candidate_id": "clip__01",
                    "label": "Planned source",
                    "description": "The original planned description",
                    "role": "span",
                    "activity_intervals": [[0.2, 0.5]],
                    "attempts": [{"attempt": 2, "audio_id": "clip/01/a2"}],
                }
            ],
        },
    )
    ids, names = load_labels()
    mappings = write_json(
        tmp_path / "mappings.json",
        {
            "schema_version": 1,
            "mappings": {
                "final_semantics": {
                    "descriptions": ["Final source"],
                    "allowed_classes": [ids[0], ids[1]],
                    "foreign_classes": [ids[2]],
                },
                "original_plan": {
                    "descriptions": ["The original planned description"],
                    "allowed_classes": [ids[2]],
                },
            },
        },
    )
    cache = tmp_path / "raw.npz"
    prediction = FramePrediction(
        probabilities=np.array([[0.99, 0.25, 0.75], [0.99, 0.625, 0.125]], dtype=np.float32),
        frame_start_seconds=np.array([0.0, 0.04]),
        frame_end_seconds=np.array([0.04, 0.06]),
        class_ids=(ids[2], ids[1], ids[0]),
        class_names=(names[2], names[1], names[0]),
        metadata={"audio_path": str(audio), "purpose": "synthetic CLI aggregation only"},
    )
    prediction.save(cache)
    return SimpleNamespace(
        audio=audio,
        metadata=metadata,
        dsp=dsp,
        mappings=mappings,
        cache=cache,
        ids=ids,
        prediction=prediction,
    )


def adapt_args(files, *extra):
    return [
        "adapt-waves",
        "--metadata",
        str(files.metadata),
        "--dsp-report",
        str(files.dsp),
        "--mappings",
        str(files.mappings),
        *extra,
    ]


def map_args(files, *extra):
    return [
        "map-source",
        "--description",
        "Final source",
        "--mappings",
        str(files.mappings),
        "--prediction",
        str(files.cache),
        *extra,
    ]


def test_adapt_preserves_plan_but_maps_final_relabelled_semantics(source_files, tmp_path, capsys):
    output = tmp_path / "normalized.json"
    assert main(adapt_args(source_files, "--output", str(output))) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out)["stem_count"] == 1
    assert report["input_format"] == "waves_materialized"
    assert report["temporal_reference"] == "waves_pass1_planned"
    assert report["summary"]["mapping_statuses"] == {"supported": 1}
    row = report["stems"][0]
    assert row["audio_path"] == str(source_files.audio.resolve())
    assert row["source_description"] == "Final source"
    assert row["description_origin"] == "final_label"
    assert row["planned_description"] == "The original planned description"
    assert row["role"] == "onset"
    assert row["planned_role"] == "span"
    assert row["expected_intervals"] == [[0.2, 0.5]]
    assert row["provenance"]["reference_status"] == "ambiguous"
    assert {"relabelled", "role_changed"}.issubset(row["issues"])
    assert row["mapping"]["mapping_key"] == "final_semantics"
    assert row["mapping"]["target_class_ids"] == list(source_files.ids[:2])


def test_map_source_aggregates_raw_cache_by_class_id(source_files, tmp_path, capsys):
    output = tmp_path / "mapped.json"
    original = source_files.cache.read_bytes()
    assert main(map_args(source_files, "--output", str(output))) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out)["mapping_status"] == "supported"
    assert report["target_probability"] == [0.75, 0.625]
    assert report["aggregation"] == "max_allowed_class_probability"
    assert report["frame_start_seconds"] == [0.0, 0.04]
    assert report["frame_end_seconds"] == [0.04, 0.06]
    assert report["prediction"]["metadata"] == source_files.prediction.metadata
    assert report["summary"]["decision"] is None
    assert source_files.cache.read_bytes() == original


def test_unsupported_description_keeps_null_target_probability(source_files, capsys):
    assert main(map_args(source_files, "--description", "Unmapped sound")) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["mapping"]["status"] == "unsupported_mapping"
    assert report["mapping"]["reason"] == "no_explicit_mapping"
    assert report["target_probability"] is None
    assert report["summary"]["decision"] is None
    assert "aggregation" not in report
    assert report["summary"]["frame_count"] == 2


def test_mapping_without_prediction_resolves_but_does_not_invent_scores(source_files, capsys):
    assert (
        main(
            [
                "map-source",
                "--description",
                "Final source",
                "--mappings",
                str(source_files.mappings),
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["mapping"]["status"] == "supported"
    assert report["target_probability"] is None
    assert "frame_start_seconds" not in report
    assert "prediction" not in report


@pytest.mark.parametrize("protected_name", ["metadata", "dsp", "mappings", "audio"])
def test_adapt_output_cannot_overwrite_inputs(source_files, capsys, protected_name):
    protected = getattr(source_files, protected_name)
    original = protected.read_bytes()

    assert main(adapt_args(source_files, "--output", str(protected))) == 1

    assert "must not overwrite" in capsys.readouterr().err
    assert protected.read_bytes() == original


@pytest.mark.parametrize("protected_name", ["mappings", "cache", "audio"])
def test_mapping_output_cannot_overwrite_inputs(source_files, capsys, protected_name):
    protected = getattr(source_files, protected_name)
    original = protected.read_bytes()

    assert main(map_args(source_files, "--output", str(protected))) == 1

    assert "must not overwrite" in capsys.readouterr().err
    assert protected.read_bytes() == original


@pytest.mark.parametrize("field", ["source_attempt_file", "mixed_audio", "attempt_file", "video"])
def test_adapt_output_protects_original_media_referenced_by_inputs(
    source_files, tmp_path, capsys, field
):
    original_path = tmp_path / "original-media.wav"
    original_path.write_bytes(b"original media that must survive metadata export")
    metadata = json.loads(source_files.metadata.read_text(encoding="utf-8"))
    dsp = json.loads(source_files.dsp.read_text(encoding="utf-8"))
    if field == "source_attempt_file":
        metadata["stems"][0][field] = str(original_path)
    elif field == "mixed_audio":
        metadata[field] = str(original_path)
    elif field == "attempt_file":
        dsp["candidates"][0]["attempts"][0]["file"] = str(original_path)
    else:
        dsp["video"] = str(original_path)
    write_json(source_files.metadata, metadata)
    write_json(source_files.dsp, dsp)
    before = original_path.read_bytes()

    assert main(adapt_args(source_files, "--output", str(original_path))) == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert original_path.read_bytes() == before


@pytest.mark.parametrize("command_args", [adapt_args, map_args])
def test_output_cannot_overwrite_audio_through_hardlink(
    source_files, tmp_path, capsys, command_args
):
    target = tmp_path / "source-alias.json"
    try:
        target.hardlink_to(source_files.audio)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = source_files.audio.read_bytes()

    assert main(command_args(source_files, "--output", str(target))) == 1

    assert "must not overwrite" in capsys.readouterr().err
    assert source_files.audio.read_bytes() == original
    assert target.read_bytes() == original


def test_explicit_missing_candidate_source_is_an_error(source_files, tmp_path, capsys):
    output = tmp_path / "should-not-exist.json"
    missing = tmp_path / "missing-dsp.json"
    assert (
        main(adapt_args(source_files, "--dsp-report", str(missing), "--output", str(output))) == 1
    )

    assert "missing-dsp.json" in capsys.readouterr().err
    assert not output.exists()


def test_omitted_candidate_source_reports_missing_support(source_files, capsys):
    assert main(["adapt-waves", "--metadata", str(source_files.metadata)]) == 0

    row = json.loads(capsys.readouterr().out)["stems"][0]
    assert row["source_description"] == "Final source"
    assert row["expected_intervals"] is None
    assert row["provenance"]["reference_status"] == "missing"
    assert {"missing_support", "missing_candidate_metadata"}.issubset(row["issues"])


def test_frozen_final_requires_explicit_reports(source_files, capsys):
    assert main(["adapt-waves", "--frozen-finals", str(source_files.metadata)]) == 1
    assert "requires --frozen-reports" in capsys.readouterr().err


def test_phase3_commands_run_in_fresh_process_without_inference_imports(source_files, tmp_path):
    adapted = tmp_path / "offline-adapted.json"
    mapped = tmp_path / "offline-mapped.json"
    commands = [
        adapt_args(source_files, "--output", str(adapted)),
        map_args(source_files, "--output", str(mapped)),
    ]
    script = """
import builtins
import json
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchaudio', 'librosa', 'soundfile'}:
        raise AssertionError('Offline Phase 3 imported ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from waves_sed.cli import main
for args in json.loads(sys.argv[1]):
    code = main(args)
    if code:
        raise SystemExit(code)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(commands)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(adapted.read_text(encoding="utf-8"))["summary"]["stem_count"] == 1
    assert json.loads(mapped.read_text(encoding="utf-8"))["target_probability"] == [0.75, 0.625]
