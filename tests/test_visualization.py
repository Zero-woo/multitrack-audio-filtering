"""Native-rate diagnostic plots preserve impulses and reject stale evidence."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig, EventConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import StemMetadata
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256
from waves_sed.temporal_reference import TemporalSupport
from waves_sed.visualization import (
    _build_figure,
    _envelope_from_reader,
    _step_values,
    _validate_report,
    _waveform,
    render_stem,
)

sf = pytest.importorskip("soundfile")
pytest.importorskip("matplotlib")


@pytest.fixture
def evidence(tmp_path):
    audio = tmp_path / "stem.wav"
    samples = np.zeros((44101, 2), dtype=np.float32)
    samples[11] = (0.8, -0.8)
    sf.write(audio, samples, 44100, subtype="FLOAT")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mappings": {
                    "bark": {"descriptions": ["dog barking"], "allowed_classes": ["/m/05tny_"]}
                },
            }
        ),
        encoding="utf-8",
    )
    mapping = ManualMapping.load(mapping_path)
    stem = StemMetadata(
        stem_id="clip::dog",
        candidate_id="dog",
        clip_key="clip",
        audio_path=audio,
        source_description="dog barking",
        role="onset",
        expected_intervals=((0, 0.5),),
        provenance={
            "reference_origin": "waves_pass1_planned",
            "reference_status": "planned",
            "raw_final_stem": {"sha256": sha256(audio)},
        },
    )
    prediction = FramePrediction(
        np.array([[0.9], [0.1], [0.8]], dtype=np.float32),
        np.array([0, 0.5, 1.0]),
        np.array([0.5, 1.0, 44101 / 44100]),
        ("/m/05tny_",),
        ("Bark",),
        {
            "audio_sha256": sha256(audio),
            "audio_path": str(audio),
            "backend": "synthetic",
            "duration_seconds": 44101 / 44100,
        },
    )
    cache = tmp_path / "raw.npz"
    prediction.save(cache)
    config = EvaluationConfig()
    report = evaluate_stem(stem, prediction, mapping, config, prediction_path=cache)
    return stem, prediction, report, mapping, config


def test_envelope_preserves_impulses_stereo_and_native_duration(evidence):
    stem, *_ = evidence
    metadata, envelope = _waveform(stem.audio_path, 7)
    assert metadata["status"] == "available"
    assert metadata["native_sample_rate"] == 44100
    assert metadata["channels"] == 2
    assert metadata["sample_count"] == 44101
    assert metadata["duration_seconds"] == 44101 / 44100
    assert metadata["envelope_bin_count"] == 7
    assert metadata["audio_sha256"] == sha256(stem.audio_path)
    assert envelope["maximum"][0] == pytest.approx(0.8)
    assert envelope["minimum"][0] == pytest.approx(-0.8)
    assert envelope["end_seconds"][-1] == 44101 / 44100
    assert np.all(envelope["end_seconds"][:-1] == envelope["start_seconds"][1:])


def test_streaming_reader_bounds_memory_and_visits_every_sample():
    class Reader:
        frames, samplerate, channels = 1_000_003, 48000, 2
        cursor = 0
        requested = []

        def read(self, frames, dtype, always_2d):
            self.requested.append(frames)
            result = np.zeros((frames, self.channels), dtype=dtype)
            for impulse, magnitude in ((123456, 0.8), (999999, -0.7)):
                if self.cursor <= impulse < self.cursor + frames:
                    result[impulse - self.cursor, 0] = magnitude
            self.cursor += frames
            return result

    reader = Reader()
    envelope = _envelope_from_reader(reader, 17)
    assert reader.cursor == reader.frames
    assert max(reader.requested) * reader.channels <= 65536
    assert sum(reader.requested) == reader.frames
    assert envelope["maximum"].max() == pytest.approx(0.8)
    assert envelope["minimum"].min() == pytest.approx(-0.7)
    assert len(envelope["minimum"]) == 17


def test_more_bins_than_samples_keeps_no_empty_bins(tmp_path):
    path = tmp_path / "tiny.wav"
    sf.write(path, [0.3, -0.4, 0.2], 22050, subtype="FLOAT")
    metadata, envelope = _waveform(path, 4000)
    assert metadata["envelope_bin_count"] == 3
    assert envelope["minimum"] == pytest.approx([0.3, -0.4, 0.2])
    assert np.all(envelope["end_seconds"] > envelope["start_seconds"])


def test_single_bin_keeps_both_polarities(evidence):
    stem, *_ = evidence
    _, envelope = _waveform(stem.audio_path, 1)
    assert envelope["minimum"] == pytest.approx([-0.8])
    assert envelope["maximum"] == pytest.approx([0.8])


@pytest.mark.parametrize("suffix,signature", [("png", b"\x89PNG\r\n\x1a\n"), ("svg", b"<?xml")])
def test_exports_and_keeps_raw_cache_unchanged(evidence, tmp_path, suffix, signature):
    stem, prediction, report, mapping, config = evidence
    raw = prediction.probabilities.copy()
    cache = Path(report["provenance"]["prediction"]["path"])
    before = cache.read_bytes()
    output = tmp_path / "plots" / f"stem.{suffix}"
    item = render_stem(stem, prediction, report, mapping, config, output, max_waveform_points=21)
    assert output.read_bytes().startswith(signature)
    assert item["target_curve_status"] == "available"
    assert item["waveform"]["envelope_bin_count"] == 21
    assert item["decision"] is None
    assert np.array_equal(prediction.probabilities, raw)
    assert cache.read_bytes() == before
    json.dumps(item, allow_nan=False)


def test_figure_uses_partial_frame_edges_and_unsmoothed_curve(evidence):
    stem, prediction, report, mapping, config = evidence
    config = replace(config, event=EventConfig(median_window_frames=3))
    report = evaluate_stem(
        stem,
        prediction,
        mapping,
        config,
        prediction_path=report["provenance"]["prediction"]["path"],
    )
    waveform, envelope = _waveform(stem.audio_path, 10)
    figure = _build_figure(
        stem, prediction, report, mapping, config, waveform, envelope, "available"
    )
    raw, smoothed, threshold = figure.axes[2].lines
    assert raw.get_ydata() == pytest.approx([0.9, 0.9, 0.1, 0.1, 0.8, 0.8])
    assert raw.get_xdata()[-1] == 44101 / 44100
    assert smoothed.get_ydata() == pytest.approx([0.9, 0.9, 0.8, 0.8, 0.8, 0.8])
    assert threshold.get_ydata() == [0.5, 0.5]
    assert all(axis.get_xlim()[1] == 44101 / 44100 for axis in figure.axes)
    figure.clear()


def test_gaps_are_not_interpolated():
    x, y = _step_values(np.array([0, 2]), np.array([1, 3]), np.array([0.2, 0.8]))
    assert np.isnan(x[2]) and np.isnan(y[2])
    assert x[[0, 1, 3, 4]].tolist() == [0, 1, 2, 3]


def test_ambiguous_support_hatched_and_outside_prediction_not_cropped(evidence):
    stem, prediction, old, mapping, config = evidence
    stem = replace(
        stem,
        expected_intervals=((0, 4),),
        provenance={**stem.provenance, "reference_status": "ambiguous"},
    )
    report = evaluate_stem(
        stem, prediction, mapping, config, prediction_path=old["provenance"]["prediction"]["path"]
    )
    waveform, envelope = _waveform(stem.audio_path, 10)
    status, _ = _validate_report(stem, prediction, report, mapping, config)
    assert status == "available"
    assert report["evaluation_status"] == "unavailable"
    figure = _build_figure(stem, prediction, report, mapping, config, waveform, envelope, status)
    assert figure.axes[1].collections[0].get_hatch() == "///"
    assert "excluded from metrics" in figure.axes[1].texts[0].get_text()
    assert all(axis.get_xlim() == (0, 4) for axis in figure.axes)
    assert "ground_truth" not in " ".join(text.get_text() for text in figure.texts)
    figure.clear()


@pytest.mark.parametrize(
    "field,value",
    [
        ("stem_id", "another"),
        ("source_description", "speech"),
        ("role", "span"),
        ("sed_backend", "fake"),
    ],
)
def test_rejects_mislabeled_report(evidence, tmp_path, field, value):
    stem, prediction, report, mapping, config = evidence
    report = copy.deepcopy(report)
    report[field] = value
    output = tmp_path / "bad.png"
    with pytest.raises(ValueError, match="match"):
        render_stem(stem, prediction, report, mapping, config, output)
    assert not output.exists()


def test_rejects_changed_config(evidence, tmp_path):
    stem, prediction, report, mapping, config = evidence
    with pytest.raises(ValueError, match="config is stale"):
        render_stem(
            stem,
            prediction,
            report,
            mapping,
            replace(config, onset_tolerance_seconds=0.9),
            tmp_path / "bad.png",
        )


def test_rejects_changed_mapping(evidence, tmp_path):
    stem, prediction, report, mapping, config = evidence
    mapping.provenance = {**mapping.provenance, "sha256": "a" * 64}
    with pytest.raises(ValueError, match="mapping provenance"):
        render_stem(stem, prediction, report, mapping, config, tmp_path / "bad.png")


def test_rejects_changed_prediction_arrays(evidence, tmp_path):
    stem, prediction, report, mapping, config = evidence
    prediction.probabilities[0, 0] = 0.85
    with pytest.raises(ValueError, match="arrays"):
        render_stem(stem, prediction, report, mapping, config, tmp_path / "bad.png")


def test_rejects_changed_prediction_file(evidence, tmp_path):
    stem, prediction, report, mapping, config = evidence
    cache = Path(report["provenance"]["prediction"]["path"])
    cache.write_bytes(cache.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="cache digest"):
        render_stem(stem, prediction, report, mapping, config, tmp_path / "bad.png")


def test_rejects_changed_audio(evidence, tmp_path):
    stem, prediction, report, mapping, config = evidence
    sf.write(stem.audio_path, np.ones(44101), 44100)
    with pytest.raises(ValueError, match="audio changed"):
        render_stem(stem, prediction, report, mapping, config, tmp_path / "bad.png")


def test_in_memory_report_cannot_bind_raw_arrays(evidence, tmp_path):
    stem, prediction, _, mapping, config = evidence
    report = evaluate_stem(stem, prediction, mapping, config)
    item = render_stem(stem, prediction, report, mapping, config, tmp_path / "memory.png")
    assert item["target_curve_status"] == "unavailable"
    assert item["target_curve_reason"] == "missing_prediction_cache_digest"


def test_missing_cache_keeps_waveform_and_unavailable_curves(evidence, tmp_path):
    stem, _, _, mapping, config = evidence
    report = evaluate_stem(stem, None, mapping, config)
    item = render_stem(stem, None, report, mapping, config, tmp_path / "missing.png")
    assert item["waveform"]["status"] == "available"
    assert item["target_curve_status"] == "unavailable"


def test_missing_audio_can_use_verified_manifest_for_curve(evidence, tmp_path):
    stem, prediction, old, mapping, config = evidence
    stem.audio_path.unlink()
    report = evaluate_stem(
        stem, prediction, mapping, config, prediction_path=old["provenance"]["prediction"]["path"]
    )
    item = render_stem(stem, prediction, report, mapping, config, tmp_path / "missing_audio.png")
    assert item["waveform"]["reason"] == "missing_audio"
    assert item["target_curve_status"] == "available"


def test_unreadable_waveform_does_not_invent_samples(tmp_path):
    path = tmp_path / "invalid.wav"
    path.write_bytes(b"not audio")
    metadata, envelope = _waveform(path, 10)
    assert metadata["reason"] == "unreadable_audio"
    assert envelope is None


def test_unsupported_mapping_has_no_target_lines(evidence):
    stem, prediction, old, mapping, config = evidence
    stem = replace(stem, source_description="unmapped")
    report = evaluate_stem(
        stem, prediction, mapping, config, prediction_path=old["provenance"]["prediction"]["path"]
    )
    status, reason = _validate_report(stem, prediction, report, mapping, config)
    assert status == "unavailable"
    waveform, envelope = _waveform(stem.audio_path, 10)
    figure = _build_figure(
        stem, prediction, report, mapping, config, waveform, envelope, status, reason
    )
    assert len(figure.axes[2].lines) == 0
    assert len(figure.axes[3].collections) == 0
    figure.clear()


@pytest.mark.parametrize("source_kind", ["audio", "cache", "mapping", "source_media", "explicit"])
def test_output_hardlink_cannot_overwrite_inputs(evidence, tmp_path, source_kind):
    stem, prediction, report, mapping, config = evidence
    other = tmp_path / "source.svg"
    other.write_text("preserve", encoding="utf-8")
    protected = []
    sources = {
        "audio": stem.audio_path,
        "cache": Path(report["provenance"]["prediction"]["path"]),
        "mapping": Path(mapping.provenance["path"]),
        "source_media": other,
        "explicit": other,
    }
    if source_kind == "source_media":
        stem = replace(stem, provenance={**stem.provenance, "source_media_paths": [str(other)]})
        report = evaluate_stem(stem, prediction, mapping, config, prediction_path=sources["cache"])
    if source_kind == "explicit":
        protected = [other]
    output = tmp_path / "alias.svg"
    output.hardlink_to(sources[source_kind])
    before = sources[source_kind].read_bytes()
    with pytest.raises(ValueError, match="overwrite input"):
        render_stem(stem, prediction, report, mapping, config, output, protected_inputs=protected)
    assert sources[source_kind].read_bytes() == before


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_waveform_bin_budget(evidence, tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        render_stem(*evidence, tmp_path / "bad.png", max_waveform_points=value)


def test_requires_supported_export_suffix(evidence, tmp_path):
    with pytest.raises(ValueError, match=".png or .svg"):
        render_stem(*evidence, tmp_path / "bad.pdf")


def test_empty_external_reference_label_is_not_called_missing(evidence):
    class EmptyReference:
        def resolve(self, stem, *, allow_ambiguous=False):
            return TemporalSupport("external_annotation", "annotated", (), True, None, {})

    stem, prediction, old, mapping, config = evidence
    report = evaluate_stem(
        stem,
        prediction,
        mapping,
        config,
        reference=EmptyReference(),
        prediction_path=old["provenance"]["prediction"]["path"],
    )
    waveform, envelope = _waveform(stem.audio_path, 10)
    figure = _build_figure(
        stem, prediction, report, mapping, config, waveform, envelope, "available"
    )
    assert figure.axes[1].texts[0].get_text() == "Explicit empty reference"
    figure.clear()


def test_long_headers_fit_export_and_remain_in_their_rows(evidence):
    stem, prediction, report, mapping, config = evidence
    stem = replace(stem, source_description="W" * 2000, stem_id="W" * 2000)
    report = copy.deepcopy(report)
    report["reason_codes"] = ["W" * 2000]
    waveform, envelope = _waveform(stem.audio_path, 10)
    figure = _build_figure(
        stem, prediction, report, mapping, config, waveform, envelope, "unavailable"
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    headers = figure.texts[:-1]
    extents = [artist.get_window_extent(renderer) for artist in headers]
    assert all(bounds.x1 <= figure.bbox.width * 0.98 + 1 for bounds in extents)
    assert all(first.y0 > second.y1 for first, second in zip(extents, extents[1:]))
    assert extents[-1].y0 > figure.axes[0].get_window_extent(renderer).y1
    figure.clear()


def test_relative_prediction_audio_path_protected_from_both_bases(evidence, tmp_path, monkeypatch):
    stem, prediction, _, mapping, config = evidence
    working = tmp_path / "working"
    working.mkdir()
    cwd_source = working / "protected.svg"
    cwd_source.write_text("cwd source", encoding="utf-8")
    cache_source = tmp_path / "protected.svg"
    cache_source.write_text("cache source", encoding="utf-8")
    monkeypatch.chdir(working)
    prediction.metadata["audio_path"] = "protected.svg"
    cache = tmp_path / "raw.npz"
    prediction.save(cache)
    report = evaluate_stem(stem, prediction, mapping, config, prediction_path=cache)
    for path in (cwd_source, cache_source):
        with pytest.raises(ValueError, match="overwrite input"):
            render_stem(stem, prediction, report, mapping, config, path)


def test_empty_audio_marks_waveform_unavailable(tmp_path):
    audio = tmp_path / "empty.wav"
    sf.write(audio, np.empty(0, dtype=np.float32), 44100)
    metadata, envelope = _waveform(audio, 10)
    assert metadata["reason"] == "unreadable_audio"
    assert envelope is None
