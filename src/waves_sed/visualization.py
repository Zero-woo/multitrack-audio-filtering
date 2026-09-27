"""Headless diagnostic exports from cached evidence, with native-rate waveforms.

Plotting and audio decoding are optional dependencies, imported only on rendering.
The evaluation report remains authoritative: this module never reruns metrics.
"""

import textwrap
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from waves_sed.evaluation import validate_cache_identity
from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.events import smooth_target
from waves_sed.mapping import ManualMapping, aggregate_target
from waves_sed.metadata import StemMetadata
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import sha256
from waves_sed.reporting import preflight_outputs
from waves_sed.temporal_metrics import validate_intervals


def _envelope_from_reader(reader, max_points: int) -> dict:
    """Stream every sample; each bin retains extrema across every audio channel.

    Unlike stride sampling or averaging, impulses and opposite-phase stereo remain
    visible. Bin boundaries are integer sample indices at the original sample rate.
    Reads are bounded to approximately 65536 sample values, independent of duration.
    """
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_waveform_points must be a positive integer")
    frames, sample_rate, channels = reader.frames, reader.samplerate, reader.channels
    if frames < 1 or sample_rate < 1 or channels < 1:
        raise ValueError("Audio must contain samples at a positive native sample rate")
    bin_count = min(frames, max_points)
    edges = np.linspace(0, frames, bin_count + 1, dtype=np.int64)
    minima = np.full(bin_count, np.inf)
    maxima = np.full(bin_count, -np.inf)
    cursor = 0
    block_size = max(1, 65536 // channels)
    while cursor < frames:
        samples = reader.read(
            frames=min(block_size, frames - cursor), dtype="float32", always_2d=True
        )
        if not len(samples):
            raise ValueError("Audio ended before its declared sample count")
        if not np.isfinite(samples).all():
            raise ValueError("Audio contains nonfinite samples")
        indices = np.searchsorted(edges[1:], np.arange(cursor, cursor + len(samples)), side="right")
        np.minimum.at(minima, indices, samples.min(axis=1))
        np.maximum.at(maxima, indices, samples.max(axis=1))
        cursor += len(samples)
    return {
        "start_seconds": edges[:-1].astype(np.float64) / sample_rate,
        "end_seconds": edges[1:].astype(np.float64) / sample_rate,
        "minimum": minima,
        "maximum": maxima,
        "native_sample_rate": sample_rate,
        "channels": channels,
        "sample_count": frames,
        "duration_seconds": frames / sample_rate,
        "envelope_bin_count": bin_count,
        "method": "streaming_min_max_all_channels_integer_sample_bins",
    }


def _waveform(path: Path | None, max_points: int) -> tuple[dict, dict | None]:
    if path is None or not Path(path).is_file():
        return {"status": "unavailable", "reason": "missing_audio"}, None
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("Waveform rendering requires pip install '.[visualization]'") from error
    try:
        digest = sha256(path)
        with sf.SoundFile(path) as reader:
            envelope = _envelope_from_reader(reader, max_points)
    except (OSError, RuntimeError, ValueError) as error:
        return {"status": "unavailable", "reason": "unreadable_audio", "error": str(error)}, None
    if sha256(path) != digest:
        raise ValueError("Source audio changed during waveform rendering")
    metadata = {
        key: value
        for key, value in envelope.items()
        if key not in {"start_seconds", "end_seconds", "minimum", "maximum"}
    }
    return {"status": "available", "audio_sha256": digest, **metadata}, envelope


def _provenance_paths(value) -> list[Path]:
    paths = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path" and isinstance(item, str):
                paths.append(Path(item))
            elif key == "source_media_paths" and isinstance(item, list):
                paths.extend(Path(path) for path in item if isinstance(path, str))
            else:
                paths.extend(_provenance_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_provenance_paths(item))
    return paths


def _validate_report(stem, prediction, report, mapping, config) -> tuple[str, str | None]:
    """Bind report, current source bytes, and rendered raw arrays before plotting."""
    config.validate()
    if report.get("schema_version") != 1:
        raise ValueError("Visualization requires evaluation report schema_version 1")
    for name in ("stem_id", "candidate_id", "clip_key", "source_description", "role"):
        if report.get(name) != getattr(stem, name):
            raise ValueError(f"Visualization report {name} does not match the stem")
    provenance = report.get("provenance") or {}
    if provenance.get("stem") != stem.to_dict():
        raise ValueError("Visualization report stem provenance is stale")
    if provenance.get("evaluation_config") != config.to_dict():
        raise ValueError("Visualization report evaluation config is stale")
    if provenance.get("mapping") != mapping.provenance:
        raise ValueError("Visualization report mapping provenance is stale")
    reference = report.get("reference") or {}
    if report.get("expected_intervals") != reference.get("intervals"):
        raise ValueError("Visualization report reference intervals are inconsistent")
    validate_intervals(report.get("expected_intervals") or ())
    if report.get("detected_intervals") is not None:
        validate_intervals(report["detected_intervals"])
    prediction_provenance = provenance.get("prediction")
    if prediction is None:
        if report.get("detection_status") == "available":
            return "unavailable", "missing_prediction"
        return "unavailable", "missing_prediction"
    prediction.validate()
    if not isinstance(prediction_provenance, dict):
        raise ValueError("Visualization prediction is absent from report provenance")
    checks = {
        "metadata": prediction.metadata,
        "class_ids": list(prediction.class_ids),
        "class_names": list(prediction.class_names),
        "frame_count": len(prediction.probabilities),
    }
    if any(prediction_provenance.get(key) != value for key, value in checks.items()):
        raise ValueError("Visualization prediction metadata does not match the report")
    if report.get("sed_backend") != prediction.metadata.get("backend"):
        raise ValueError("Visualization report backend does not match prediction")
    resolution = mapping.resolve(stem.source_description, prediction.class_ids)
    if report.get("mapping") != resolution.to_dict():
        raise ValueError("Visualization report mapping resolution is stale")
    path, digest = prediction_provenance.get("path"), prediction_provenance.get("sha256")
    if path is not None:
        if not digest or not Path(path).is_file() or sha256(path) != digest:
            raise ValueError("Visualization prediction cache digest does not match the report")
        cached = FramePrediction.load(path)
        if cached.metadata != prediction.metadata or any(
            not np.array_equal(getattr(cached, name), getattr(prediction, name))
            for name in ("probabilities", "frame_start_seconds", "frame_end_seconds")
        ):
            raise ValueError("Visualization prediction arrays do not match the report cache")
        if cached.class_ids != prediction.class_ids or cached.class_names != prediction.class_names:
            raise ValueError("Visualization prediction classes do not match the report cache")
    identity = validate_cache_identity(stem, prediction)
    old_identity = report.get("cache_validation") or {}
    previous_audio = old_identity.get("current_audio_sha256")
    if previous_audio is not None and identity.get("current_audio_sha256") not in {
        None,
        previous_audio,
    }:
        raise ValueError("Source audio changed since the evaluation report")
    if report.get("detection_status") != "available":
        return "unavailable", "report_detection_unavailable"
    if not old_identity.get("verified") or not identity.get("verified"):
        return "unavailable", "unverified_audio_identity"
    if path is None or digest is None:
        return "unavailable", "missing_prediction_cache_digest"
    if resolution.status != "supported":
        return "unavailable", "unsupported_mapping"
    if report.get("detected_intervals") is None:
        raise ValueError("Available report detection requires detected intervals")
    return "available", None


def _step_values(starts, ends, values) -> tuple[np.ndarray, np.ndarray]:
    """Preserve half-open bin edges and insert NaNs at unobserved temporal gaps."""
    x, y = [], []
    for index, value in enumerate(values):
        if index and starts[index] != ends[index - 1]:
            x.append(np.nan)
            y.append(np.nan)
        x.extend((starts[index], ends[index]))
        y.extend((value, value))
    return np.asarray(x), np.asarray(y)


def _metric_text(report: dict) -> str:
    metrics = report.get("metrics")
    if metrics is None:
        return "Metrics unavailable"
    role_keys = {
        "onset": ("event_recall", "event_precision", "mean_onset_error_ms", "matched_event_count"),
        "span": (
            "temporal_iou",
            "expected_coverage",
            "detected_precision",
            "out_of_window_activation",
        ),
        "ambience": (
            "occupancy_in_expected_span",
            "mean_target_confidence",
            "out_of_window_activity",
        ),
    }
    parts = []
    for key in role_keys.get(report.get("role"), ()):
        value = metrics.get(key)
        unit = " s" if key.startswith("out_of_window") else ""
        parts.append(f"{key}={value:.3g}{unit}" if value is not None else f"{key}=unavailable")
    return " | ".join(parts)


def _label(text: str, width: int) -> str:
    """Keep plot headers bounded; complete descriptions remain in report JSON."""
    return textwrap.fill(" ".join(text.split()), width, max_lines=2, placeholder="...")


def _build_figure(
    stem, prediction, report, mapping, config, waveform, envelope, curve_status, curve_reason=None
):
    try:
        from matplotlib import font_manager
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.text import Text
    except ImportError as error:
        raise RuntimeError("Figure export requires pip install '.[visualization]'") from error
    figure = Figure(figsize=(12, 9), dpi=140)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4, 1, height_ratios=(2, 0.7, 2, 0.7), hspace=0.2)
    axes = [figure.add_subplot(grid[0])]
    axes.extend(figure.add_subplot(grid[index], sharex=axes[0]) for index in range(1, 4))
    figure.subplots_adjust(left=0.11, right=0.98, top=0.73, bottom=0.10)
    description = stem.source_description or "Missing source description"
    title = _label(f"{description}  |  {stem.role or 'unknown role'}", 110)
    figure.text(0.11, 0.975, title, va="top", fontsize=13, weight="bold")
    figure.text(
        0.11,
        0.915,
        _label(
            f"Stem: {stem.stem_id} | Backend: {report.get('sed_backend') or 'unavailable'}", 130
        ),
        va="top",
        fontsize=9,
    )
    reference = report.get("reference") or {}
    figure.text(
        0.11,
        0.88,
        _label(
            f"Reference: {reference.get('origin') or 'unavailable'} | status: {reference.get('status') or 'missing'}"
            f" | used for metrics: {reference.get('usable') is True and report.get('evaluation_status') == 'evaluated'}"
            f" | Evaluation: {report.get('evaluation_status')} | Current target curve: {curve_status}",
            135,
        ),
        va="top",
        fontsize=9,
    )
    reasons = ", ".join(report.get("reason_codes", [])) or "none"
    figure.text(0.11, 0.835, _label(f"Reasons: {reasons}", 140), va="top", fontsize=8)
    metric_prefix = (
        "Reported metrics (current curve unavailable): " if curve_status != "available" else ""
    )
    figure.text(
        0.11, 0.785, _label(metric_prefix + _metric_text(report), 145), va="top", fontsize=8
    )
    wave_axis, reference_axis, target_axis, detection_axis = axes
    if envelope is None:
        wave_axis.text(
            0.5,
            0.5,
            f"Waveform unavailable: {waveform['reason']}",
            transform=wave_axis.transAxes,
            ha="center",
        )
        wave_axis.set_ylim(-1, 1)
    else:
        x, low = _step_values(
            envelope["start_seconds"], envelope["end_seconds"], envelope["minimum"]
        )
        _, high = _step_values(
            envelope["start_seconds"], envelope["end_seconds"], envelope["maximum"]
        )
        wave_axis.fill_between(x, low, high, color="#6b7280", linewidth=0.4)
        wave_axis.plot(x, low, color="#4b5563", linewidth=0.35)
        wave_axis.plot(x, high, color="#4b5563", linewidth=0.35)
        wave_axis.text(
            0.01,
            0.95,
            f"{waveform['native_sample_rate']} Hz native | {waveform['channels']} channel(s), min/max",
            transform=wave_axis.transAxes,
            va="top",
            fontsize=8,
        )
    wave_axis.set_ylabel("Amplitude")
    expected = report.get("expected_intervals")
    ambiguous = reference.get("status") == "ambiguous"
    if expected:
        reference_axis.broken_barh(
            [(start, end - start) for start, end in expected],
            (0.15, 0.7),
            facecolors="#E69F00" if ambiguous else "#0072B2",
            alpha=0.55,
            edgecolors="#8c6100" if ambiguous else "#0072B2",
            hatch="///" if ambiguous else None,
        )
        label = "Ambiguous planned support" if ambiguous else "Expected support"
        if not reference.get("usable"):
            label += " (excluded from metrics)"
        reference_axis.text(
            0.01, 0.98, label, transform=reference_axis.transAxes, va="top", fontsize=8
        )
    else:
        empty_label = (
            "Explicit empty reference"
            if reference.get("usable")
            else "Empty planned support (unavailable)"
        )
        reference_axis.text(
            0.5,
            0.5,
            "Expected support unavailable" if expected is None else empty_label,
            transform=reference_axis.transAxes,
            ha="center",
            fontsize=9,
        )
    reference_axis.set_ylabel("Expected")
    if curve_status == "available":
        resolution = mapping.resolve(stem.source_description, prediction.class_ids)
        target = aggregate_target(prediction, resolution)
        x, y = _step_values(prediction.frame_start_seconds, prediction.frame_end_seconds, target)
        target_axis.plot(x, y, color="#0072B2", linewidth=1.2, label="Raw target max")
        if config.event.median_window_frames > 1:
            smoothed = smooth_target(prediction, target, config.event.median_window_frames)
            _, smooth_y = _step_values(
                prediction.frame_start_seconds, prediction.frame_end_seconds, smoothed
            )
            target_axis.plot(
                x,
                smooth_y,
                color="#D55E00",
                linewidth=1.2,
                label=f"Median ({config.event.median_window_frames} frames)",
            )
        target_axis.axhline(
            config.event.threshold,
            color="#333333",
            linestyle="--",
            linewidth=0.8,
            label=f"Extraction threshold {config.event.threshold:g}",
        )
        target_axis.legend(loc="upper right", fontsize=8, ncols=3)
        detected = report["detected_intervals"]
        if detected:
            detection_axis.broken_barh(
                [(start, end - start) for start, end in detected],
                (0.15, 0.7),
                facecolors="#009E73",
                alpha=0.65,
            )
        else:
            detection_axis.text(
                0.5,
                0.5,
                "No detected events",
                transform=detection_axis.transAxes,
                ha="center",
                fontsize=9,
            )
    else:
        target_axis.text(
            0.5,
            0.5,
            _label(f"Target curve unavailable: {curve_reason or 'see evaluation reasons'}", 100),
            transform=target_axis.transAxes,
            ha="center",
            fontsize=9,
        )
        detection_axis.text(
            0.5,
            0.5,
            "Detection unavailable",
            transform=detection_axis.transAxes,
            ha="center",
            fontsize=9,
        )
    target_axis.set_ylim(-0.02, 1.08)
    target_axis.set_ylabel("Probability")
    detection_axis.set_ylabel("Detected")
    detection_axis.set_xlabel("Time (seconds)")
    for axis in (reference_axis, detection_axis):
        axis.set_ylim(0, 1.35)
        axis.set_yticks([])
    ends = [0.0]
    if envelope is not None:
        ends.append(envelope["duration_seconds"])
    if prediction is not None:
        ends.append(float(prediction.frame_end_seconds[-1]))
    ends.extend(end for _, end in expected or ())
    ends.extend(end for _, end in report.get("detected_intervals") or ())
    duration = max(ends)
    for axis in axes:
        axis.grid(axis="x", color="#dddddd", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    axes[0].set_xlim(0, duration if duration else 1)
    figure.text(
        0.11,
        0.035,
        "WAVES planned timing measures internal consistency. Extraction settings are not quality decisions.",
        fontsize=8,
        color="#444444",
    )
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = ["DejaVu Sans"]
    for candidate in ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic"):
        if candidate in available_fonts:
            font_family.append(candidate)
            break
    for artist in figure.findobj(Text):
        artist.set_fontfamily(font_family)
        artist.set_parse_math(False)
    # Character-count wrapping alone is insufficient for CJK or unusually wide
    # labels. Fit headers against the actual renderer without growing their rows.
    renderer = figure.canvas.get_renderer()
    available_width = figure.bbox.width * (0.98 - 0.11)
    for artist in figure.texts:
        original = " ".join(artist.get_text().split())
        width = max(1, len(original))
        while True:
            wrapped = _label(original, width)
            maximum = max(
                (
                    renderer.get_text_width_height_descent(
                        line, artist.get_fontproperties(), ismath=False
                    )[0]
                    for line in wrapped.splitlines()
                ),
                default=0,
            )
            if maximum <= available_width or width == 1:
                artist.set_text(wrapped)
                break
            width = max(1, int(width * 0.9))
    return figure


def render_stem(
    stem: StemMetadata,
    prediction: FramePrediction | None,
    report: dict,
    mapping: ManualMapping,
    config: EvaluationConfig,
    output: Path,
    *,
    max_waveform_points: int = 4000,
    protected_inputs: list[Path] | None = None,
) -> dict:
    """Export PNG/SVG diagnostics without inference or changing the report/cache.

    Trusted curves require a report-bound cache path and digest; in-memory-only
    reports render explicit unavailable curves because they do not bind raw arrays.
    Source/report identity mismatches are rejected before any output is written.
    """
    if (
        isinstance(max_waveform_points, bool)
        or not isinstance(max_waveform_points, int)
        or max_waveform_points < 1
    ):
        raise ValueError("max_waveform_points must be a positive integer")
    output = Path(output)
    format_name = output.suffix.lower().lstrip(".")
    if format_name not in {"png", "svg"}:
        raise ValueError("Visualization output must use .png or .svg")
    protected = [Path(path) for path in protected_inputs or []]
    protected.extend(_provenance_paths(stem.provenance))
    protected.extend(_provenance_paths(report.get("provenance", {})))
    protected.extend(_provenance_paths(mapping.provenance))
    if stem.audio_path is not None:
        protected.append(Path(stem.audio_path))
    if prediction is not None and prediction.metadata.get("audio_path"):
        source = Path(prediction.metadata["audio_path"])
        cache = (report.get("provenance", {}).get("prediction") or {}).get("path")
        protected.append(source)
        if not source.is_absolute() and cache is not None:
            protected.append(Path(cache).parent / source)
    preflight_outputs([output], protected)
    curve_status, curve_reason = _validate_report(stem, prediction, report, mapping, config)
    waveform, envelope = _waveform(stem.audio_path, max_waveform_points)
    old_digest = (report.get("cache_validation") or {}).get("current_audio_sha256")
    if waveform.get("audio_sha256") and old_digest and waveform["audio_sha256"] != old_digest:
        raise ValueError("Source audio changed since the evaluation report")
    figure = _build_figure(
        stem, prediction, report, mapping, config, waveform, envelope, curve_status, curve_reason
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        figure.savefig(temporary, format=format_name, dpi=140, facecolor="white")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        figure.clear()
    return {
        "stem_id": stem.stem_id,
        "path": str(output.resolve()),
        "format": format_name,
        "waveform": waveform,
        "target_curve_status": curve_status,
        "target_curve_reason": curve_reason,
        "reference_status": report["reference"].get("status"),
        "reference_origin": report["reference"].get("origin"),
        "evaluation_status": report["evaluation_status"],
        "decision": None,
    }
