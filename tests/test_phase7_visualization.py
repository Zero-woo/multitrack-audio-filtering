"""Configured plot decisions remain bound to the evidence and explicit policy."""

import copy
import json

import pytest
from test_phase5_cli import files as files
from test_phase5_cli import (
    load_json,
    render_manifest,
    requires_plotting,
    stem_key,
    visual_args,
    write_json,
)

from waves_sed.cli import main
from waves_sed.decision import FilterConfig
from waves_sed.evaluation import evaluate_stem
from waves_sed.evaluation_config import EvaluationConfig
from waves_sed.mapping import ManualMapping
from waves_sed.metadata import load_stems
from waves_sed.visualization import _build_figure, _decision_lines, _waveform, render_stem


def policy(files, *, minimum=None):
    return write_json(
        files.manifest.parent / "filter.json",
        {
            "schema_version": 1,
            "filter": {
                "onset": {
                    "min_event_recall": minimum
                    if minimum is not None
                    else {"pass": 0.9, "fail": 0.3},
                    "max_mean_onset_error_ms": {"pass": 30, "fail": 100},
                }
            },
        },
    )


def evidence(files, path):
    stem = load_stems(files.manifest)[0]
    mapping = ManualMapping.load(files.mapping)
    config = EvaluationConfig.load(files.config)
    filter_config = FilterConfig.load(path)
    report = evaluate_stem(
        stem,
        files.prediction,
        mapping,
        config,
        prediction_path=files.cache,
        filter_config=filter_config,
    )
    return stem, mapping, config, filter_config, report


@pytest.mark.parametrize(
    "decision,minimum,second_event",
    [
        ("PASS", {"pass": 0.9, "fail": 0.3}, False),
        ("REVIEW", {"pass": 0.9, "fail": 0.3}, True),
        ("FAIL", {"pass": 0.9, "fail": 0.75}, True),
    ],
)
@requires_plotting
def test_cli_displays_configured_decision_and_every_condition(
    files, decision, minimum, second_event
):
    if second_event:
        manifest = load_json(files.manifest)
        manifest["stems"][0]["expected_intervals"] = [[0.1, 0.2], [0.3, 0.4]]
        write_json(files.manifest, manifest)
    path = policy(files, minimum=minimum)
    original = files.cache.read_bytes()
    assert main(visual_args(files, "--filter-config", str(path))) == 0
    result = render_manifest(files)
    row = result["stems"][0]
    assert row["evaluation"]["decision"] == row["visualization"]["decision"] == decision
    assert result["decision"] is None
    assert result["provenance"]["filter_config"] == FilterConfig.load(path).to_dict()
    gallery = (files.output / "index.html").read_text(encoding="utf-8")
    assert f"Temporal consistency decision: {decision}" in gallery
    assert "min_event_recall" in gallery and "max_mean_onset_error_ms" in gallery
    assert "PASS &gt;= 0.9" in gallery
    assert "No quality decision is assigned" not in gallery
    assert "no aggregate decision is assigned" in gallery
    assert files.cache.read_bytes() == original


@requires_plotting
def test_plot_includes_all_checked_policy_conditions_without_truncation(files):
    stem, mapping, config, filter_config, report = evidence(files, policy(files))
    waveform, envelope = _waveform(stem.audio_path, 16)
    figure = _build_figure(
        stem, files.prediction, report, mapping, config, waveform, envelope, "available"
    )
    try:
        text = "\n".join(artist.get_text() for artist in figure.texts)
        assert "Temporal consistency decision: PASS" in text
        assert "min_event_recall" in text
        assert "max_mean_onset_error_ms" in text
        assert "PASS >= 0.9, FAIL < 0.3" in text
        assert "PASS <= 30, FAIL > 100" in text
        assert figure.get_figheight() > 9
    finally:
        figure.clear()
    result = render_stem(
        stem,
        files.prediction,
        report,
        mapping,
        config,
        files.output / "policy.svg",
        filter_config=filter_config,
    )
    assert result["decision"] == "PASS"


@pytest.mark.parametrize(
    "problem",
    [
        "decision",
        "audit",
        "missing_audit",
        "missing_config",
        "changed_policy",
        "config_provenance",
        "report_config",
        "evidence",
    ],
)
def test_renderer_rejects_forged_or_stale_decision_before_writing(files, problem):
    path = policy(files)
    stem, mapping, config, filter_config, report = evidence(files, path)
    report = copy.deepcopy(report)
    if problem == "decision":
        report["decision"] = "FAIL"
    elif problem == "audit":
        report["decision_details"]["checks"][0]["value"] = 0.1
    elif problem == "missing_audit":
        del report["decision_details"]
    elif problem == "missing_config":
        filter_config = None
    elif problem == "changed_policy":
        filter_config = FilterConfig.load(policy(files, minimum=0.5))
    elif problem == "config_provenance":
        report["provenance"]["filter_config_provenance"]["sha256"] = "0" * 64
    elif problem == "report_config":
        report["provenance"]["filter_config"]["filter"]["onset"]["min_event_recall"] = None
    else:
        report["metrics"]["event_recall"] = 0.1
    output = files.output / "stale.png"
    with pytest.raises(ValueError, match="decision|filter config"):
        render_stem(
            stem,
            files.prediction,
            report,
            mapping,
            config,
            output,
            filter_config=filter_config,
        )
    assert not output.exists()


def test_assigned_decision_without_audit_or_config_is_rejected(files):
    stem = load_stems(files.manifest)[0]
    mapping = ManualMapping.load(files.mapping)
    config = EvaluationConfig.load(files.config)
    report = evaluate_stem(stem, files.prediction, mapping, config, prediction_path=files.cache)
    report["decision"] = "PASS"
    with pytest.raises(ValueError, match="requires its filter config and audit"):
        render_stem(stem, files.prediction, report, mapping, config, files.output / "forged.png")
    assert not files.output.exists()


@requires_plotting
def test_disabled_policy_and_absent_policy_do_not_claim_a_decision(files):
    path = write_json(files.manifest.parent / "disabled.json", {"schema_version": 1, "filter": {}})
    assert main(visual_args(files, "--filter-config", str(path))) == 0
    report = render_manifest(files)["stems"][0]["evaluation"]
    assert report["decision"] is None
    assert "Temporal consistency decision: unassigned" in _decision_lines(report)[0]
    assert report["decision_details"]["status"] == "disabled"
    assert main(visual_args(files)) == 0
    report = render_manifest(files)["stems"][0]["evaluation"]
    assert "decision_details" not in report
    assert _decision_lines(report) == []
    assert "No quality decision is assigned" in (files.output / "index.html").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("target", ["index.html", "visualization-index.json", "plot"])
def test_cli_protects_filter_input_before_any_output_in_selected_workflow(files, target, capsys):
    path = policy(files)
    relative = f"plots/{stem_key(files.stem.stem_id)}.png" if target == "plot" else target
    output = files.output / relative
    output.parent.mkdir(parents=True)
    try:
        output.hardlink_to(path)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = path.read_bytes()
    assert (
        main(visual_args(files, "--filter-config", str(path), "--stem-id", files.stem.stem_id)) == 1
    )
    assert "must not overwrite" in capsys.readouterr().err
    assert path.read_bytes() == output.read_bytes() == original
    assert {item for item in files.output.rglob("*") if item.is_file()} == {output}


def test_renderer_protects_config_even_if_external_protection_is_omitted(files):
    path = policy(files)
    stem, mapping, config, filter_config, report = evidence(files, path)
    output = files.manifest.parent / "alias.png"
    try:
        output.hardlink_to(path)
    except OSError as error:
        pytest.skip(f"Hard links unavailable: {error}")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        render_stem(
            stem, files.prediction, report, mapping, config, output, filter_config=filter_config
        )
    assert output.read_bytes() == path.read_bytes() == original


@pytest.mark.parametrize(
    "contents",
    [
        "{}",
        '{"schema_version":1,"filter":{"unknown":{}}}',
        '{"schema_version":1,"filter":{},"filter":{}}',
    ],
)
def test_cli_rejects_invalid_policy_before_outputs(files, contents):
    path = files.manifest.parent / "invalid-filter.json"
    path.write_text(contents, encoding="utf-8")
    assert main(visual_args(files, "--filter-config", str(path))) == 1
    assert not files.output.exists()


def test_policy_descriptions_are_escaped_in_gallery(files):
    from waves_sed.phase5 import _gallery

    _, _, _, _, report = evidence(files, policy(files))
    # The gallery escapes even unexpected metadata; public rendering separately validates audits.
    report["decision_details"]["reason_codes"] = ['<script>alert("decision")</script>']
    html = _gallery(
        [{"stem_id": "stem", "status": "failed", "error": "none", "evaluation": report}]
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    json.dumps(report, allow_nan=False)


def test_policy_changed_after_load_fails_before_first_plot(files, monkeypatch):
    from waves_sed import phase5

    path = policy(files)
    original_preflight = phase5.preflight_outputs

    def change_policy(outputs, protected):
        original_preflight(outputs, protected)
        policy(files, minimum=0.7)

    monkeypatch.setattr(phase5, "preflight_outputs", change_policy)
    assert main(visual_args(files, "--filter-config", str(path))) == 1
    assert not files.output.exists()


@requires_plotting
def test_policy_changed_during_render_does_not_publish_gallery_or_index(files, monkeypatch):
    from waves_sed import visualization

    path = policy(files)
    original_render = visualization.render_stem

    def change_policy(*args, **kwargs):
        result = original_render(*args, **kwargs)
        policy(files, minimum=0.7)
        return result

    monkeypatch.setattr(visualization, "render_stem", change_policy)
    assert main(visual_args(files, "--filter-config", str(path))) == 1
    assert not (files.output / "index.html").exists()
    assert not (files.output / "visualization-index.json").exists()


def test_renderer_rejects_invalid_policy_type_before_plotting(files):
    stem, mapping, config, _, report = evidence(files, policy(files))
    with pytest.raises(ValueError, match="filter_config must"):
        render_stem(
            stem,
            files.prediction,
            report,
            mapping,
            config,
            files.output / "invalid.svg",
            filter_config={},
        )
    assert not files.output.exists()


@requires_plotting
def test_unsupported_mapping_keeps_explicit_unsupported_decision(files):
    path = policy(files)
    write_json(files.mapping, {"schema_version": 1, "mappings": {}})
    assert main(visual_args(files, "--filter-config", str(path))) == 0
    row = render_manifest(files)["stems"][0]
    assert row["evaluation"]["decision"] == row["visualization"]["decision"] == "UNSUPPORTED"
    assert "Temporal consistency decision: UNSUPPORTED" in (files.output / "index.html").read_text(
        encoding="utf-8"
    )


def test_display_preserves_threshold_boundary_precision(files):
    _, _, _, _, report = evidence(files, policy(files))
    report["decision_details"]["checks"][0]["value"] = 0.89999999
    assert "event_recall=0.89999999" in "\n".join(_decision_lines(report))
