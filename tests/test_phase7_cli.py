"""Optional decisions preserve cached metrics and require explicit policy evidence."""

import csv
import json
import os
import subprocess
import sys
from copy import deepcopy

import pytest
from test_phase4_cli import cached_stem as cached_stem
from test_phase4_cli import dataset, evaluate_args, write_json

from waves_sed.cli import main
from waves_sed.provenance import sha256


def policy(files, rule=None, *, role="onset"):
    return write_json(
        files.output.parent / "filter.json",
        {"schema_version": 1, "filter": {role: rule or {"min_event_recall": 1.0}}},
    )


def test_explicit_policy_does_not_change_metrics_events_or_cache(cached_stem):
    files = cached_stem
    original = files.cache.read_bytes()
    assert main(evaluate_args(files)) == 0
    baseline = dataset(files)["stems"][0]
    config = policy(files)
    assert main(evaluate_args(files, "--filter-config", str(config))) == 0
    row = dataset(files)["stems"][0]
    assert row["decision"] == "PASS"
    assert row["decision_details"]["status"] == "decided"
    assert row["decision_details"]["config_provenance"] == {
        "path": str(config.resolve()),
        "sha256": sha256(config),
    }
    stripped = deepcopy(row)
    stripped["decision"] = None
    del stripped["decision_details"]
    del stripped["provenance"]["filter_config"]
    del stripped["provenance"]["filter_config_provenance"]
    assert stripped == baseline
    assert files.cache.read_bytes() == original
    with (files.output / "stems.csv").open(encoding="utf-8", newline="") as stream:
        assert next(csv.DictReader(stream))["decision"] == "PASS"


@pytest.mark.parametrize(
    "problem", ["cache", "identity", "reference", "ambiguous", "outside", "mapping", "role"]
)
def test_unusable_evidence_has_review_or_unsupported_policy(cached_stem, problem):
    files = cached_stem
    config = policy(files)
    manifest = json.loads(files.manifest.read_text())
    if problem == "cache":
        write_json(files.index, {"schema_version": 1, "predictions": {}})
    elif problem == "identity":
        files.prediction.metadata["audio_sha256"] = "0" * 64
        files.prediction.save(files.cache)
    elif problem == "mapping":
        write_json(files.mapping, {"schema_version": 1, "mappings": {}})
    elif problem == "role":
        manifest["stems"][0]["role"] = "unsupported-role"
    elif problem == "outside":
        manifest["stems"][0]["expected_intervals"] = [[0.1, 0.5]]
    elif problem == "ambiguous":
        manifest["stems"][0]["provenance"]["reference_status"] = "ambiguous"
        evaluation = json.loads(files.config.read_text())
        evaluation["allow_ambiguous_reference"] = True
        write_json(files.config, evaluation)
    else:
        manifest["stems"][0]["expected_intervals"] = None
        manifest["stems"][0]["provenance"]["reference_status"] = "missing"
    write_json(files.manifest, manifest)
    assert main(evaluate_args(files, "--filter-config", str(config))) == 0
    row = dataset(files)["stems"][0]
    assert row["decision"] == ("UNSUPPORTED" if problem in {"mapping", "role"} else "REVIEW")
    assert row["decision_details"]["reason_codes"]
    if problem == "ambiguous":
        assert row["evaluation_status"] == "evaluated"
        assert row["metrics"]["event_recall"] == 1


@pytest.mark.parametrize("configured", ["disabled", "other_role"])
def test_no_enabled_role_rule_never_assigns_a_decision(cached_stem, configured):
    files = cached_stem
    config = (
        policy(files, {"min_event_recall": None})
        if configured == "disabled"
        else policy(files, {"min_tiou": 0.8}, role="span")
    )
    assert main(evaluate_args(files, "--filter-config", str(config))) == 0
    row = dataset(files)["stems"][0]
    assert row["decision"] is None
    assert row["decision_details"]["status"] == (
        "disabled" if configured == "disabled" else "not_configured"
    )


@pytest.mark.parametrize("missing_error_only", [True, False])
def test_no_detections_does_not_turn_undefined_error_into_a_pass(cached_stem, missing_error_only):
    files = cached_stem
    evaluation = json.loads(files.config.read_text())
    evaluation["event"]["threshold"] = 0.99
    write_json(files.config, evaluation)
    rules = {"max_mean_onset_error_ms": 50}
    if not missing_error_only:
        rules["min_event_recall"] = 1
    config = policy(files, rules)
    assert main(evaluate_args(files, "--filter-config", str(config))) == 0
    row = dataset(files)["stems"][0]
    assert row["metrics"]["mean_onset_error_ms"] is None
    assert row["decision"] == ("REVIEW" if missing_error_only else "FAIL")


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ({"pass": 0.9, "fail": 0.7}, "PASS"),
        ({"pass": 0.99, "fail": 0.9}, "REVIEW"),
        ({"pass": 1.0, "fail": 0.99}, "FAIL"),
    ],
)
def test_review_bands_reuse_same_cache(cached_stem, bounds, expected):
    files = cached_stem
    manifest = json.loads(files.manifest.read_text())
    manifest["stems"][0].update(role="span", planned_role="span", expected_intervals=[[0.09, 0.31]])
    write_json(files.manifest, manifest)
    before = files.cache.read_bytes()
    config = policy(files, {"min_tiou": bounds}, role="span")
    assert main(evaluate_args(files, "--filter-config", str(config))) == 0
    row = dataset(files)["stems"][0]
    assert row["metrics"]["temporal_iou"] == pytest.approx(0.2 / 0.22)
    assert row["decision"] == expected
    assert files.cache.read_bytes() == before


def test_invalid_filter_config_does_not_write_reports(cached_stem):
    files = cached_stem
    config = policy(files)
    config.write_text('{"schema_version":1,"filter":{"onset":{"min_event_recall":NaN}}}')
    assert main(evaluate_args(files, "--filter-config", str(config))) == 1
    assert not files.output.exists()


@pytest.mark.parametrize("name", ["dataset.json", "stems.csv", "clips.csv"])
def test_policy_input_and_hardlink_are_preflight_protected(cached_stem, name):
    files = cached_stem
    config = policy(files)
    original = config.read_bytes()
    files.output.mkdir()
    target = files.output / name
    target.hardlink_to(config)
    assert main(evaluate_args(files, "--filter-config", str(config))) == 1
    assert target.read_bytes() == config.read_bytes() == original
    assert list(files.output.iterdir()) == [target]


def test_optional_decisions_run_without_inference_audio_or_plot_libraries(cached_stem):
    files = cached_stem
    config = policy(files)
    code = """
import builtins,json,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','torchaudio','librosa','soundfile','matplotlib'}:
        raise AssertionError('unexpected optional import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from waves_sed.cli import main
raise SystemExit(main(json.loads(sys.argv[1])))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            json.dumps(evaluate_args(files, "--filter-config", str(config))),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stderr
    assert dataset(files)["stems"][0]["decision"] == "PASS"


def test_filter_change_during_evaluation_is_detected_before_report_writes(cached_stem, monkeypatch):
    from waves_sed import phase4

    files = cached_stem
    config = policy(files)
    original = phase4.evaluate_stem

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        config.write_text(config.read_text() + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(phase4, "evaluate_stem", changed)
    assert main(evaluate_args(files, "--filter-config", str(config))) == 1
    assert not files.output.exists()
