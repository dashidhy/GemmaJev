import hashlib
import json
from pathlib import Path

import pytest

from gemmajev import assets, cli


def test_failed_download_preserves_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "weights"
    destination.write_bytes(b"old data")

    def download(args, **kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(b"bad data")

    monkeypatch.setattr(assets.subprocess, "run", download)
    expected = hashlib.sha256(b"new data").hexdigest()
    with pytest.raises(RuntimeError, match="verification"):
        assets.download_file("https://example.test/model", destination, size=8, sha256=expected)
    assert destination.read_bytes() == b"old data"
    assert not list(tmp_path.glob("*.part"))


def test_verified_download_is_reused_and_replacement_is_complete(tmp_path, monkeypatch):
    destination = tmp_path / "weights"
    calls = []

    def download(args, **kwargs):
        calls.append(args)
        assert not destination.exists()
        Path(args[args.index("--output") + 1]).write_bytes(b"verified")

    monkeypatch.setattr(assets.subprocess, "run", download)
    digest = hashlib.sha256(b"verified").hexdigest()
    for _ in range(2):
        assert (
            assets.download_file("https://example.test/model", destination, sha256=digest)
            == destination
        )
    assert len(calls) == 1
    assert destination.read_bytes() == b"verified"


def test_model_manifest_is_independent_of_asset_workspace(tmp_path):
    for name in ("e4b", "12b"):
        config = assets.model_config(name, tmp_path)
        assert Path(config["model_path"]).is_relative_to(tmp_path / ".models")
        assert len(config["model_sha256"]) == 64
    with pytest.raises(ValueError, match="model must"):
        assets.model_config("unsupported", tmp_path)


def test_build_sets_asset_workspace_without_shell_interpolation(tmp_path, monkeypatch):
    calls = []
    workspace = tmp_path / "space and literal $(not-a-command)"
    monkeypatch.setattr(assets, "check_platform", dict)
    monkeypatch.setattr(
        assets.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    assert assets.build_runtime(workspace) == workspace / ".runtime/build/bin/gemmajev-worker"
    assert calls[0][1]["env"]["GEMMAJEV_WORKSPACE"] == str(workspace)
    assert calls[0][0][0] == "bash"
    assert "shell" not in calls[0][1]


def test_invalid_api_json_does_not_construct_engine(tmp_path, monkeypatch, capsys):
    import gemmajev.engine

    def unexpected(**kwargs):
        raise AssertionError("Invalid input must not construct an engine")

    monkeypatch.setattr(gemmajev.engine, "GemmaJev", unexpected)
    request = tmp_path / "request.json"
    request.write_text('{"task":"first","task":"duplicate"}')
    assert cli.main(["decide", str(request)]) == 2
    assert "Duplicate JSON key" in capsys.readouterr().err
    request.write_text(json.dumps({"task": "classify", "options": {"one": "Only one"}}))
    assert cli.main(["decide", str(request)]) == 2


def test_benchmark_failures_are_user_facing(monkeypatch, capsys):
    import gemmajev.benchmarks

    def fail(argv):
        raise ValueError("Prepare the dataset first")

    monkeypatch.setattr(gemmajev.benchmarks, "main", fail)
    assert cli.main(["benchmark", "run"]) == 2
    assert capsys.readouterr().err == "gemmajev: Prepare the dataset first\n"


def test_setup_hint_preserves_selected_model_and_workspace(tmp_path, monkeypatch, capsys):
    import shlex

    workspace = tmp_path / "my models"
    monkeypatch.setattr(cli, "check_platform", lambda: {"chip": "test", "memory_bytes": 16 * 2**30})
    monkeypatch.setattr(cli, "build_runtime", lambda root: root / ".runtime/worker")
    monkeypatch.setattr(cli, "prepare_model", lambda model, root: root / ".models" / model)
    assert cli.main(["setup", "--model", "e4b", "--workspace", str(workspace)]) == 0
    hint = capsys.readouterr().out.split("Ready. Run: ")[1].strip()
    assert shlex.split(hint) == [
        "uv", "run", "gemmajev", "demo", "--model", "e4b", "--workspace", str(workspace)
    ]
