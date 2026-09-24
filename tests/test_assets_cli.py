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
        assert Path(config["mmproj_path"]).is_relative_to(tmp_path / ".models")
        assert len(config["model_sha256"]) == 64
        assert len(config["mmproj_sha256"]) == 64
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
    output = capsys.readouterr().out
    hint = output.split("To open Playground: ")[1].strip()
    assert shlex.split(hint) == [
        "uv", "run", "gemmajev", "setup", "--model", "e4b", "--vision",
        "--workspace", str(workspace),
    ]
    assert "gemmajev demo" not in output


def test_vision_runtime_uses_separate_path_and_explicit_build_mode(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "check_platform", dict)
    monkeypatch.setattr(assets.subprocess, "run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("GEMMAJEV_VISION", "1")
    assert assets.build_runtime(tmp_path) == assets.runtime_path(tmp_path)
    assert calls[-1]["env"]["GEMMAJEV_VISION"] == "0"
    assert assets.build_runtime(tmp_path, vision=True) == (
        tmp_path / ".runtime/build/bin/gemmajev-vision-worker"
    )
    assert calls[-1]["env"]["GEMMAJEV_VISION"] == "1"


def test_vision_setup_downloads_matching_projector_only_when_requested(tmp_path, monkeypatch):
    calls = []

    def download(url, path, **kwargs):
        calls.append((url, path, kwargs))
        return Path(path)

    monkeypatch.setattr(assets, "download_file", download)
    config = assets.model_config("e4b", tmp_path)
    assert assets.prepare_model("e4b", tmp_path) == Path(config["model_path"])
    assert len(calls) == 1
    assert assets.prepare_model("e4b", tmp_path, vision=True) == Path(config["model_path"])
    assert len(calls) == 3
    url, path, options = calls[-1]
    assert config["revision"] in url and url.endswith(config["mmproj_file"])
    assert path == config["mmproj_path"]
    assert options == {"size": config["mmproj_size"], "sha256": config["mmproj_sha256"]}


@pytest.mark.parametrize("flag", ["--vision", "--audio"])
def test_cli_vision_setup_prepares_both_workers_and_suggests_playground(
    tmp_path, monkeypatch, capsys, flag,
):
    import shlex

    calls = []
    monkeypatch.setattr(cli, "check_platform", lambda: {"chip": "test", "memory_bytes": 16 * 2**30})
    monkeypatch.setattr(cli, "build_runtime", lambda root, **kwargs: calls.append(("build", kwargs)))
    monkeypatch.setattr(
        cli, "prepare_model", lambda model, root, **kwargs: calls.append((model, kwargs))
    )
    assert cli.main(["setup", flag, "--model", "e4b", "--workspace", str(tmp_path)]) == 0
    assert calls == [("build", {}), ("build", {"vision": True}), ("e4b", {"vision": True})]
    output = capsys.readouterr().out
    assert shlex.split(output.split("Ready. Run: ")[1].strip()) == [
        "uv", "run", "gemmajev", "demo", "--model", "e4b", "--workspace", str(tmp_path),
    ]


@pytest.mark.parametrize("kind", ["image", "audio"])
def test_cli_media_decision_preserves_request_and_enables_runtime(tmp_path, monkeypatch, capsys, kind):
    from gemmajev import engine

    request = {
        "task": "Identify the attached media.", "state": "A media file is attached.",
        "query": "Which object?", "options": {"cat": "A cat", "dog": "A dog"},
    }
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request))
    attachment = tmp_path / ("example.png" if kind == "image" else "example.wav")
    calls = []

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def close(self):
            calls.append("closed")

        def decide(self, value, **kwargs):
            calls.append((value, kwargs))
            return {"cat": 0.8, "dog": 0.2}

    monkeypatch.setattr(engine, "GemmaJev", FakeModel)
    assert cli.main([
        "decide", str(source), "--model", "e4b", f"--{kind}", str(attachment),
        "--image-tokens", "140",
    ]) == 0
    assert calls == [
        {"model": "e4b", "workspace": None,
         **({"vision": True, "image_tokens": 140} if kind == "image" else {"audio": True})},
        (request, {f"{kind}_path": attachment}), "closed",
    ]
    assert json.loads(capsys.readouterr().out) == {"cat": 0.8, "dog": 0.2}


def test_cli_rejects_two_attachments_before_loading(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["decide", "missing.json", "--image", "image.png", "--audio", "audio.wav"])
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
