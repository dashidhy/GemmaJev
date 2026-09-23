import json

import pytest

from gemmajev.benchmarks.cli import evaluate


def source(identifier="sample"):
    return {
        "id": identifier,
        "dataset": "authored144",
        "group_id": identifier,
        "primitive": "choice",
        "gold_id": "supported",
        "caller_task": "Read Evidence and answer Question.",
        "state_label": "Evidence",
        "query_label": "Question",
        "state": "All buses stop here.",
        "question": "Do buses stop here?",
        "options": [
            {"id": "supported", "description": "Yes"},
            {"id": "contradicted", "description": "No"},
        ],
    }


class FakeEngine:
    calls = 0
    closed = 0
    fail = False

    def __init__(self, **kwargs):
        pass

    def load(self):
        pass

    def close(self):
        type(self).closed += 1

    def decide(self, request):
        type(self).calls += 1
        if type(self).fail:
            raise RuntimeError("simulated interruption")
        assert set(request) == {"task", "state", "query", "options"}
        return {"supported": 0.8, "contradicted": 0.2}


@pytest.fixture(autouse=True)
def reset_fake():
    FakeEngine.calls = 0
    FakeEngine.closed = 0
    FakeEngine.fail = False


def run(tmp_path, rows):
    return evaluate(
        rows,
        model="e4b",
        workspace=tmp_path,
        output=tmp_path / "run",
        fingerprint={"test": "fixed"},
        engine_factory=FakeEngine,
        cooldown=0,
        batch_size=1,
    )


def test_resume_does_not_repeat_finished_predictions(tmp_path):
    rows = [source("one"), source("two")]
    first = run(tmp_path, rows)
    assert FakeEngine.calls == 2 and FakeEngine.closed == 2
    assert run(tmp_path, rows) == first
    assert FakeEngine.calls == 2
    rows[0]["state"] = "changed input"
    with pytest.raises(ValueError, match="different protocol"):
        run(tmp_path, rows)


def test_failure_closes_worker_and_is_not_reported_as_a_complete_panel(tmp_path):
    FakeEngine.fail = True
    with pytest.raises(RuntimeError, match="interruption"):
        run(tmp_path, [source()])
    assert FakeEngine.closed == 1
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status == {"state": "interrupted", "completed": 0, "planned": 1}
    assert not (tmp_path / "run/metrics.json").exists()
    FakeEngine.fail = False
    assert run(tmp_path, [source()])["e4b"]["authored144"]["n"] == 1


def test_missing_dataset_cli_gives_runnable_prepare_command(tmp_path):
    from gemmajev.benchmarks.cli import main

    workspace = tmp_path / "data cache"
    with pytest.raises(FileNotFoundError) as caught:
        main(["run", "--model", "e4b", "--dataset", "wanli", "--workspace", str(workspace)])
    message = str(caught.value)
    assert "uv run --extra data gemmajev benchmark prepare --dataset wanli" in message
    assert "--workspace '" in message
    assert "download=True" not in message


def test_wrapper_change_invalidates_resume_fingerprint(tmp_path):
    from gemmajev.benchmarks.cli import SCORING_SOURCES, execution_fingerprint

    package = tmp_path / "package"
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "tasks.json").write_text('{"seed": 7}')
    for name in SCORING_SOURCES:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    weights, binary = tmp_path / "weights", tmp_path / "worker"
    weights.write_bytes(b"checkpoint")
    binary.write_bytes(b"runtime")
    original = execution_fingerprint(weights, binary, package_root=package, config_root=manifests)
    assert {"engine.py", "assets.py", "models.toml", "request.py"} <= original[
        "scoring_sources"
    ].keys()
    evaluate(
        [source()],
        model="e4b",
        workspace=tmp_path,
        output=tmp_path / "run",
        fingerprint=original,
        engine_factory=FakeEngine,
        cooldown=0,
    )
    (package / "engine.py").write_text("changed wrapper")
    updated = execution_fingerprint(weights, binary, package_root=package, config_root=manifests)
    assert updated["runtime_sha256"] == original["runtime_sha256"]
    assert updated["model_sha256"] == original["model_sha256"]
    with pytest.raises(ValueError, match="different protocol"):
        evaluate(
            [source()],
            model="e4b",
            workspace=tmp_path,
            output=tmp_path / "run",
            fingerprint=updated,
            engine_factory=FakeEngine,
            cooldown=0,
        )
    assert FakeEngine.calls == 1
