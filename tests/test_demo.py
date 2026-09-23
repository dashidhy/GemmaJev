"""Playground lifecycle and local HTTP behavior, without weights or GPU calls."""

import http.client
import json
import socket
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from typing import ClassVar

import pytest

from gemmajev.demo import (
    EXAMPLES,
    Playground,
    _engine_factory,
    create_server,
    load_example,
)


class FakeEngine:
    instances: ClassVar[list] = []

    def __init__(self, model, workspace):
        self.model = model
        self.loaded = False
        self.closed = False
        self.calls = 0
        self.load_calls = 0
        self.instances.append(self)

    def load(self):
        self.load_calls += 1
        assert not any(item.loaded for item in self.instances if item is not self)
        assert not self.closed
        self.loaded = True

    def decide(self, request):
        assert self.loaded and not self.closed
        self.calls += 1
        ids = list(request["options"])
        return {key: (0.8 if index == 0 else 0.2 / (len(ids) - 1)) for index, key in enumerate(ids)}

    def close(self):
        self.closed = True
        self.loaded = False


@pytest.fixture
def manager():
    FakeEngine.instances = []
    value = Playground(engine_factory=FakeEngine)
    yield value
    value.close()


@pytest.fixture
def server(manager):
    value = create_server(port=0, manager=manager)
    value.server_activate()
    thread = threading.Thread(target=value.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield value
    finally:
        value.shutdown()
        value.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def exchange(server, method="GET", path="/api/config", *, payload=None, body=None, headers=None):
    """Use a real local socket, not a handler mock or third-party HTTP client."""
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        content = response.read()
        response_headers = dict(response.getheaders())
        if response.getheader("Content-Type", "").startswith("application/json"):
            content = json.loads(content)
        return response.status, response_headers, content
    finally:
        connection.close()


def decision_payload(model="12b"):
    return {"model": model, "request": load_example("customer_support")}


def test_packaged_examples_have_semantic_ids_and_self_consistent_terms():
    for name in EXAMPLES:
        request = load_example(name)
        assert set(request) == {"task", "state", "query", "options"}
        assert len(request["options"]) == 3
        assert all(len(option_id) > 1 for option_id in request["options"])
    evidence = load_example("evidence_check")
    assert evidence["state"].startswith("Evidence:")
    assert evidence["query"].startswith("Claim:")
    assert "evidence" in evidence["task"] and "claim" in evidence["task"]


def test_demo_uses_smaller_context_without_changing_injected_factory(monkeypatch):
    import gemmajev.engine

    monkeypatch.setattr(gemmajev.engine, "GemmaJev", lambda **kwargs: kwargs)
    assert _engine_factory(model="12b", workspace=None) == {
        "model": "12b",
        "workspace": None,
        "context_size": 2048,
    }


def test_model_reuse_switch_and_failure_release(manager):
    request = load_example("customer_support")
    first = manager.run("12b", request)
    second = manager.run("12b", request)
    assert len(FakeEngine.instances) == 1
    assert FakeEngine.instances[0].calls == 2
    assert first["load_ms"] >= 0 and second["load_ms"] == 0
    manager.switch("e4b")
    assert FakeEngine.instances[0].closed
    assert len(FakeEngine.instances) == 2
    active = FakeEngine.instances[-1]
    assert active.loaded and active.load_calls == 1 and active.calls == 0
    manager.switch("e4b")
    result = manager.run("e4b", request)
    assert result["load_ms"] == 0 and active.load_calls == 1 and active.calls == 1
    assert len(FakeEngine.instances) == 2
    active.decide = lambda _: {"unrelated": 1.0}
    with pytest.raises(RuntimeError, match="invalid probability"):
        manager.run("e4b", request)
    assert active.closed
    manager.run("e4b", request)
    assert len(FakeEngine.instances) == 3


def test_preloading_is_idempotent_and_first_decision_reuses_the_engine(manager):
    assert manager.load("12b") is None
    manager.load("12b")
    assert len(FakeEngine.instances) == 1
    engine = FakeEngine.instances[0]
    assert engine.loaded and engine.load_calls == 1 and engine.calls == 0

    request = load_example("customer_support")
    result = manager.run("12b", request)
    assert result["load_ms"] == 0
    assert engine.load_calls == 1 and engine.calls == 1
    assert len(FakeEngine.instances) == 1
    request["state"] = "Changed input"
    request["options"]["delivery"] = "Changed meaning"
    assert result["request"] == load_example("customer_support")


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_preload_releases_partial_engine_and_can_retry(manager, monkeypatch, error_type):
    original_load = FakeEngine.load

    def fail_load(engine):
        engine.loaded = True  # Simulate native resources allocated before failure.
        raise error_type("Loading interrupted")

    monkeypatch.setattr(FakeEngine, "load", fail_load)
    with pytest.raises(error_type, match="Loading interrupted"):
        manager.load("12b")
    assert len(FakeEngine.instances) == 1
    assert FakeEngine.instances[0].closed and not FakeEngine.instances[0].loaded

    monkeypatch.setattr(FakeEngine, "load", original_load)
    manager.load("12b")
    assert len(FakeEngine.instances) == 2
    assert FakeEngine.instances[1].loaded and FakeEngine.instances[1].calls == 0


def test_switch_waits_for_current_request_before_releasing(manager):
    request = load_example("customer_support")
    manager.run("12b", request)
    engine = FakeEngine.instances[0]
    entered = threading.Event()
    release = threading.Event()
    finished_switch = threading.Event()
    original = engine.decide

    def decide(value):
        entered.set()
        assert release.wait(timeout=3)
        assert not engine.closed
        return original(value)

    def switch():
        manager.switch("e4b")
        finished_switch.set()

    engine.decide = decide
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(manager.run, "12b", request)
        assert entered.wait(timeout=3)
        switching = pool.submit(switch)
        try:
            assert not finished_switch.wait(timeout=0.05)
        finally:
            release.set()
        running.result(timeout=3)
        switching.result(timeout=3)
    assert engine.closed and finished_switch.is_set()


def test_server_binds_before_loading_but_does_not_listen(manager):
    server = create_server(model="e4b", port=0, manager=manager)
    try:
        assert server.manager is manager and server.model == "e4b"
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[1] > 0
        assert FakeEngine.instances == []
        with pytest.raises(OSError):
            socket.create_connection(server.server_address, timeout=0.2)
    finally:
        server.server_close()


def fake_launch_server(events, *, load_error=None):
    def load(model):
        events.append(("load", model))
        if load_error is not None:
            raise load_error("Loading interrupted")

    return SimpleNamespace(
        manager=SimpleNamespace(load=load, close=lambda: events.append("engine-close")),
        server_address=("127.0.0.1", 7862),
        server_port=7862,
        model="e4b",
        server_activate=lambda: events.append("listen"),
        serve_forever=lambda **kwargs: events.append("serve"),
        server_close=lambda: events.append("server-close"),
    )


@pytest.mark.parametrize("open_browser", [False, True])
def test_launch_loads_before_listening_and_opening_browser(monkeypatch, open_browser):
    from gemmajev import demo

    events = []
    server = fake_launch_server(events)

    def build_server(model, workspace, port):
        assert model == "e4b" and workspace == "/test/workspace" and port == 7862
        events.append("bind")
        return server

    monkeypatch.setattr(demo, "create_server", build_server)
    monkeypatch.setattr(webbrowser, "open", lambda url: events.append(("browser", url)))
    demo.launch(
        model="e4b", workspace="/test/workspace", port=7862, open_browser=open_browser,
    )
    expected = ["bind", ("load", "e4b"), "listen"]
    if open_browser:
        expected.append(("browser", "http://127.0.0.1:7862"))
    expected.extend(["serve", "server-close", "engine-close"])
    assert events == expected


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_launch_load_failure_never_listens_or_opens_browser(monkeypatch, error_type):
    from gemmajev import demo

    events = []
    server = fake_launch_server(events, load_error=error_type)
    monkeypatch.setattr(demo, "create_server", lambda *args, **kwargs: server)
    monkeypatch.setattr(webbrowser, "open", lambda url: events.append("browser"))
    expected_error = (
        nullcontext() if error_type is KeyboardInterrupt
        else pytest.raises(error_type, match="Loading interrupted")
    )
    with expected_error:
        demo.launch(model="12b")
    assert events == [("load", "12b"), "server-close", "engine-close"]


def test_config_exposes_examples_without_inference(server):
    status, _, config = exchange(server)
    assert status == 200 and config["model"] == "12b"
    assert config["examples"] == [
        {"id": name, "title": title, "request": load_example(name)}
        for name, title in EXAMPLES.items()
    ]
    assert FakeEngine.instances == []


def test_static_files_are_packaged_and_arbitrary_paths_are_not_served(server):
    for path, content_type in [
        ("/", "text/html"), ("/app.js", "javascript"), ("/style.css", "text/css"),
    ]:
        status, headers, body = exchange(server, path=path)
        assert status == 200 and body
        assert content_type in headers["Content-Type"]
    for path in [
        "/README.md", "/../pyproject.toml", "/%2e%2e/pyproject.toml",
        "/.models/anything", "/api/not-a-route",
    ]:
        status, _, _ = exchange(server, path=path)
        assert status == 404
    assert FakeEngine.instances == []


def test_decision_uses_preloaded_model_preserves_option_order_and_switches(server, manager):
    manager.load("12b")
    payload = decision_payload()
    payload["request"]["options"] = dict(reversed(list(payload["request"]["options"].items())))
    status, _, result = exchange(server, "POST", "/api/decide", payload=payload)
    assert status == 200 and result["request"] == payload["request"]
    assert list(result["probabilities"]) == list(payload["request"]["options"])
    assert result["model"] == "12b" and result["load_ms"] == 0 and result["request_ms"] >= 0
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    first = FakeEngine.instances[0]
    assert first.load_calls == 1 and first.calls == 1

    status, _, result = exchange(server, "POST", "/api/model", payload={"model": "e4b"})
    assert status == 200 and result == {"model": "e4b"}
    assert first.closed and len(FakeEngine.instances) == 2
    selected = FakeEngine.instances[-1]
    assert selected.loaded and selected.load_calls == 1 and selected.calls == 0
    assert exchange(server)[2]["model"] == "e4b"
    status, _, result = exchange(server, "POST", "/api/decide", payload=decision_payload("e4b"))
    assert status == 200 and result["model"] == "e4b" and result["load_ms"] == 0
    assert len(FakeEngine.instances) == 2 and selected.calls == 1 and selected.load_calls == 1


def test_invalid_requests_never_load_model_and_later_valid_request_succeeds(server):
    bad_request = {**load_example("customer_support"), "task": ""}
    cases = [
        ("/api/decide", {}),
        ("/api/decide", []),
        ("/api/decide", {"model": "unknown", "request": load_example("customer_support")}),
        ("/api/decide", {"model": "12b", "request": bad_request}),
        ("/api/decide", {"model": "12b", "request": {
            **load_example("customer_support"), "options": {"single": "Only option"},
        }}),
        ("/api/model", {"model": "unknown"}),
    ]
    for path, payload in cases:
        status, _, content = exchange(server, "POST", path, payload=payload)
        assert status == 400 and isinstance(content["error"], str) and content["error"]
    malformed_bodies = [
        b"not JSON",
        b'{"model":"e4b","model":"12b"}',
        b'{"model":"12b","request":{"options":{"same":"First","same":"Second"}}}',
        b"\xff",
    ]
    for body in malformed_bodies:
        status, _, content = exchange(
            server, "POST", "/api/decide", body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 400 and content["error"]
    assert FakeEngine.instances == []
    assert exchange(server, "POST", "/api/decide", payload=decision_payload())[0] == 200


def test_non_json_and_oversized_requests_are_rejected_without_loading(server):
    body = json.dumps(decision_payload()).encode()
    for content_type in ["text/plain", "application/x-www-form-urlencoded"]:
        status, _, content = exchange(
            server, "POST", "/api/decide", body=body,
            headers={"Content-Type": content_type},
        )
        assert status == 415 and content["error"]
    status, _, content = exchange(
        server, "POST", "/api/decide", body=b" " * (256 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert status == 413 and content["error"]
    assert FakeEngine.instances == []


def test_foreign_hosts_and_cross_origin_requests_cannot_run_model(server):
    port = server.server_address[1]
    for headers in [
        {"Host": "attacker.invalid"},
        {"Origin": "https://attacker.invalid"},
        {"Host": f"127.0.0.1:{port}", "Origin": f"http://localhost:{port}"},
        {"Origin": "null"},
    ]:
        status, _, content = exchange(
            server, "POST", "/api/decide", payload=decision_payload(), headers=headers,
        )
        assert status == 403 and content["error"]
    assert exchange(server, headers={"Host": "attacker.invalid"})[0] == 403
    assert FakeEngine.instances == []
    for hostname in ["127.0.0.1", "localhost"]:
        headers = {"Host": f"{hostname}:{port}", "Origin": f"http://{hostname}:{port}"}
        assert exchange(server, "POST", "/api/decide", payload=decision_payload(),
                        headers=headers)[0] == 200


def test_runtime_failure_releases_engine_and_next_request_recovers(server, manager):
    manager.load("12b")
    failed = FakeEngine.instances[0]
    payload = decision_payload()
    payload["request"]["state"] = "long input " * 1000
    received = []

    def fail(request):
        received.append(request)
        raise RuntimeError("Native inference failed: input exceeds the context window")

    failed.decide = fail
    status, _, content = exchange(server, "POST", "/api/decide", payload=payload)
    assert status == 503 and "2,048" in content["error"]
    assert "nothing was truncated" in content["error"].lower()
    assert received == [payload["request"]]  # Never silently truncate input to make it fit.
    assert failed.closed
    assert exchange(server, "POST", "/api/decide", payload=decision_payload())[0] == 200
    assert len(FakeEngine.instances) == 2


def test_missing_model_is_reported_as_unavailable(server, monkeypatch):
    def missing(engine):
        raise FileNotFoundError("Model file is missing. Run gemmajev setup --model 12b.")

    monkeypatch.setattr(FakeEngine, "load", missing)
    status, _, content = exchange(server, "POST", "/api/decide", payload=decision_payload())
    assert status == 503 and "setup" in content["error"]
    assert FakeEngine.instances[0].closed


def test_concurrent_decision_and_switch_are_rejected_instead_of_queued(server, manager):
    manager.load("12b")
    engine = FakeEngine.instances[0]
    entered = threading.Event()
    release = threading.Event()
    original = engine.decide

    def blocking(request):
        entered.set()
        assert release.wait(timeout=5)
        assert not engine.closed
        return original(request)

    engine.decide = blocking
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(exchange, server, "POST", "/api/decide", payload=decision_payload())
        assert entered.wait(timeout=3)
        try:
            for path, payload in [
                ("/api/decide", decision_payload()), ("/api/model", {"model": "e4b"}),
            ]:
                status, _, content = exchange(server, "POST", path, payload=payload)
                assert status == 409 and content["error"]
            assert exchange(server)[0] == 200  # Static/config reads remain responsive.
            assert not engine.closed
        finally:
            release.set()
        assert running.result(timeout=3)[0] == 200
    assert len(FakeEngine.instances) == 1 and engine.calls == 1
    assert exchange(server, "POST", "/api/model", payload={"model": "e4b"})[0] == 200
    assert engine.closed


def test_missing_packaged_asset_returns_error_without_loading(server, monkeypatch, tmp_path):
    from gemmajev import demo

    monkeypatch.setattr(demo, "files", lambda _: tmp_path)
    status, _, content = exchange(server, path="/app.js")
    assert status == 503 and content["error"]
    assert FakeEngine.instances == []


def test_switch_load_failure_returns_error_releases_resources_and_can_retry(
    server, manager, monkeypatch,
):
    manager.load("12b")
    previous = FakeEngine.instances[0]
    original_load = FakeEngine.load

    def fail_load(engine):
        assert previous.closed
        engine.loaded = True  # Loading may allocate native resources before failing.
        raise RuntimeError("Model loading failed.")

    monkeypatch.setattr(FakeEngine, "load", fail_load)
    status, _, content = exchange(server, "POST", "/api/model", payload={"model": "e4b"})
    assert status == 503 and "loading failed" in content["error"]
    failed = FakeEngine.instances[-1]
    assert previous.closed and failed.closed and not failed.loaded
    assert all(engine.calls == 0 for engine in FakeEngine.instances)
    assert exchange(server)[2]["model"] == "12b"  # Only successful switches change config.

    monkeypatch.setattr(FakeEngine, "load", original_load)
    status, _, content = exchange(server, "POST", "/api/model", payload={"model": "e4b"})
    assert status == 200 and content == {"model": "e4b"}
    selected = FakeEngine.instances[-1]
    assert len(FakeEngine.instances) == 3 and selected.loaded
    assert selected.load_calls == 1 and selected.calls == 0
    status, _, result = exchange(server, "POST", "/api/decide", payload=decision_payload("e4b"))
    assert status == 200 and result["load_ms"] == 0
    assert selected.calls == 1 and selected.load_calls == 1


def test_switch_responds_only_after_load_and_rejects_other_model_operations(
    server, manager, monkeypatch,
):
    manager.load("12b")
    previous = FakeEngine.instances[0]
    entered = threading.Event()
    release = threading.Event()
    original_load = FakeEngine.load

    def blocking_load(engine):
        assert previous.closed  # Release the old model before allocating the replacement.
        entered.set()
        assert release.wait(timeout=5)
        original_load(engine)

    monkeypatch.setattr(FakeEngine, "load", blocking_load)
    with ThreadPoolExecutor(max_workers=1) as pool:
        switching = pool.submit(exchange, server, "POST", "/api/model", payload={"model": "e4b"})
        assert entered.wait(timeout=3)
        try:
            assert not switching.done()
            assert not FakeEngine.instances[-1].loaded
            for path, payload in [
                ("/api/decide", decision_payload()), ("/api/model", {"model": "12b"}),
            ]:
                status, _, content = exchange(server, "POST", path, payload=payload)
                assert status == 409 and content["error"]
            status, _, config = exchange(server)
            assert status == 200 and config["model"] == "12b"
            assert not switching.done()
        finally:
            release.set()
        status, _, content = switching.result(timeout=3)
        assert status == 200 and content == {"model": "e4b"}
    selected = FakeEngine.instances[-1]
    assert selected.loaded and selected.load_calls == 1 and selected.calls == 0
    assert exchange(server)[2]["model"] == "e4b"
    status, _, result = exchange(server, "POST", "/api/decide", payload=decision_payload("e4b"))
    assert status == 200 and result["load_ms"] == 0
    assert selected.calls == 1 and selected.load_calls == 1
